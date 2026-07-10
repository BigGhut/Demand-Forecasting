"""Preprocessing module for the Corporación Favorita dataset.

Cleans, merges, aggregates, and fills the raw DataFrames produced by
:class:`~src.data.loader.DataLoader` so they are ready for feature
engineering and modelling.
"""

from __future__ import annotations

from typing import Final, Literal

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel, field_validator


# ---------------------------------------------------------------------------
# Pydantic config (validation at boundary)
# ---------------------------------------------------------------------------

AggregationLevel = Literal["store_family", "family", "store", "total"]


class PreprocessorConfig(BaseModel):
    """Validated configuration for :class:`DataPreprocessor`."""

    aggregation_level: AggregationLevel = "store_family"

    @field_validator("aggregation_level", mode="before")
    @classmethod
    def _validate_level(cls, v: str) -> str:
        allowed: set[str] = {"store_family", "family", "store", "total"}
        if v not in allowed:
            raise ValueError(
                f"aggregation_level must be one of {sorted(allowed)}, got '{v}'"
            )
        return v


# ---------------------------------------------------------------------------
# Aggregation key look-up
# ---------------------------------------------------------------------------

_AGG_KEYS: Final[dict[str, list[str]]] = {
    "store_family": ["date", "store_nbr", "family"],
    "family": ["date", "family"],
    "store": ["date", "store_nbr"],
    "total": ["date"],
}


# ---------------------------------------------------------------------------
# Main preprocessor
# ---------------------------------------------------------------------------


class DataPreprocessor:
    """Cleans, aggregates, and prepares raw data for modelling.

    Parameters
    ----------
    aggregation_level : AggregationLevel
        Granularity of the output DataFrame.

        * ``"store_family"`` — one row per (date, store, product family)
        * ``"family"``       — one row per (date, product family)
        * ``"store"``        — one row per (date, store)
        * ``"total"``        — one row per date
    """

    def __init__(
        self,
        aggregation_level: AggregationLevel = "store_family",
    ) -> None:
        cfg = PreprocessorConfig(aggregation_level=aggregation_level)
        self._agg_level: AggregationLevel = cfg.aggregation_level
        self._agg_keys: list[str] = _AGG_KEYS[self._agg_level]
        logger.info(
            "DataPreprocessor initialised  ·  aggregation_level={lvl}",
            lvl=self._agg_level,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(self, datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Full preprocessing pipeline: clean → merge → aggregate → fill → cast.

        Parameters
        ----------
        datasets : dict[str, pd.DataFrame]
            Output of :meth:`DataLoader.load_all`.

        Returns
        -------
        pd.DataFrame
            Clean, merged, aggregated DataFrame ready for feature engineering.
        """
        logger.info("Starting preprocessing pipeline …")

        train = datasets["train"].copy()
        stores = datasets["stores"].copy()
        oil = datasets["oil"].copy()
        holidays = datasets["holidays"].copy()
        transactions = datasets["transactions"].copy()

        train = self._clean_sales(train)
        merged = self._merge_external(train, stores, oil, holidays, transactions)
        aggregated = self._aggregate(merged)
        filled = self._fill_missing(aggregated)
        result = self._cast_types(filled)

        logger.success(
            "Preprocessing complete — {rows:,} rows × {cols} columns",
            rows=len(result),
            cols=len(result.columns),
        )
        return result

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def _clean_sales(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle negative sales and ensure the date column is datetime.

        * Negative ``sales`` values are clipped to 0.
        * The ``date`` column is parsed to ``datetime64[ns]``.
        """
        df = df.copy()

        # Ensure datetime
        df["date"] = pd.to_datetime(df["date"])

        # Clip negative sales to 0 (vectorised)
        neg_count: int = int((df["sales"] < 0).sum())
        if neg_count > 0:
            logger.warning(
                "Clipping {n:,} negative sales values to 0", n=neg_count
            )
            df["sales"] = df["sales"].clip(lower=0)

        logger.debug("Sales cleaned — {rows:,} rows", rows=len(df))
        return df

    def _merge_external(
        self,
        sales: pd.DataFrame,
        stores: pd.DataFrame,
        oil: pd.DataFrame,
        holidays: pd.DataFrame,
        transactions: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge all auxiliary datasets onto the sales table.

        Join keys
        ---------
        * ``stores``       → ``store_nbr``
        * ``oil``          → ``date``
        * ``holidays``     → ``date``
        * ``transactions`` → ``(date, store_nbr)``
        """
        df = sales.copy()

        # --- stores (store metadata) ---
        df = df.merge(stores, on="store_nbr", how="left")
        logger.debug("Merged stores metadata — {cols} new columns",
                      cols=len(stores.columns) - 1)

        # --- oil ---
        oil = oil.copy()
        oil["date"] = pd.to_datetime(oil["date"])
        df = df.merge(oil, on="date", how="left")
        logger.debug("Merged oil prices")

        # --- holidays ---
        holidays = holidays.copy()
        holidays["date"] = pd.to_datetime(holidays["date"])
        # Avoid column-name collision: rename 'type' from holidays
        holidays = holidays.rename(columns={"type": "holiday_type"})
        
        # Merge holidays (may temporarily duplicate rows if multiple events on the same day)
        df = df.merge(holidays, on="date", how="left")
        
        # Determine if the holiday is active for the store's location
        is_active_holiday = (
            df["holiday_type"].notna()
            & (df["transferred"] != True)
            & (
                (df["locale"] == "National")
                | ((df["locale"] == "Regional") & (df["locale_name"] == df["state"]))
                | ((df["locale"] == "Local") & (df["locale_name"] == df["city"]))
            )
        )
        
        # Reset holiday fields to NaN for non-active holidays
        holiday_cols = ["holiday_type", "locale", "locale_name", "description", "transferred"]
        df.loc[~is_active_holiday, holiday_cols] = np.nan
        df["is_localized_holiday"] = is_active_holiday.astype(np.int8)
        
        # Deduplicate to restore original rows
        if "id" in df.columns:
            df = df.sort_values(by=["id", "is_localized_holiday"], ascending=[True, False])
            df = df.drop_duplicates(subset=["id"], keep="first")
        else:
            # Fallback to date, store_nbr, family
            df = df.sort_values(by=["date", "store_nbr", "family", "is_localized_holiday"], ascending=[True, True, True, False])
            df = df.drop_duplicates(subset=["date", "store_nbr", "family"], keep="first")
            
        logger.debug("Merged holidays with localized mapping")

        # --- transactions ---
        transactions = transactions.copy()
        transactions["date"] = pd.to_datetime(transactions["date"])
        df = df.merge(transactions, on=["date", "store_nbr"], how="left")
        logger.debug("Merged transactions")

        logger.info(
            "External merge done — shape {shape}", shape=df.shape
        )
        return df

    def _aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate to the configured granularity level.

        Numeric columns (``sales``, ``onpromotion``, ``transactions``,
        ``dcoilwtico``) are summed or averaged as appropriate.
        """
        if self._agg_level == "store_family":
            # Already at finest useful level — no aggregation needed
            logger.debug("Aggregation level is store_family; skipping groupby")
            return df

        agg_spec: dict[str, str] = {
            "sales": "sum",
            "onpromotion": "sum",
        }

        # Only aggregate columns that exist
        if "transactions" in df.columns:
            agg_spec["transactions"] = "sum"
        if "dcoilwtico" in df.columns:
            agg_spec["dcoilwtico"] = "mean"

        result = (
            df.groupby(self._agg_keys, observed=True)
            .agg(agg_spec)
            .reset_index()
        )

        logger.info(
            "Aggregated to '{lvl}' — {rows:,} rows",
            lvl=self._agg_level,
            rows=len(result),
        )
        return result

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill oil prices and zero-fill sales for missing dates.

        Creates a *complete* date range per group key (store, family, …)
        so that every combination has a contiguous daily time series.
        """
        df = df.copy()

        # --- Forward-fill oil prices (gaps on weekends / holidays) ---
        if "dcoilwtico" in df.columns:
            oil_missing_before: int = int(df["dcoilwtico"].isna().sum())
            df = df.sort_values("date")
            df["dcoilwtico"] = df["dcoilwtico"].ffill()
            # Back-fill any leading NaNs
            df["dcoilwtico"] = df["dcoilwtico"].bfill()
            oil_missing_after: int = int(df["dcoilwtico"].isna().sum())
            logger.info(
                "Oil prices filled: {before:,} → {after:,} missing",
                before=oil_missing_before,
                after=oil_missing_after,
            )

        # --- Complete date range per group ---
        group_keys: list[str] = [
            k for k in self._agg_keys if k != "date"
        ]

        if group_keys:
            date_range = pd.date_range(
                start=df["date"].min(),
                end=df["date"].max(),
                freq="D",
            )

            # Build a full index via cross-join with unique group combos
            groups = df[group_keys].drop_duplicates()
            full_idx = groups.assign(_key=1).merge(
                pd.DataFrame({"date": date_range, "_key": 1}),
                on="_key",
            ).drop(columns="_key")

            df = full_idx.merge(df, on=["date"] + group_keys, how="left")

            # Zero-fill sales & onpromotion for newly created rows
            df["sales"] = df["sales"].fillna(0)
            if "onpromotion" in df.columns:
                df["onpromotion"] = df["onpromotion"].fillna(0)
            if "transactions" in df.columns:
                df["transactions"] = df["transactions"].fillna(0)

            # Re-forward-fill oil after expansion
            if "dcoilwtico" in df.columns:
                df = df.sort_values("date")
                df["dcoilwtico"] = df["dcoilwtico"].ffill().bfill()

            logger.info(
                "Date range completed — {rows:,} rows after fill",
                rows=len(df),
            )

        return df

    def _cast_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert columns to efficient dtypes.

        * ``date`` → ``datetime64[ns]``
        * Category-like string columns → ``pd.Categorical``
        * Numeric columns → appropriate numeric types
        """
        df = df.copy()

        df["date"] = pd.to_datetime(df["date"])

        category_cols: list[str] = [
            "family",
            "city",
            "state",
            "type",
            "holiday_type",
            "locale",
            "locale_name",
            "transferred",
        ]

        for col in category_cols:
            if col in df.columns:
                df[col] = df[col].astype("category")

        # Integer-safe casting for numeric IDs
        int_cols: list[str] = ["store_nbr", "cluster"]
        for col in int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(
                    pd.Int32Dtype()
                )

        float_cols: list[str] = ["sales", "onpromotion", "dcoilwtico", "transactions"]
        for col in float_cols:
            if col in df.columns:
                df[col] = df[col].astype(np.float32)

        logger.debug("Type casting complete")
        return df
