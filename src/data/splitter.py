"""Chronological train / validation / test splitting for time series.

Ensures strict temporal ordering — **no random shuffling** — and provides
both a single split and expanding-window cross-validation splits.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger
from pydantic import BaseModel, field_validator


# ---------------------------------------------------------------------------
# Pydantic config (validation at boundary)
# ---------------------------------------------------------------------------


class SplitterConfig(BaseModel):
    """Validated configuration for :class:`TimeSeriesSplitter`."""

    forecast_horizon: int = 28
    validation_size: int = 28

    @field_validator("forecast_horizon", "validation_size")
    @classmethod
    def _must_be_positive(cls, v: int, info: object) -> int:  # noqa: ANN001
        if v <= 0:
            raise ValueError(
                f"{getattr(info, 'field_name', 'value')} must be > 0, got {v}"
            )
        return v


# ---------------------------------------------------------------------------
# Main splitter
# ---------------------------------------------------------------------------


class TimeSeriesSplitter:
    """Chronological splitting for time series data.

    Parameters
    ----------
    forecast_horizon : int
        Number of days in the **test** set (default 28).
    validation_size : int
        Number of days in the **validation** set (default 28).
    """

    def __init__(
        self,
        forecast_horizon: int = 28,
        validation_size: int = 28,
    ) -> None:
        cfg = SplitterConfig(
            forecast_horizon=forecast_horizon,
            validation_size=validation_size,
        )
        self._forecast_horizon: int = cfg.forecast_horizon
        self._validation_size: int = cfg.validation_size
        logger.info(
            "TimeSeriesSplitter initialised  ·  "
            "forecast_horizon={h}  ·  validation_size={v}",
            h=self._forecast_horizon,
            v=self._validation_size,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(
        self,
        df: pd.DataFrame,
        date_column: str = "date",
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split *df* chronologically into train / val / test.

        * **test**  — last ``forecast_horizon`` unique dates.
        * **val**   — previous ``validation_size`` unique dates before test.
        * **train** — everything before val.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame (must contain *date_column*).
        date_column : str
            Name of the date column.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            ``(train, val, test)`` — each a **copy** of the relevant slice.

        Raises
        ------
        ValueError
            If the date column is not sorted or if the data is too short.
        """
        self._validate_date_column(df, date_column)

        unique_dates: pd.DatetimeIndex = pd.DatetimeIndex(
            df[date_column].unique()
        ).sort_values()

        required_days: int = self._forecast_horizon + self._validation_size + 1
        if len(unique_dates) < required_days:
            raise ValueError(
                f"Not enough data for split: need ≥ {required_days} unique dates, "
                f"got {len(unique_dates)}"
            )

        test_start = unique_dates[-self._forecast_horizon]
        val_start = unique_dates[-(self._forecast_horizon + self._validation_size)]

        train_mask = df[date_column] < val_start
        val_mask = (df[date_column] >= val_start) & (df[date_column] < test_start)
        test_mask = df[date_column] >= test_start

        train = df.loc[train_mask].copy()
        val = df.loc[val_mask].copy()
        test = df.loc[test_mask].copy()

        logger.info(
            "Split complete ·  train: {t0} → {t1} ({tn:,} rows)  |  "
            "val: {v0} → {v1} ({vn:,} rows)  |  "
            "test: {s0} → {s1} ({sn:,} rows)",
            t0=train[date_column].min().date(),
            t1=train[date_column].max().date(),
            tn=len(train),
            v0=val[date_column].min().date(),
            v1=val[date_column].max().date(),
            vn=len(val),
            s0=test[date_column].min().date(),
            s1=test[date_column].max().date(),
            sn=len(test),
        )

        return train, val, test

    def get_cv_splits(
        self,
        df: pd.DataFrame,
        n_windows: int = 5,
        stride: int = 14,
        date_column: str = "date",
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate expanding-window cross-validation splits.

        The most recent window ends at the **last** date in *df*.
        Earlier windows are shifted back by *stride* days each.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        n_windows : int
            Number of CV folds to create (default 5).
        stride : int
            Number of days between successive test-set start dates
            (default 14).
        date_column : str
            Name of the date column.

        Returns
        -------
        list[tuple[pd.DataFrame, pd.DataFrame]]
            List of ``(train, test)`` pairs — each element is a **copy**.

        Raises
        ------
        ValueError
            If the date column is unsorted or the data is too short for the
            requested number of windows.
        """
        self._validate_date_column(df, date_column)

        unique_dates: pd.DatetimeIndex = pd.DatetimeIndex(
            df[date_column].unique()
        ).sort_values()

        # Minimum dates needed:
        #   (n_windows - 1) * stride + forecast_horizon + 1  (for earliest train)
        min_required: int = (n_windows - 1) * stride + self._forecast_horizon + 1
        if len(unique_dates) < min_required:
            raise ValueError(
                f"Not enough data for {n_windows} CV windows with "
                f"stride={stride}: need ≥ {min_required} unique dates, "
                f"got {len(unique_dates)}"
            )

        splits: list[tuple[pd.DataFrame, pd.DataFrame]] = []

        for i in range(n_windows):
            # Offset from the end: window 0 is the most recent
            offset: int = (n_windows - 1 - i) * stride
            test_end_idx: int = len(unique_dates) - offset
            test_start_idx: int = test_end_idx - self._forecast_horizon

            test_start = unique_dates[test_start_idx]
            test_end = unique_dates[test_end_idx - 1]  # inclusive

            train_mask = df[date_column] < test_start
            test_mask = (df[date_column] >= test_start) & (
                df[date_column] <= test_end
            )

            train = df.loc[train_mask].copy()
            test = df.loc[test_mask].copy()

            splits.append((train, test))

            logger.info(
                "CV window {i}/{n}  ·  train: → {te}  ({tn:,} rows)  |  "
                "test: {ts} → {td}  ({tsn:,} rows)",
                i=i + 1,
                n=n_windows,
                te=(test_start - pd.Timedelta(days=1)).date(),
                tn=len(train),
                ts=test_start.date(),
                td=test_end.date(),
                tsn=len(test),
            )

        return splits

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_date_column(
        df: pd.DataFrame,
        date_column: str,
    ) -> None:
        """Ensure *date_column* exists and is sorted.

        Raises
        ------
        ValueError
            If the column is missing or not sorted.
        """
        if date_column not in df.columns:
            raise ValueError(
                f"Date column '{date_column}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

        dates = pd.to_datetime(df[date_column])
        if not dates.is_monotonic_increasing:
            raise ValueError(
                f"Date column '{date_column}' is not sorted in ascending order. "
                "Sort the DataFrame before splitting."
            )
