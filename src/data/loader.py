"""Loader module for the Corporación Favorita (Store Sales) dataset.

Loads raw CSV files, validates schemas, caches as Parquet, and handles
common data quality issues (e.g., missing oil prices).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd
from loguru import logger
from pydantic import BaseModel, DirectoryPath, field_validator


# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------

_SCHEMAS: Final[dict[str, list[str]]] = {
    "train": [
        "id",
        "date",
        "store_nbr",
        "family",
        "sales",
        "onpromotion",
    ],
    "stores": ["store_nbr", "city", "state", "type", "cluster"],
    "oil": ["date", "dcoilwtico"],
    "holidays": [
        "date",
        "type",
        "locale",
        "locale_name",
        "description",
        "transferred",
    ],
    "transactions": ["date", "store_nbr", "transactions"],
}


# ---------------------------------------------------------------------------
# Pydantic config (validation at boundary)
# ---------------------------------------------------------------------------


class DataLoaderConfig(BaseModel):
    """Validated configuration for :class:`DataLoader`."""

    data_dir: DirectoryPath
    cache_dir: Path = Path("data/processed")

    @field_validator("cache_dir", mode="before")
    @classmethod
    def _resolve_cache_dir(cls, v: str | Path) -> Path:
        return Path(v)


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------


class DataLoader:
    """Loads and validates the raw Favorita dataset.

    Parameters
    ----------
    data_dir : Path
        Root directory that contains the raw CSV files
        (``train.csv``, ``stores.csv``, …).
    cache_dir : Path, optional
        Directory for Parquet caches.  Defaults to ``data/processed/``.
    """

    # Mapping from logical name → CSV filename
    _FILE_MAP: Final[dict[str, str]] = {
        "train": "train.csv",
        "stores": "stores.csv",
        "oil": "oil.csv",
        "holidays": "holidays_events.csv",
        "transactions": "transactions.csv",
    }

    _DATE_COLUMNS: Final[dict[str, str | list[str]]] = {
        "train": "date",
        "oil": "date",
        "holidays": "date",
        "transactions": "date",
    }

    def __init__(
        self,
        data_dir: Path,
        cache_dir: Path = Path("data/processed"),
    ) -> None:
        cfg = DataLoaderConfig(data_dir=data_dir, cache_dir=cache_dir)
        self._data_dir: Path = cfg.data_dir
        self._cache_dir: Path = cfg.cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "DataLoader initialised  ·  data_dir={dir}  ·  cache_dir={cache}",
            dir=self._data_dir,
            cache=self._cache_dir,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_train(self) -> pd.DataFrame:
        """Load ``train.csv`` with schema validation."""
        return self._load("train")

    def load_stores(self) -> pd.DataFrame:
        """Load ``stores.csv`` with schema validation."""
        return self._load("stores")

    def load_oil(self) -> pd.DataFrame:
        """Load ``oil.csv``, forward-filling missing prices."""
        df = self._load("oil")
        missing_before: int = int(df["dcoilwtico"].isna().sum())
        if missing_before > 0:
            logger.warning(
                "Oil prices contain {n} missing values — will be handled downstream",
                n=missing_before,
            )
        return df

    def load_holidays(self) -> pd.DataFrame:
        """Load ``holidays_events.csv`` with schema validation."""
        return self._load("holidays")

    def load_transactions(self) -> pd.DataFrame:
        """Load ``transactions.csv`` with schema validation."""
        return self._load("transactions")

    def load_all(self) -> dict[str, pd.DataFrame]:
        """Load **every** dataset and return as a name → DataFrame mapping.

        Returns
        -------
        dict[str, pd.DataFrame]
            Keys: ``"train"``, ``"stores"``, ``"oil"``,
            ``"holidays"``, ``"transactions"``.
        """
        datasets: dict[str, pd.DataFrame] = {
            "train": self.load_train(),
            "stores": self.load_stores(),
            "oil": self.load_oil(),
            "holidays": self.load_holidays(),
            "transactions": self.load_transactions(),
        }
        logger.success("All datasets loaded successfully")
        return datasets

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self, name: str) -> pd.DataFrame:
        """Load a single dataset with caching and validation.

        1. Try reading from Parquet cache.
        2. Fall back to CSV.
        3. Validate schema.
        4. Write Parquet cache for next time.
        """
        cache_path = self._cache_dir / f"{name}.parquet"

        if cache_path.exists():
            logger.debug("Loading {name} from cache: {path}", name=name, path=cache_path)
            df = pd.read_parquet(cache_path)
            self._validate_schema(df, _SCHEMAS[name], name)
            return df

        csv_path = self._data_dir / self._FILE_MAP[name]
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Expected CSV file not found: {csv_path}"
            )

        logger.info("Reading {name} from CSV: {path}", name=name, path=csv_path)

        parse_dates: list[str] | bool = (
            [self._DATE_COLUMNS[name]]
            if name in self._DATE_COLUMNS and isinstance(self._DATE_COLUMNS[name], str)
            else False
        )

        df = pd.read_csv(csv_path, parse_dates=parse_dates)  # type: ignore[arg-type]

        self._validate_schema(df, _SCHEMAS[name], name)

        # Cache as Parquet for fast subsequent loads
        df.to_parquet(cache_path, index=False)
        logger.info(
            "Cached {name} → {path}  ({rows:,} rows × {cols} cols)",
            name=name,
            path=cache_path,
            rows=len(df),
            cols=len(df.columns),
        )

        return df

    @staticmethod
    def _validate_schema(
        df: pd.DataFrame,
        expected_columns: list[str],
        name: str,
    ) -> None:
        """Validate that *df* contains all *expected_columns*.

        Raises
        ------
        ValueError
            If any expected column is missing from the DataFrame.
        """
        actual: set[str] = set(df.columns)
        missing: set[str] = set(expected_columns) - actual
        if missing:
            raise ValueError(
                f"Schema validation failed for '{name}': "
                f"missing columns {sorted(missing)}. "
                f"Available columns: {sorted(actual)}"
            )
        logger.debug(
            "Schema OK for '{name}' — {n} expected columns present",
            name=name,
            n=len(expected_columns),
        )
