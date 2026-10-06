from __future__ import annotations

import re

import numpy as np
import pandas as pd
from loguru import logger

from src.utils.config import FeatureConfig, get_feature_config
from src.features.calendar_features import CalendarFeatureGenerator
from src.features.external_features import ExternalFeatureGenerator
from src.features.lag_features import LagFeatureGenerator
from src.features.rolling_features import RollingFeatureGenerator
from src.features.expanding_features import ExpandingFeatureGenerator

# lag_N, oil_lag_N, transactions_lag_N, rolling_<stat>_lagN_wW.
# Promotion lags are excluded: the promo calendar is known ahead of time.
_FEATURE_LAG = re.compile(
    r"^(?:lag|oil_lag|transactions_lag)_(\d+)$|^rolling_[A-Za-z]+_lag(\d+)_w\d+$"
)


def _feature_lag(name: str) -> int | None:
    match = _FEATURE_LAG.match(name)
    if match is None:
        return None
    raw = match.group(1) or match.group(2)
    return int(raw)


class FeatureEngineeringPipeline:
    """Orchestrates all feature generators in the correct order.
    
    Order is critical:
    1. Calendar features (no dependencies)
    2. External features (no dependencies on other features)
    3. Lag features (depends on target column)
    4. Rolling features (depends on lag columns)
    5. Expanding features (depends on target column, uses lag approach)
    """
    
    def __init__(self, feature_config: FeatureConfig | None = None,
                 forecast_horizon: int = 28) -> None:
        self.config = feature_config or get_feature_config()
        self.forecast_horizon = forecast_horizon
        self._leakage_errors: list[str] = []

        self.generators = {
            "calendar": CalendarFeatureGenerator(
                features=self.config.calendar.features,
                binary_features=self.config.calendar.binary
            ),
            "external": ExternalFeatureGenerator(
                oil_lags=self.config.external.oil_lags,
                include_promotion=self.config.external.include_promotion,
                include_transactions=self.config.external.include_transactions,
                forecast_horizon=self.forecast_horizon,
            ),
            "lags": LagFeatureGenerator(
                lags=self.config.lags,
                forecast_horizon=self.forecast_horizon
            ),
            "rolling": RollingFeatureGenerator(
                windows=self.config.rolling.windows,
                functions=self.config.rolling.functions,
                base_lags=self.config.rolling.base_lags,
                forecast_horizon=self.forecast_horizon,
            ),
            "expanding": ExpandingFeatureGenerator(
                functions=self.config.expanding.functions,
                min_periods=self.config.expanding.min_periods,
                forecast_horizon=self.forecast_horizon,
            )
        }
        
    def fit_transform(self, df: pd.DataFrame, 
                       target_col: str = 'sales',
                       group_cols: list[str] | None = None,
                       date_col: str = 'date') -> pd.DataFrame:
        """Apply all feature generators and return enriched DataFrame."""
        return self.transform(df, target_col, group_cols, date_col, validate=True)
    
    def transform(self, df: pd.DataFrame, 
                       target_col: str = 'sales',
                       group_cols: list[str] | None = None,
                       date_col: str = 'date',
                       validate: bool = False) -> pd.DataFrame:
        """Transform data through the feature pipeline."""
        df_out = df.copy()
        
        # 1. Calendar
        df_out = self.generators["calendar"].transform(df_out, date_col=date_col)
        
        # 2. External
        df_out = self.generators["external"].transform(df_out, group_cols=group_cols)
        
        # 3. Lags
        df_out = self.generators["lags"].transform(df_out, target_col=target_col, group_cols=group_cols)
        
        # 4. Rolling
        df_out = self.generators["rolling"].transform(
            df_out, group_cols=group_cols, target_col=target_col
        )

        # 5. Expanding
        df_out = self.generators["expanding"].transform(df_out, target_col=target_col, group_cols=group_cols)

        if validate and not self.validate_no_leakage(df_out, target_col, date_col):
            detail = "; ".join(self._leakage_errors) or "min lag is inside the forecast horizon"
            raise ValueError(f"Leakage validation failed: {detail}")
            
        total_feats = sum(len(g.get_feature_names()) for g in self.generators.values())
        logger.info(f"Pipeline completed: {total_feats} total features generated.")
        
        return df_out
    
    def get_all_feature_names(self) -> dict[str, list[str]]:
        """Return feature names grouped by generator type."""
        return {
            name: gen.get_feature_names()
            for name, gen in self.generators.items()
        }

    def future_covariate_names(self) -> list[str]:
        """Columns known for the whole horizon: calendar, planned promos, lagged oil.

        Oil enters only when its lag is at least the horizon, so the value on
        the last forecast day is the oil price at the forecast origin.
        Transactions and target history are not included.
        """
        names = list(self.generators["calendar"].get_feature_names())
        for name in self.generators["external"].get_feature_names():
            if name.startswith("transactions_lag_"):
                continue
            lag = _feature_lag(name)
            if lag is not None and lag < self.forecast_horizon:
                continue
            names.append(name)
        return names

    def shifted_covariate_names(self) -> list[str]:
        """History already lagged by at least the horizon.

        Pass these to CatBoost as future covariates with lag 0. The value on
        a forecast date only depends on sales and traffic at or before the
        origin, and a raw column cannot be dropped into the same list.
        """
        names: list[str] = []
        for key in ("lags", "rolling", "expanding"):
            names.extend(self.generators[key].get_feature_names())
        for name in self.generators["external"].get_feature_names():
            if name.startswith("transactions_lag_"):
                names.append(name)
        return names

    def validate_no_leakage(self, df: pd.DataFrame, target_col: str,
                             date_col: str) -> bool:
        """Check that no modelled feature at time t reads sales or oil inside the horizon.

        Returns False when a check fails. ``fit_transform`` raises on that result.
        """
        errors: list[str] = []
        horizon = self.forecast_horizon

        lags_gen = self.generators["lags"]
        if min(lags_gen.lags) < horizon:
            errors.append(
                f"min target lag {min(lags_gen.lags)} < forecast_horizon {horizon}"
            )

        rolling = self.generators["rolling"]
        short_roll = [lag for lag in rolling.base_lags if lag < horizon]
        if short_roll:
            errors.append(f"rolling base lags {short_roll} < forecast_horizon {horizon}")

        expanding = self.generators["expanding"]
        if expanding.base_lag < horizon:
            errors.append(
                f"expanding base lag {expanding.base_lag} < forecast_horizon {horizon}"
            )

        external = self.generators["external"]
        short_oil = [lag for lag in external.oil_lags if lag < horizon]
        if short_oil:
            errors.append(f"oil lags {short_oil} < forecast_horizon {horizon}")
        if external.transactions_lag < horizon:
            errors.append(
                f"transactions lag {external.transactions_lag} < forecast_horizon {horizon}"
            )

        for name in (n for feats in self.get_all_feature_names().values() for n in feats):
            lag = _feature_lag(name)
            if lag is not None and lag < horizon:
                errors.append(f"{name} uses lag {lag} < forecast_horizon {horizon}")

        if (
            "expanding_mean" in df.columns
            and target_col in df.columns
            and date_col in df.columns
            and df[date_col].is_unique
        ):
            safe = (
                df[target_col]
                .shift(expanding.base_lag)
                .expanding(min_periods=expanding.min_periods)
                .mean()
            )
            got = df["expanding_mean"]
            mask = got.notna() & safe.notna()
            matches_safe = bool(mask.any() and _close(got[mask], safe[mask]))
            if mask.any() and not matches_safe:
                errors.append(
                    "expanding_mean does not match the expanding mean of "
                    f"{target_col} shifted by {expanding.base_lag}"
                )
            # A constant series makes the shifted mean and the raw mean identical,
            # so a match with the raw mean is a leak only when the safe mean differs.
            leaky = df[target_col].expanding(min_periods=expanding.min_periods).mean()
            leak_mask = got.notna() & leaky.notna()
            matches_leaky = bool(leak_mask.any() and _close(got[leak_mask], leaky[leak_mask]))
            if matches_leaky and not matches_safe:
                errors.append("expanding_mean was computed on the raw target")

        self._leakage_errors = errors
        if errors:
            for message in errors:
                logger.error(f"Leakage detected: {message}")
            return False

        logger.info("Leakage validation passed: every target, oil, and transaction lag is >= horizon.")
        return True
    
    def drop_na_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows with NaN values created by lagging/rolling."""
        initial_len = len(df)
        
        feature_cols = [f for feats in self.get_all_feature_names().values() for f in feats]
        check_cols = [c for c in feature_cols if c in df.columns]
        
        df_out = df.dropna(subset=check_cols)
        dropped = initial_len - len(df_out)
        logger.info(f"Dropped {dropped} rows due to NaN values in generated features.")
        return df_out


def _close(left: pd.Series, right: pd.Series) -> bool:
    return bool(np.allclose(left.to_numpy(dtype=float), right.to_numpy(dtype=float), equal_nan=True))
