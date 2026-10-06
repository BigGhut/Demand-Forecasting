from __future__ import annotations

import pandas as pd
from loguru import logger

class RollingFeatureGenerator:
    """Generate rolling statistics computed ON TOP OF lag features.
    
    CRITICAL: Two-step pattern to prevent data leakage:
    1. The lag column already shifts data by N periods
    2. Rolling stats are computed on the lag column, NOT the raw target
    This means rolling_mean_lag28_w7 = rolling_mean(lag_28, window=7)
    for a 28-day horizon. Base lags inside the horizon are replaced, not skipped.
    """
    
    def __init__(self, windows: list[int] | None = None,
                 functions: list[str] | None = None,
                 base_lags: list[int] | None = None,
                 forecast_horizon: int = 28) -> None:
        self.windows = windows or [7, 14, 28]
        self.functions = functions or ["mean", "std"]
        self.forecast_horizon = forecast_horizon
        requested = list(base_lags) if base_lags is not None else [forecast_horizon]
        valid = [lag for lag in requested if lag >= forecast_horizon]
        if not valid:
            logger.warning(
                f"Rolling base lags {requested} sit inside the horizon {forecast_horizon}. "
                f"Using lag {forecast_horizon} instead of dropping the features."
            )
            valid = [forecast_horizon]
        elif len(valid) < len(requested):
            skipped = [lag for lag in requested if lag < forecast_horizon]
            logger.warning(
                f"Skipping rolling base lags {skipped}: they sit inside the horizon {forecast_horizon}."
            )
        self.base_lags = valid
        self._feature_names: list[str] = []

    def transform(self, df: pd.DataFrame,
                  group_cols: list[str] | None = None,
                  target_col: str = "sales") -> pd.DataFrame:
        df_out = df.copy()
        self._feature_names = []
        temporary_cols: list[str] = []

        for lag in self.base_lags:
            lag_col = f"lag_{lag}"
            if lag_col not in df_out.columns:
                if target_col not in df_out.columns:
                    raise KeyError(
                        f"Cannot build rolling features: missing {lag_col} and {target_col}."
                    )
                logger.info(
                    f"{lag_col} is absent. Rolling stats use {target_col} shifted by {lag}."
                )
                if group_cols:
                    df_out[lag_col] = df_out.groupby(group_cols)[target_col].shift(lag)
                else:
                    df_out[lag_col] = df_out[target_col].shift(lag)
                temporary_cols.append(lag_col)

        for lag in self.base_lags:
            lag_col = f"lag_{lag}"
            
            # Using groupby rolling if group_cols provided
            if group_cols:
                grouped = df_out.groupby(group_cols)[lag_col]
            else:
                grouped = df_out[lag_col]
                
            for window in self.windows:
                rolling_obj = grouped.rolling(window=window, min_periods=1)
                
                for func in self.functions:
                    feat_name = f"rolling_{func}_lag{lag}_w{window}"
                    self._feature_names.append(feat_name)
                    
                    if func == 'mean':
                        res = rolling_obj.mean()
                    elif func == 'std':
                        res = rolling_obj.std()
                    elif func == 'min':
                        res = rolling_obj.min()
                    elif func == 'max':
                        res = rolling_obj.max()
                    elif func == 'median':
                        res = rolling_obj.median()
                    else:
                        raise ValueError(f"Unsupported rolling function: {func}")
                        
                    if group_cols:
                        df_out[feat_name] = res.reset_index(level=group_cols, drop=True)
                    else:
                        df_out[feat_name] = res
                        
                    logger.debug(f"Computed {feat_name}")

        if temporary_cols:
            df_out = df_out.drop(columns=temporary_cols)

        if not self._feature_names:
            raise RuntimeError(
                f"RollingFeatureGenerator produced no features for base lags {self.base_lags}."
            )

        logger.info(f"Generated {len(self._feature_names)} rolling features.")
        return df_out

    def get_feature_names(self) -> list[str]:
        return self._feature_names
