from __future__ import annotations

import pandas as pd
from loguru import logger

class RollingFeatureGenerator:
    """Generate rolling statistics computed ON TOP OF lag features.
    
    CRITICAL: Two-step pattern to prevent data leakage:
    1. The lag column already shifts data by N periods
    2. Rolling stats are computed on the lag column, NOT the raw target
    This means rolling_mean_lag1_w7 = rolling_mean(lag_1, window=7)
    """
    
    def __init__(self, windows: list[int] | None = None,
                 functions: list[str] | None = None,
                 base_lags: list[int] | None = None) -> None:
        self.windows = windows or [7, 14, 28]
        self.functions = functions or ["mean", "std"]
        self.base_lags = base_lags or [1, 7]
        self._feature_names: list[str] = []

    def transform(self, df: pd.DataFrame,
                  group_cols: list[str] | None = None) -> pd.DataFrame:
        df_out = df.copy()
        self._feature_names = []
        
        # Validate base lag columns exist
        valid_base_lags = []
        for lag in self.base_lags:
            lag_col = f"lag_{lag}"
            if lag_col not in df.columns:
                logger.warning(f"Base lag column {lag_col} not found in DataFrame. Skipping rolling features for this lag.")
            else:
                valid_base_lags.append(lag)
                
        for lag in valid_base_lags:
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
                    
        logger.info(f"Generated {len(self._feature_names)} rolling features.")
        return df_out

    def get_feature_names(self) -> list[str]:
        return self._feature_names
