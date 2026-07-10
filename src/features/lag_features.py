from __future__ import annotations

import pandas as pd
from loguru import logger

class LagFeatureGenerator:
    """Generate lag features from the target variable.
    
    Lags represent historical values at fixed offsets.
    CRITICAL: minimum lag must be >= forecast_horizon to prevent leakage.
    """
    
    def __init__(self, lags: list[int] | None = None,
                 forecast_horizon: int = 28) -> None:
        raw_lags = lags or [1, 7, 14, 28]
        self.forecast_horizon = forecast_horizon
        
        valid_lags = []
        for lag in raw_lags:
            if lag < self.forecast_horizon:
                logger.warning(f"Filtering out lag {lag} to prevent data leakage (horizon={self.forecast_horizon}).")
            else:
                valid_lags.append(lag)
                
        if not valid_lags:
            valid_lags = [self.forecast_horizon]
            logger.warning(f"All lags filtered out! Falling back to lag={self.forecast_horizon}")
            
        self.lags = valid_lags
        self._feature_names: list[str] = []

    def transform(self, df: pd.DataFrame, target_col: str = 'sales',
                  group_cols: list[str] | None = None) -> pd.DataFrame:
        df_out = df.copy()
        
        missing = [c for c in ([target_col] + (group_cols or [])) if c not in df.columns]
        if missing:
            raise KeyError(f"Columns not found in DataFrame: {missing}")
            
        self._feature_names = []
        for lag in self.lags:
            feat_name = f"lag_{lag}"
            self._feature_names.append(feat_name)
            
            if group_cols:
                df_out[feat_name] = df_out.groupby(group_cols)[target_col].shift(lag)
            else:
                df_out[feat_name] = df_out[target_col].shift(lag)
                
            nan_count = df_out[feat_name].isna().sum()
            logger.info(f"Created feature {feat_name}: {nan_count} NaN values produced.")
            
        return df_out

    def get_feature_names(self) -> list[str]:
        return self._feature_names
