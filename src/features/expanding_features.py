from __future__ import annotations

import pandas as pd
from loguru import logger

class ExpandingFeatureGenerator:
    """Generate expanding window statistics for capturing global drift.
    
    Expanding mean captures the long-term average of the series,
    less sensitive to individual anomalies than rolling stats.
    Also computed ON TOP OF lag features to prevent leakage.
    """
    
    def __init__(self, functions: list[str] | None = None,
                 min_periods: int = 28) -> None:
        self.functions = functions or ["mean", "std"]
        self.min_periods = min_periods
        self._feature_names: list[str] = []

    def transform(self, df: pd.DataFrame, target_col: str = 'sales',
                  group_cols: list[str] | None = None) -> pd.DataFrame:
        df_out = df.copy()
        self._feature_names = []
        
        # To avoid data leakage, we compute expanding stats on lag_1 (or the smallest lag)
        # If lag_1 is not available, we should probably warn or compute it, but we assume it's created
        lag_col = "lag_1" 
        if lag_col not in df.columns:
            logger.warning(f"Expanding features should ideally be computed on lagged target. {lag_col} not found. Computing on raw target (LEAKAGE RISK if used for forecasting!).")
            base_col = target_col
        else:
            base_col = lag_col
            
        if group_cols:
            expanding_obj = df_out.groupby(group_cols)[base_col].expanding(min_periods=self.min_periods)
        else:
            expanding_obj = df_out[base_col].expanding(min_periods=self.min_periods)
            
        for func in self.functions:
            feat_name = f"expanding_{func}"
            self._feature_names.append(feat_name)
            
            if func == 'mean':
                res = expanding_obj.mean()
            elif func == 'std':
                res = expanding_obj.std()
            else:
                raise ValueError(f"Unsupported expanding function: {func}")
                
            if group_cols:
                df_out[feat_name] = res.reset_index(level=group_cols, drop=True)
            else:
                df_out[feat_name] = res
                
            logger.debug(f"Computed {feat_name}")
            
        logger.info(f"Generated {len(self._feature_names)} expanding features.")
        return df_out

    def get_feature_names(self) -> list[str]:
        return self._feature_names
