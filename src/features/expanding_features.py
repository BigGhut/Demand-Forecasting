from __future__ import annotations

import pandas as pd
from loguru import logger

class ExpandingFeatureGenerator:
    """Generate expanding window statistics for capturing global drift.

    Statistics are computed on a lag that is at least ``forecast_horizon``
    steps back. At the last forecast step that lag still points at the
    forecast origin, so the validation target never enters the feature.
    """

    def __init__(self, functions: list[str] | None = None,
                 min_periods: int = 28,
                 forecast_horizon: int = 28) -> None:
        self.functions = functions or ["mean", "std"]
        self.min_periods = min_periods
        self.forecast_horizon = forecast_horizon
        self.base_lag = forecast_horizon
        self._feature_names: list[str] = []

    def transform(self, df: pd.DataFrame, target_col: str = 'sales',
                  group_cols: list[str] | None = None) -> pd.DataFrame:
        df_out = df.copy()
        self._feature_names = []

        if target_col not in df_out.columns:
            raise KeyError(f"Column not found in DataFrame: {target_col}")

        candidates: list[tuple[int, str]] = []
        for col in df_out.columns:
            if not col.startswith("lag_"):
                continue
            suffix = col[4:]
            if suffix.isdigit() and int(suffix) >= self.forecast_horizon:
                candidates.append((int(suffix), col))

        temporary = False
        if candidates:
            self.base_lag, base_col = min(candidates)
        else:
            # Do not fall back to the raw target. A shift of `horizon` is the
            # shortest history that is known for every step of the forecast.
            self.base_lag = self.forecast_horizon
            base_col = f"_leakfree_lag_{self.base_lag}"
            temporary = True
            logger.info(
                f"No lag column >= {self.forecast_horizon}. "
                f"Expanding stats use {target_col} shifted by {self.base_lag}."
            )
            if group_cols:
                df_out[base_col] = df_out.groupby(group_cols)[target_col].shift(self.base_lag)
            else:
                df_out[base_col] = df_out[target_col].shift(self.base_lag)
            
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
                
            logger.debug(f"Computed {feat_name} on lag_{self.base_lag}")

        if temporary:
            df_out = df_out.drop(columns=[base_col])

        logger.info(
            f"Generated {len(self._feature_names)} expanding features on lag_{self.base_lag}."
        )
        return df_out

    def get_feature_names(self) -> list[str]:
        return self._feature_names
