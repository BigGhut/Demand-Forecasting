from __future__ import annotations

import pandas as pd
import numpy as np
from loguru import logger

class CalendarFeatureGenerator:
    """Extract calendar/date-part features from the date column.
    
    These are deterministic features known ahead of time (future covariates).
    """
    
    def __init__(self, features: list[str] | None = None,
                 binary_features: list[str] | None = None) -> None:
        self.features = features or ["day_of_week", "month", "quarter", "day_of_month", "week_of_year"]
        self.binary_features = binary_features or ["is_weekend", "is_month_start", "is_month_end", "is_payday"]
        self._feature_names: list[str] = []

    def transform(self, df: pd.DataFrame, 
                  date_col: str = 'date') -> pd.DataFrame:
        if date_col not in df.columns:
            raise KeyError(f"Date column '{date_col}' not found.")
            
        df_out = df.copy()
        date_series = pd.to_datetime(df_out[date_col])
        self._feature_names = []
        
        # Calendar parts
        if "day_of_week" in self.features:
            df_out["day_of_week"] = date_series.dt.dayofweek.astype(np.int8)
            self._feature_names.append("day_of_week")
        if "month" in self.features:
            df_out["month"] = date_series.dt.month.astype(np.int8)
            self._feature_names.append("month")
        if "quarter" in self.features:
            df_out["quarter"] = date_series.dt.quarter.astype(np.int8)
            self._feature_names.append("quarter")
        if "day_of_month" in self.features:
            df_out["day_of_month"] = date_series.dt.day.astype(np.int8)
            self._feature_names.append("day_of_month")
        if "week_of_year" in self.features:
            df_out["week_of_year"] = date_series.dt.isocalendar().week.astype(np.int8)
            self._feature_names.append("week_of_year")
            
        # Binary features
        if "is_weekend" in self.binary_features:
            df_out["is_weekend"] = (date_series.dt.dayofweek >= 5).astype(np.int8)
            self._feature_names.append("is_weekend")
        if "is_month_start" in self.binary_features:
            df_out["is_month_start"] = date_series.dt.is_month_start.astype(np.int8)
            self._feature_names.append("is_month_start")
        if "is_month_end" in self.binary_features:
            df_out["is_month_end"] = date_series.dt.is_month_end.astype(np.int8)
            self._feature_names.append("is_month_end")
        if "is_payday" in self.binary_features:
            # Assume 15th and last day of month are paydays
            is_15th = date_series.dt.day == 15
            is_end = date_series.dt.is_month_end
            df_out["is_payday"] = (is_15th | is_end).astype(np.int8)
            self._feature_names.append("is_payday")
            
        # Continuous Payday distance features
        min_date = date_series.min() - pd.Timedelta(days=31)
        max_date = date_series.max() + pd.Timedelta(days=31)
        all_dates = pd.date_range(start=min_date, end=max_date, freq='D')
        is_payday_all = (all_dates.day == 15) | (all_dates.is_month_end)
        payday_dates = all_dates[is_payday_all]
        
        idx_next = np.searchsorted(payday_dates, date_series, side='left')
        next_paydays = payday_dates[idx_next]
        df_out["days_until_next_payday"] = (next_paydays - date_series).dt.days.astype(np.int8)
        self._feature_names.append("days_until_next_payday")
        
        idx_prev = np.searchsorted(payday_dates, date_series, side='right') - 1
        prev_paydays = payday_dates[idx_prev]
        df_out["days_since_last_payday"] = (date_series - prev_paydays).dt.days.astype(np.int8)
        self._feature_names.append("days_since_last_payday")
            
        logger.info(f"Generated {len(self._feature_names)} calendar features (including payday continuous distances).")
        return df_out

    def get_feature_names(self) -> list[str]:
        return self._feature_names
