from __future__ import annotations

import pandas as pd
from loguru import logger
from darts import TimeSeries

from src.utils.config import get_base_config

class TimeSeriesCV:
    """Chronological time-series cross-validation using expanding windows.
    
    Compatible with Darts TimeSeries objects.
    """
    
    def __init__(self, forecast_horizon: int | None = None,
                 stride: int | None = None,
                 n_windows: int | None = None) -> None:
        config = get_base_config().pipeline
        self.forecast_horizon = forecast_horizon or config.forecast_horizon
        self.stride = stride or config.cv_stride
        self.n_windows = n_windows or config.cv_windows
        
    def generate_splits(self, series: TimeSeries) -> list[tuple[TimeSeries, TimeSeries]]:
        """Generate (train, val) splits using an expanding window.
        
        val is always `forecast_horizon` points long.
        train expands by `stride` at each step.
        """
        n = len(series)
        total_val_points = self.forecast_horizon + (self.n_windows - 1) * self.stride
        
        if n <= total_val_points:
            raise ValueError(
                f"Series length ({n}) is too short for {self.n_windows} windows "
                f"with horizon {self.forecast_horizon} and stride {self.stride}."
            )
            
        initial_train_size = n - total_val_points
        logger.info(
            f"Generating {self.n_windows} CV splits. "
            f"Initial train size: {initial_train_size}, Horizon: {self.forecast_horizon}, Stride: {self.stride}"
        )
        
        splits = []
        for i in range(self.n_windows):
            train_end = initial_train_size + i * self.stride
            val_end = train_end + self.forecast_horizon
            
            train_split = series[:train_end]
            val_split = series[train_end:val_end]
            splits.append((train_split, val_split))
            
        return splits
