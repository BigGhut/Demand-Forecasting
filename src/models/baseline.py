from __future__ import annotations

import pandas as pd
from loguru import logger

from darts import TimeSeries
from darts.models import NaiveSeasonal, NaiveMovingAverage, ExponentialSmoothing

class BaselineModels:
    """A collection of naive and baseline models for comparison."""
    
    @staticmethod
    def get_seasonal_naive(K: int = 7) -> NaiveSeasonal:
        """Seasonal naive model. Repeats the value from K periods ago (e.g. 7 for weekly)."""
        logger.info(f"Initializing NaiveSeasonal model with K={K}")
        return NaiveSeasonal(K=K)
        
    @staticmethod
    def get_moving_average(input_chunk_length: int = 14) -> NaiveMovingAverage:
        """Moving average baseline."""
        logger.info(f"Initializing NaiveMovingAverage model with input_chunk_length={input_chunk_length}")
        return NaiveMovingAverage(input_chunk_length=input_chunk_length)
        
    @staticmethod
    def get_exponential_smoothing(seasonal_periods: int = 7) -> ExponentialSmoothing:
        """Exponential smoothing baseline."""
        logger.info(f"Initializing ExponentialSmoothing model with seasonal_periods={seasonal_periods}")
        return ExponentialSmoothing(seasonal_periods=seasonal_periods)
