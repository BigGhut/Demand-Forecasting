from __future__ import annotations

import pandas as pd
from loguru import logger

from darts import TimeSeries
from darts.models import CatBoostModel as DartsCatBoostModel

from src.utils.config import CatBoostConfig, get_base_config

class CatBoostResidualModel:
    """CatBoost model wrapper using Darts integration for learning on Prophet residuals.
    
    Uses likelihood='gaussian' to output mean and variance (RMSEWithUncertainty equivalent).
    """
    
    def __init__(self, config: CatBoostConfig | None = None,
                 lags: int | list[int] | None = None,
                 lags_past_covariates: int | list[int] | None = None,
                 categorical_past_covariates: list[str] | None = None) -> None:
        self.config = config or get_base_config().catboost
        
        # Build kwargs for CatBoost based on our config
        self.catboost_kwargs = {
            "iterations": self.config.iterations,
            "learning_rate": self.config.learning_rate,
            "depth": self.config.depth,
            "task_type": self.config.task_type,
            "verbose": self.config.verbose,
            "early_stopping_rounds": self.config.early_stopping_rounds,
        }
        
        # Default to specific lags [-28, -35, -42, -56] to prevent recursive forecasting accumulation error
        # Darts requires list lags to be negative integers.
        self.lags = lags if lags is not None else [-28, -35, -42, -56]
        self.lags_past_covariates = lags_past_covariates if lags_past_covariates is not None else 28
        
        # Darts specific params
        self.model = DartsCatBoostModel(
            lags=self.lags,
            lags_past_covariates=self.lags_past_covariates,
            categorical_past_covariates=categorical_past_covariates,
            likelihood='gaussian',  # Enables probabilistic forecasting
            random_state=42,
            **self.catboost_kwargs
        )
        self._is_fitted = False
        
    def fit(self, series: TimeSeries, past_covariates: TimeSeries | None = None) -> CatBoostResidualModel:
        """Fit CatBoost model on the target series (residuals) with covariates."""
        logger.info(f"Fitting CatBoost model with config: {self.config}")
        self.model.fit(series, past_covariates=past_covariates)
        self._is_fitted = True
        return self
        
    def predict(self, n: int, past_covariates: TimeSeries | None = None,
                num_samples: int = 100) -> TimeSeries:
        """Forecast the next n periods. 
        Returns stochastic forecasts from the Gaussian distribution.
        """
        if not self._is_fitted:
            raise ValueError("Model is not fitted yet.")
        logger.info(f"Predicting next {n} steps with CatBoost (num_samples={num_samples})")
        return self.model.predict(n=n, past_covariates=past_covariates, num_samples=num_samples)
