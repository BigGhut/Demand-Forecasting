from __future__ import annotations

import pandas as pd
from loguru import logger

from darts import TimeSeries

from src.models.prophet_model import ProphetForecaster
from src.models.catboost_model import CatBoostResidualModel
from src.transforms.target_transforms import TargetTransformer

class HybridProphetCatBoost:
    """The full cascade: Prophet -> Residuals -> CatBoost + Inverse Transform."""
    
    def __init__(self, 
                 prophet_model: ProphetForecaster,
                 catboost_model: CatBoostResidualModel,
                 target_transformer: TargetTransformer | None = None) -> None:
        self.prophet = prophet_model
        self.catboost = catboost_model
        self.target_transformer = target_transformer or TargetTransformer(method="none")
        self._is_fitted = False

    def fit(self, 
            series: TimeSeries, 
            future_covariates: TimeSeries | None = None,
            past_covariates: TimeSeries | None = None) -> HybridProphetCatBoost:
        """Fit the hybrid cascade.
        
        1. Fit Prophet on target
        2. Get Prophet residuals (in-sample error)
        3. Fit CatBoost on Prophet residuals
        """
        # Transform target if needed
        # Assuming `series` passed here is already transformed by `TargetTransformer.fit_transform` upstream,
        # but if we want this class to manage it:
        # Actually it's better if `TargetTransformer` transforms standard pd.Series. 
        # Let's assume `series` is already transformed before passing to fit.
        
        logger.info("Starting Phase 1: Fitting Prophet...")
        self.prophet.fit(series, future_covariates=future_covariates)
        
        logger.info("Extracting residuals from Prophet...")
        residuals = self.prophet.get_residuals(series, future_covariates=future_covariates)
        
        logger.info("Starting Phase 2: Fitting CatBoost on residuals...")
        self.catboost.fit(residuals, past_covariates=past_covariates)
        
        self._is_fitted = True
        logger.success("Hybrid model fit complete.")
        return self

    def predict(self, n: int, 
                future_covariates: TimeSeries | None = None,
                past_covariates: TimeSeries | None = None,
                num_samples: int = 100) -> TimeSeries:
        """Forecast n periods.
        
        1. Predict with Prophet
        2. Predict residuals with CatBoost
        3. Add together
        """
        if not self._is_fitted:
            raise ValueError("Hybrid model is not fitted yet.")
            
        logger.info(f"Predicting next {n} steps with Hybrid Model.")
        
        prophet_pred = self.prophet.predict(n, future_covariates=future_covariates)
        
        # Note: Prophet pred is deterministic (1 sample), CatBoost is probabilistic
        catboost_pred = self.catboost.predict(n, past_covariates=past_covariates, num_samples=num_samples)
        
        logger.info("Synthesizing final forecast (Prophet + CatBoost residuals).")
        # Darts uses the left operand's shape for the output array. 
        # Since catboost is stochastic (num_samples > 1), it must be on the left.
        final_pred = catboost_pred + prophet_pred
        
        return final_pred
