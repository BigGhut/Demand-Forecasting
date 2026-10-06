from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from scipy.special import inv_boxcox

from darts import TimeSeries

from src.models.prophet_model import ProphetForecaster
from src.models.catboost_model import CatBoostResidualModel
from src.transforms.target_transforms import TargetTransformer
from src.utils.config import get_base_config


def _as_series(series: TimeSeries) -> pd.Series:
    raw = series.to_series()
    if isinstance(raw, pd.DataFrame):
        raw = raw.iloc[:, 0]
    return raw


def transform_series(
    transformer: TargetTransformer,
    series: TimeSeries,
    future_covariates: TimeSeries | None = None,
    past_covariates: TimeSeries | None = None,
) -> tuple[TimeSeries, TimeSeries | None, TimeSeries | None]:
    """Fit the target transform on ``series`` and align covariate start dates.

    ``series`` is this window's training target. Box-Cox λ is estimated from
    those points only; a longer covariate index is not part of the fit.
    The covariate series keep their end, which is the forecast window.
    Only a leading timestamp dropped by differencing is trimmed.
    """
    raw = _as_series(series)
    transformer.fit(raw)
    transformed = transformer.transform(raw)
    series_t = TimeSeries.from_series(transformed, freq=series.freq or "D")
    return (
        series_t,
        _trim_start(future_covariates, series_t),
        _trim_start(past_covariates, series_t),
    )


def _trim_start(covariates: TimeSeries | None, series: TimeSeries) -> TimeSeries | None:
    if covariates is None:
        return None
    if covariates.start_time() < series.start_time():
        return covariates.slice(series.start_time(), covariates.end_time())
    return covariates


def inverse_target(
    transformer: TargetTransformer,
    predicted: TimeSeries,
    anchor_level: float | None = None,
) -> TimeSeries:
    """Map a forecast from transformed space back to sales."""
    method = transformer.method
    if method == "none":
        return predicted
    if method == "log":
        return predicted.map(np.expm1)
    if method == "box_cox":
        lam = transformer._boxcox_lambda
        shift = transformer._boxcox_shift
        return predicted.map(lambda values: inv_boxcox(values, lam) - shift)
    if method == "difference":
        if anchor_level is None:
            raise ValueError("Difference inverse needs the last observed sales level.")
        levels = np.cumsum(predicted.all_values(copy=True), axis=0) + anchor_level
        return TimeSeries.from_times_and_values(
            predicted.time_index,
            levels,
            freq=predicted.freq,
            columns=predicted.components,
        )
    raise ValueError(f"Cannot invert target transform '{method}'.")


class HybridProphetCatBoost:
    """The full cascade: Prophet -> Residuals -> CatBoost + Inverse Transform."""
    
    def __init__(self, 
                 prophet_model: ProphetForecaster,
                 catboost_model: CatBoostResidualModel,
                 target_transformer: TargetTransformer | None = None) -> None:
        self.prophet = prophet_model
        self.catboost = catboost_model
        if target_transformer is None:
            method = get_base_config().pipeline.target_transform
            target_transformer = TargetTransformer(method=method)
        self.target_transformer = target_transformer
        self._anchor_level: float | None = None
        self._is_fitted = False

    def fit(self, 
            series: TimeSeries, 
            future_covariates: TimeSeries | None = None,
            past_covariates: TimeSeries | None = None) -> HybridProphetCatBoost:
        """Fit the hybrid cascade.

        1. Apply the configured target transform (and keep state for the inverse)
        2. Fit Prophet on the transformed target
        3. Fit CatBoost on the Prophet residuals in that same space
        """
        self._anchor_level = float(_as_series(series).iloc[-1])
        # New parameters for this training window. A λ fit on another window,
        # or on train concatenated with validation, is discarded here.
        self.target_transformer = TargetTransformer(method=self.target_transformer.method)
        series_t, future_t, past_t = transform_series(
            self.target_transformer, series, future_covariates, past_covariates
        )
        logger.info(
            f"Starting Phase 1: Fitting Prophet on '{self.target_transformer.method}' target..."
        )
        self.prophet.fit(series_t, future_covariates=future_t)

        logger.info("Extracting residuals from Prophet...")
        residuals = self.prophet.get_residuals(series_t, future_covariates=future_t)

        logger.info("Starting Phase 2: Fitting CatBoost on residuals...")
        self.catboost.fit(residuals, past_covariates=past_t)

        self._is_fitted = True
        logger.success("Hybrid model fit complete.")
        return self

    def inverse_transform(self, series: TimeSeries) -> TimeSeries:
        """Invert a series produced in the transformed target space."""
        return inverse_target(self.target_transformer, series, self._anchor_level)

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
        return self.inverse_transform(final_pred)
