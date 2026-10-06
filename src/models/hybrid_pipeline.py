from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from loguru import logger
from scipy.special import inv_boxcox

from darts import TimeSeries

from src.models.prophet_model import ProphetForecaster
from src.models.catboost_model import CatBoostResidualModel
from src.models.residual_origins import (
    OriginForecaster,
    min_residual_length,
    rolling_origin_residuals,
)
from src.transforms.target_transforms import TargetTransformer, clip_to_boxcox_domain
from src.utils.config import get_base_config


def _as_series(series: TimeSeries) -> pd.Series:
    raw = series.to_series()
    if isinstance(raw, pd.DataFrame):
        raw = raw.iloc[:, 0]
    return raw


def stack_covariates(
    left: TimeSeries | None,
    right: TimeSeries | None,
) -> TimeSeries | None:
    """Combine two covariate series that share a time index."""
    if left is None:
        return right
    if right is None:
        return left
    return left.stack(right)


def transform_series(
    transformer: TargetTransformer,
    series: TimeSeries,
    future_covariates: TimeSeries | None = None,
    shifted_covariates: TimeSeries | None = None,
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
        _trim_start(shifted_covariates, series_t),
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
        return predicted.map(
            lambda values: inv_boxcox(clip_to_boxcox_domain(values, lam), lam) - shift
        )
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
            shifted_covariates: TimeSeries | None = None,
            *,
            residual_horizon: int | None = None,
            residual_min_train_size: int | None = None,
            residual_stride: int | None = None,
            forecaster_factory: Callable[[], OriginForecaster] | None = None,
            ) -> HybridProphetCatBoost:
        """Fit the hybrid cascade.

        1. Apply the configured target transform (and keep state for the inverse)
        2. Build CatBoost targets from rolling-origin Prophet forecast errors
        3. Fit the deployment Prophet on the full transformed training series
        4. Fit CatBoost on those out-of-sample residuals

        ``get_residuals`` is not the CatBoost target. That method scores the
        same days the Prophet just fitted, and those errors will not be the
        ones left at forecast time.
        """
        self._anchor_level = float(_as_series(series).iloc[-1])
        # New parameters for this training window. A λ fit on another window,
        # or on train concatenated with validation, is discarded here.
        self.target_transformer = TargetTransformer(method=self.target_transformer.method)
        series_t, future_t, shifted_t = transform_series(
            self.target_transformer, series, future_covariates, shifted_covariates
        )
        pipeline = get_base_config().pipeline
        horizon = residual_horizon or pipeline.forecast_horizon
        min_train = residual_min_train_size or pipeline.residual_min_train_size
        logger.info(
            "Starting Phase 1: out-of-sample Prophet residuals "
            f"on '{self.target_transformer.method}' target..."
        )
        factory = forecaster_factory or (
            lambda: ProphetForecaster(config=self.prophet.config)
        )
        residuals = rolling_origin_residuals(
            series_t,
            future_t,
            horizon=horizon,
            min_train_size=min_train,
            stride=residual_stride,
            forecaster_factory=factory,
        )
        needed = min_residual_length(self.catboost.lags)
        if len(residuals) < needed:
            raise ValueError(
                f"Out-of-sample residuals have {len(residuals)} points; "
                f"CatBoost lags need at least {needed}. "
                "In-sample Prophet residuals are not used as a fallback."
            )

        # The model that will forecast the future sees the whole training
        # window. It is not one of the prefix models that labeled the residuals.
        self.prophet.fit(series_t, future_covariates=future_t)

        logger.info(
            "Starting Phase 2: Fitting CatBoost on {} out-of-sample residual points.",
            len(residuals),
        )
        # Pre-shifted columns are known on the forecast dates. Lag 0 reads
        # the column at the step being predicted, so an unshifted series
        # cannot be mixed in by accident. The covariate end stays past the
        # residual series: that is the horizon being forecast.
        self.catboost.fit(residuals, future_covariates=_trim_start(shifted_t, residuals))

        self._is_fitted = True
        logger.success("Hybrid model fit complete.")
        return self

    def inverse_transform(self, series: TimeSeries) -> TimeSeries:
        """Invert a series produced in the transformed target space."""
        return inverse_target(self.target_transformer, series, self._anchor_level)

    def predict(self, n: int, 
                future_covariates: TimeSeries | None = None,
                shifted_covariates: TimeSeries | None = None,
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
        catboost_pred = self.catboost.predict(
            n, future_covariates=shifted_covariates, num_samples=num_samples
        )
        
        logger.info("Synthesizing final forecast (Prophet + CatBoost residuals).")
        # Darts uses the left operand's shape for the output array.
        # Since catboost is stochastic (num_samples > 1), it must be on the left.
        final_pred = catboost_pred + prophet_pred
        return self.inverse_transform(final_pred)
