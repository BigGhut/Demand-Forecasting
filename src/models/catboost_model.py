from __future__ import annotations

import pandas as pd
from loguru import logger

from darts import TimeSeries
from darts.models import CatBoostModel as DartsCatBoostModel

from src.utils.config import CatBoostConfig, get_base_config


def horizon_safe_lags(forecast_horizon: int) -> list[int]:
    """Lags at or beyond the horizon.

    With these lags, an ``n <= forecast_horizon`` forecast never feeds a
    predicted residual back in as an input.
    """
    return [
        -forecast_horizon,
        -(forecast_horizon + 7),
        -(forecast_horizon + 14),
        -(forecast_horizon + 28),
    ]


def resolve_task_type(requested: str) -> str:
    """Use GPU only when CatBoost can see one. Otherwise stay on CPU."""
    if requested != "GPU":
        return "CPU"
    try:
        from catboost.utils import get_gpu_device_count

        if get_gpu_device_count() > 0:
            return "GPU"
    except Exception:
        logger.warning("CatBoost GPU check failed; using CPU.")
        return "CPU"
    logger.warning("CatBoost task_type=GPU but no GPU is available; using CPU.")
    return "CPU"


class CatBoostResidualModel:
    """CatBoost model wrapper using Darts integration for learning on Prophet residuals.
    
    ``loss_function='RMSEWithUncertainty'`` is CatBoost's Gaussian likelihood:
    Darts applies it when ``likelihood='gaussian'``. Any other configured loss
    is passed through as ``loss_function``.
    """

    def __init__(self, config: CatBoostConfig | None = None,
                 lags: int | list[int] | None = None,
                 lags_past_covariates: int | list[int] | None = None,
                 lags_future_covariates: list[int] | tuple[int, int] | None = None,
                 categorical_past_covariates: list[str] | None = None) -> None:
        self.config = config or get_base_config().catboost
        resolved_task = resolve_task_type(self.config.task_type)
        if resolved_task != self.config.task_type:
            self.config = self.config.model_copy(update={"task_type": resolved_task})

        # Build kwargs for CatBoost based on our config
        self.catboost_kwargs = {
            "iterations": self.config.iterations,
            "learning_rate": self.config.learning_rate,
            "depth": self.config.depth,
            "task_type": self.config.task_type,
            "verbose": self.config.verbose,
            "early_stopping_rounds": self.config.early_stopping_rounds,
        }

        # Darts maps this likelihood onto loss_function=RMSEWithUncertainty and
        # rejects a second objective, so the configured loss is applied here.
        likelihood: str | None
        if self.config.loss_function == "RMSEWithUncertainty":
            likelihood = "gaussian"
        else:
            likelihood = None
            self.catboost_kwargs["loss_function"] = self.config.loss_function

        # Lags at or beyond a 28-day horizon. An integer such as 7 would make
        # a 28-step forecast recursive. Darts wants list lags to be negative.
        safe_lags = horizon_safe_lags(28)
        self.lags = lags if lags is not None else safe_lags
        self.lags_past_covariates = (
            lags_past_covariates if lags_past_covariates is not None else safe_lags
        )
        self.lags_future_covariates = lags_future_covariates

        # Darts specific params
        self.model = DartsCatBoostModel(
            lags=self.lags,
            lags_past_covariates=self.lags_past_covariates,
            lags_future_covariates=self.lags_future_covariates,
            categorical_past_covariates=categorical_past_covariates,
            likelihood=likelihood,
            random_state=42,
            **self.catboost_kwargs
        )
        self._is_fitted = False

    def fit(self, series: TimeSeries,
            past_covariates: TimeSeries | None = None,
            future_covariates: TimeSeries | None = None) -> CatBoostResidualModel:
        """Fit CatBoost model on the target series (residuals) with covariates."""
        logger.info(f"Fitting CatBoost model with config: {self.config}")
        self.model.fit(
            series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )
        self._is_fitted = True
        return self

    def predict(self, n: int,
                past_covariates: TimeSeries | None = None,
                future_covariates: TimeSeries | None = None,
                num_samples: int = 100) -> TimeSeries:
        """Forecast the next n periods.
        Returns stochastic forecasts when the loss is RMSEWithUncertainty.
        """
        if not self._is_fitted:
            raise ValueError("Model is not fitted yet.")
        logger.info(f"Predicting next {n} steps with CatBoost (num_samples={num_samples})")
        return self.model.predict(
            n=n,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            num_samples=num_samples,
        )
