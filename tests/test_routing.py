import inspect

import numpy as np
import pytest
import pandas as pd
from darts import TimeSeries

from src.models.catboost_model import CatBoostResidualModel, resolve_task_type
from src.models.hybrid_pipeline import HybridProphetCatBoost, inverse_target, transform_series
from src.models.routing import ABCSegmenter, DirectCatBoostForecaster, ForecastRouter
from src.transforms.target_transforms import TargetTransformer


def _segmenter() -> ABCSegmenter:
    df = pd.DataFrame({
        "store_nbr": [1, 2],
        "family": ["PRODUCE", "PRODUCE"],
        "sales": [30.0, 70.0],
    })
    return ABCSegmenter(df)


def test_class_c_matches_hybrid_interface():
    segmenter = _segmenter()
    model = ForecastRouter(1, "PRODUCE", segmenter).get_model(forecast_horizon=28)
    assert isinstance(model, DirectCatBoostForecaster)
    assert not isinstance(model, CatBoostResidualModel)

    fit_params = inspect.signature(model.fit).parameters
    pred_params = inspect.signature(model.predict).parameters
    assert {"future_covariates", "shifted_covariates"} <= set(fit_params)
    assert {"future_covariates", "shifted_covariates", "num_samples"} <= set(pred_params)
    assert model.catboost.lags == [-28, -35, -42, -56]
    assert model.catboost.lags_future_covariates == [0]
    assert model.catboost.lags_past_covariates is None
    assert model.catboost.catboost_kwargs["task_type"] == "CPU"
    assert model.catboost.model.kwargs["loss_function"] == "RMSEWithUncertainty"
    assert model.target_transformer.method == "box_cox"


def test_class_a_uses_horizon_lags_and_configured_transform():
    segmenter = _segmenter()
    model = ForecastRouter(2, "PRODUCE", segmenter).get_model(forecast_horizon=28)
    assert isinstance(model, HybridProphetCatBoost)
    assert model.catboost.lags == [-28, -35, -42, -56]
    assert model.catboost.lags_future_covariates == [0]
    assert model.catboost.lags_past_covariates is None
    assert model.target_transformer.method == "box_cox"


def test_gpu_request_without_device_falls_back(monkeypatch):
    import catboost.utils as catboost_utils

    monkeypatch.setattr(catboost_utils, "get_gpu_device_count", lambda: 0)
    assert resolve_task_type("GPU") == "CPU"
    assert resolve_task_type("CPU") == "CPU"


def test_box_cox_timeseries_round_trip_with_zeros():
    idx = pd.date_range("2023-01-01", periods=5, freq="D")
    raw = TimeSeries.from_times_and_values(idx, np.array([0, 1, 2, 3, 4], dtype=float), freq="D")
    transformer = TargetTransformer(method="box_cox")
    transformed, _, _ = transform_series(transformer, raw)
    restored = inverse_target(transformer, transformed, anchor_level=4.0)
    np.testing.assert_allclose(restored.values().flatten(), raw.values().flatten(), atol=1e-5)
    assert transformed.values().flatten()[0] != transformed.values().flatten()[1]


def test_box_cox_lambda_follows_the_training_window_only():
    rng = np.random.default_rng(1)
    train_values = rng.uniform(1.0, 8.0, size=30)
    full_values = np.concatenate([train_values, rng.uniform(400.0, 700.0, size=10)])
    idx = pd.date_range("2023-01-01", periods=len(full_values), freq="D")
    train = TimeSeries.from_times_and_values(idx[:30], train_values, freq="D")
    full = TimeSeries.from_times_and_values(idx, full_values, freq="D")

    leaked = TargetTransformer(method="box_cox").fit(pd.Series(full_values))
    window = TargetTransformer(method="box_cox")
    # A λ already fit on train+validation must be replaced by the training window.
    window._boxcox_lambda = leaked._boxcox_lambda
    window._is_fitted = True
    transform_series(window, train)

    train_only = TargetTransformer(method="box_cox").fit(pd.Series(train_values))
    assert window._boxcox_lambda == pytest.approx(train_only._boxcox_lambda)
    assert window._boxcox_fit_size == 30
    assert window._boxcox_lambda != pytest.approx(leaked._boxcox_lambda)
