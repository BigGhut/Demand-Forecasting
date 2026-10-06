"""Rolling-origin residuals are forecast errors, not in-sample fitted errors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from darts import TimeSeries

from src.models.catboost_model import CatBoostResidualModel
from src.models.hybrid_pipeline import HybridProphetCatBoost
from src.models.prophet_model import ProphetForecaster, seasonality_mode_for
from src.models.residual_origins import rolling_origin_residuals


class _SpyForecaster:
    """Forecasts a constant equal to the number of training points."""

    def __init__(self, created: list["_SpyForecaster"], value: float | None = None) -> None:
        self.created = created
        self._fixed_value = value
        self.train_index: pd.DatetimeIndex | None = None
        self.predicted_index: pd.DatetimeIndex | None = None
        self.seen_values: pd.Series | None = None

    def fit(self, series: TimeSeries, future_covariates: TimeSeries | None = None) -> "_SpyForecaster":
        del future_covariates
        raw = series.to_series()
        if isinstance(raw, pd.DataFrame):
            raw = raw.iloc[:, 0]
        self.train_index = pd.DatetimeIndex(series.time_index)
        self.seen_values = raw
        return self

    def predict(
        self,
        n: int,
        future_covariates: TimeSeries | None = None,
        num_samples: int = 1,
    ) -> TimeSeries:
        del future_covariates, num_samples
        assert self.train_index is not None
        start = self.train_index[-1] + pd.Timedelta(days=1)
        index = pd.date_range(start, periods=n, freq="D")
        self.predicted_index = index
        self.created.append(self)
        level = float(len(self.train_index) if self._fixed_value is None else self._fixed_value)
        return TimeSeries.from_times_and_values(index, np.full(n, level), freq="D")


def _series(n: int, start: str = "2020-01-01") -> TimeSeries:
    index = pd.date_range(start, periods=n, freq="D")
    values = np.arange(n, dtype=float) + 10.0
    return TimeSeries.from_times_and_values(index, values, freq="D")


def _producing_spy(spies: list[_SpyForecaster], timestamp: pd.Timestamp) -> _SpyForecaster:
    candidates = [
        spy
        for spy in spies
        if spy.predicted_index is not None and timestamp in spy.predicted_index
    ]
    assert candidates
    return max(candidates, key=lambda spy: spy.train_index[-1])


def test_multiplicative_prophet_becomes_additive_on_non_positive_values():
    index = pd.date_range("2020-01-01", periods=4, freq="D")
    zeros = TimeSeries.from_times_and_values(index, np.zeros(4), freq="D")
    positive = TimeSeries.from_times_and_values(index, np.arange(1, 5, dtype=float), freq="D")
    assert seasonality_mode_for(zeros, "multiplicative") == "additive"
    assert seasonality_mode_for(positive, "multiplicative") == "multiplicative"
    assert seasonality_mode_for(zeros, "additive") == "additive"


def test_overlapping_origins_keep_the_shortest_lead():
    spies: list[_SpyForecaster] = []
    series = _series(16)

    residuals = rolling_origin_residuals(
        series,
        future_covariates=None,
        horizon=4,
        min_train_size=8,
        stride=2,
        forecaster_factory=lambda: _SpyForecaster(spies),
    )

    # Index 11 is inside the origin-8 block and the origin-10 block.
    # The later origin trained on 10 points and did not see day 11.
    day = series.time_index[11]
    spy = _producing_spy(spies, day)
    assert len(spy.train_index) == 10
    assert day not in spy.train_index
    assert residuals.time_index[-1] == series.end_time()
    raw = series.to_series()
    got = residuals.to_series()
    assert got.loc[day] == pytest.approx(float(raw.loc[day]) - 10.0)


def test_hybrid_does_not_train_catboost_on_insample_residuals(monkeypatch):
    calls = {"insample": 0}

    def _insample(*_args, **_kwargs):
        calls["insample"] += 1
        raise AssertionError("in-sample residuals were requested")

    monkeypatch.setattr(ProphetForecaster, "get_residuals", _insample)

    captured: dict[str, TimeSeries] = {}

    def _capture(self, series, past_covariates=None, future_covariates=None):
        del past_covariates, future_covariates
        captured["residuals"] = series
        self._is_fitted = True
        return self

    monkeypatch.setattr(CatBoostResidualModel, "fit", _capture)

    origin_spies: list[_SpyForecaster] = []
    series = _series(22)
    deployment = _SpyForecaster([], value=0.0)
    model = HybridProphetCatBoost(
        prophet_model=deployment,
        catboost_model=CatBoostResidualModel(lags=[-4, -8], lags_future_covariates=[0]),
    )
    model.fit(
        series,
        residual_horizon=4,
        residual_min_train_size=10,
        residual_stride=4,
        forecaster_factory=lambda: _SpyForecaster(origin_spies, value=0.0),
    )

    assert calls["insample"] == 0
    # The deployed Prophet sees the whole training window, including days
    # that were held out from the prefix models which labeled the residuals.
    assert deployment.train_index is not None
    assert len(deployment.train_index) == len(series)
    transformed = deployment.seen_values
    residuals = captured["residuals"].to_series()
    assert len(residuals) == 12
    assert residuals.index[-1] == series.end_time()
    assert residuals.index[0] == series.time_index[10]
    for timestamp, value in residuals.items():
        spy = _producing_spy(origin_spies, timestamp)
        assert timestamp not in spy.train_index
        assert value == pytest.approx(float(transformed.loc[timestamp]))


def test_short_oos_residuals_are_not_replaced_with_insample(monkeypatch):
    monkeypatch.setattr(
        ProphetForecaster,
        "get_residuals",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("in-sample residuals were requested")
        ),
    )
    monkeypatch.setattr(
        CatBoostResidualModel,
        "fit",
        lambda self, *args, **kwargs: self,
    )
    series = _series(14)
    model = HybridProphetCatBoost(
        prophet_model=_SpyForecaster([], value=0.0),
        catboost_model=CatBoostResidualModel(lags=[-4, -8], lags_future_covariates=[0]),
    )
    with pytest.raises(ValueError, match="not used as a fallback"):
        model.fit(
            series,
            residual_horizon=4,
            residual_min_train_size=10,
            residual_stride=4,
            forecaster_factory=lambda: _SpyForecaster([], value=0.0),
        )
