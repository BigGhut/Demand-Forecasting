"""Out-of-sample Prophet residuals for the CatBoost stage.

In-sample fitted errors are smaller and smoother than the errors Prophet
makes on a horizon it has not seen. CatBoost would then learn noise that
does not exist at forecast time.

Each origin fits a fresh Prophet on history that ends at ``t`` and scores
only ``t+1 ... t+horizon``. The default stride equals the horizon, so every
training day is forecast exactly once and the blocks concatenate into one
series. That series has to be longer than the residual lags (the longest
default lag is 56 days); a single 28-day block is not.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
import pandas as pd
from loguru import logger

from darts import TimeSeries

from src.models.prophet_model import ProphetForecaster


class OriginForecaster(Protocol):
    """Fit on a prefix and forecast the next block. Tests pass a stub."""

    def fit(
        self,
        series: TimeSeries,
        future_covariates: TimeSeries | None = None,
    ) -> Any: ...

    def predict(
        self,
        n: int,
        future_covariates: TimeSeries | None = None,
        num_samples: int = 1,
    ) -> TimeSeries: ...


def min_residual_length(lags: int | list[int] | None) -> int:
    """Points required before the longest lag has one training row."""
    if lags is None:
        return 2
    if isinstance(lags, int):
        return abs(lags) + 1
    flat: list[int] = []
    for lag in lags:
        if isinstance(lag, (list, tuple)):
            flat.extend(int(item) for item in lag)
        else:
            flat.append(int(lag))
    if not flat:
        return 2
    return max(abs(item) for item in flat) + 1


def rolling_origin_residuals(
    series: TimeSeries,
    future_covariates: TimeSeries | None,
    horizon: int,
    min_train_size: int,
    stride: int | None = None,
    forecaster_factory: Callable[[], OriginForecaster] | None = None,
) -> TimeSeries:
    """Forecast errors from origins that never trained on the scored days.

    Origins walk backward from the end of ``series`` so the last block ends
    on the last training day. CatBoost then forecasts from that same day.
    When ``stride`` is shorter than ``horizon``, overlapping days keep the
    error from the latest origin (the shortest lead).
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}.")
    if min_train_size <= 0:
        raise ValueError(f"min_train_size must be positive, got {min_train_size}.")
    step = horizon if stride is None else stride
    if step <= 0 or step > horizon:
        raise ValueError(
            f"residual stride must be in 1..horizon ({horizon}), got {step}."
        )

    n = len(series)
    origins = _origins(n, horizon, min_train_size, step)
    factory = forecaster_factory or _prophet_factory
    logger.info(
        "Building out-of-sample Prophet residuals: {} origins, "
        "horizon {}, stride {}, min train {}.",
        len(origins),
        horizon,
        step,
        min_train_size,
    )

    blocks: list[pd.Series] = []
    for origin in origins:
        train = series[:origin]
        actual = series[origin : origin + horizon]
        covariates = _cover(future_covariates, train.start_time(), actual.end_time())
        forecaster = factory()
        forecaster.fit(train, future_covariates=covariates)
        forecast = forecaster.predict(
            horizon, future_covariates=covariates, num_samples=1
        )
        blocks.append(_residual_block(actual, forecast, origin, horizon))

    return _stitch(blocks, series)


def _prophet_factory() -> ProphetForecaster:
    return ProphetForecaster()


def _origins(n: int, horizon: int, min_train_size: int, stride: int) -> list[int]:
    if n < min_train_size + horizon:
        raise ValueError(
            f"Series length {n} cannot hold a train of {min_train_size} "
            f"and a forecast block of {horizon}."
        )
    origins: list[int] = []
    origin = n - horizon
    while origin >= min_train_size:
        origins.append(origin)
        origin -= stride
    origins.reverse()
    if not origins:
        raise ValueError(
            f"No residual origin with train size >= {min_train_size} "
            f"on a series of length {n}."
        )
    return origins


def _cover(
    covariates: TimeSeries | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> TimeSeries | None:
    if covariates is None:
        return None
    if covariates.start_time() > start or covariates.end_time() < end:
        raise ValueError(
            "Future covariates must cover every residual origin, "
            f"from {start} through {end}. "
            f"They run from {covariates.start_time()} through {covariates.end_time()}."
        )
    return covariates.slice(start, end)


def _residual_block(
    actual: TimeSeries,
    forecast: TimeSeries,
    origin: int,
    horizon: int,
) -> pd.Series:
    forecast = forecast.slice_intersect(actual)
    actual = actual.slice_intersect(forecast)
    if len(actual) != horizon:
        raise ValueError(
            f"Origin {origin} produced {len(actual)} residual points, expected {horizon}."
        )
    actual_vals = np.asarray(actual.values(), dtype=float).reshape(len(actual))
    forecast_vals = np.asarray(forecast.values(), dtype=float).reshape(len(actual))
    residual = pd.Series(actual_vals - forecast_vals, index=actual.time_index, name="residual")
    return residual


def _stitch(blocks: list[pd.Series], series: TimeSeries) -> TimeSeries:
    # Later origins are appended last so an overlapping day keeps the shortest lead.
    combined = pd.concat(blocks)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    freq = series.freq or "D"
    expected = pd.date_range(combined.index[0], combined.index[-1], freq=freq)
    got = pd.DatetimeIndex(combined.index)
    if len(got) != len(expected) or not np.array_equal(got.asi8, expected.asi8):
        raise ValueError(
            "Rolling-origin residuals are not one continuous series. "
            "A stride longer than the horizon leaves days with no forecast error."
        )
    if got[-1] != series.end_time():
        raise ValueError(
            "Residual series must end on the last training day so the "
            f"forecast origin stays put. Residual end is {got[-1]}, "
            f"training end is {series.end_time()}."
        )
    return TimeSeries.from_series(combined, freq=freq)
