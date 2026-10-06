"""Expanding-window scores for the hybrid model and the written baselines."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from loguru import logger

from darts import TimeSeries

from src.evaluation.cross_validation import TimeSeriesCV
from src.evaluation.metrics import BusinessMetrics
from src.models.baseline import BaselineModels


ForecastFn = Callable[
    [TimeSeries, TimeSeries, TimeSeries | None, TimeSeries | None],
    TimeSeries | tuple[TimeSeries, str],
]

BASELINE_MODELS = ("seasonal_naive", "moving_average", "exponential_smoothing")


def evaluate_forecasts(
    series: TimeSeries,
    horizon: int,
    stride: int,
    n_windows: int,
    forecast_fn: ForecastFn,
    future_covariates: TimeSeries | None = None,
    shifted_covariates: TimeSeries | None = None,
    model_name: str = "hybrid",
) -> pd.DataFrame:
    """Score ``forecast_fn`` and the three baselines on the same windows.

    Windows come from :class:`TimeSeriesCV`: validation is ``horizon`` days,
    and the training prefix grows by ``stride``. The full report uses several
    windows. A smoke run passes one window and a short series.

    ``forecast_fn`` may return ``(prediction, model_name)`` when the routed
    model changes from window to window.
    """
    if n_windows < 1:
        raise ValueError(f"n_windows must be positive, got {n_windows}.")
    splits = TimeSeriesCV(
        forecast_horizon=horizon,
        stride=stride,
        n_windows=n_windows,
    ).generate_splits(series)
    if len(splits) != n_windows:
        raise ValueError(
            f"TimeSeriesCV returned {len(splits)} windows, expected {n_windows}."
        )

    rows: list[dict[str, object]] = []
    for i, (train, val) in enumerate(splits):
        if val.start_time() <= train.end_time():
            raise ValueError(
                f"Window {i} validation starts at {val.start_time()}, "
                f"which is not after training end {train.end_time()}."
            )
        future_w = _slice_window(future_covariates, train, val)
        shifted_w = _slice_window(shifted_covariates, train, val)
        predicted = forecast_fn(train, val, future_w, shifted_w)
        if isinstance(predicted, tuple):
            prediction, name = predicted
        else:
            prediction, name = predicted, model_name
        scored = {name: prediction}
        scored.update(_baseline_forecasts(train, horizon))
        for name, pred in scored.items():
            metrics = forecast_metrics(val, pred)
            logger.info(
                "window {}/{} {} WAPE={:.2f} MAE={:.2f}",
                i + 1,
                n_windows,
                name,
                metrics["WAPE"],
                metrics["MAE"],
            )
            rows.append(
                {
                    "window": i,
                    "train_end": train.end_time(),
                    "val_start": val.start_time(),
                    "val_end": val.end_time(),
                    "model": name,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def windows_ahead_of_best_baseline(
    scores: pd.DataFrame,
    model_name: str,
) -> tuple[int, int]:
    """Windows where ``model_name`` has a strictly lower WAPE than every baseline.

    A tie is not a win. Windows with a missing WAPE are left out of the count.
    """
    keys = ["window"]
    if "store_nbr" in scores.columns and "family" in scores.columns:
        keys = ["store_nbr", "family", "window"]
    learned = scores.loc[scores["model"] == model_name, keys + ["WAPE"]]
    baselines = scores.loc[scores["model"].isin(BASELINE_MODELS), keys + ["WAPE"]]
    if learned.empty or baselines.empty:
        return 0, 0
    best = baselines.groupby(keys, sort=False)["WAPE"].min()
    learned_wape = learned.groupby(keys, sort=False)["WAPE"].min()
    aligned = pd.concat(
        [learned_wape.rename("model"), best.rename("best")],
        axis=1,
    ).dropna()
    wins = int((aligned["model"] < aligned["best"]).sum())
    return wins, int(len(aligned))


def log_baseline_wins(scores: pd.DataFrame) -> None:
    """Log, per series and model, how many windows beat the best baseline."""
    group_cols = [col for col in ("store_nbr", "family") if col in scores.columns]
    if not group_cols:
        _log_wins_for_block(scores, label="")
        return
    for key, block in scores.groupby(group_cols, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        _log_wins_for_block(block, label=" ".join(str(part) for part in key))


def _log_wins_for_block(scores: pd.DataFrame, label: str) -> None:
    learned = [
        name for name in scores["model"].unique() if name not in BASELINE_MODELS
    ]
    prefix = f"{label}: " if label else ""
    for name in learned:
        wins, total = windows_ahead_of_best_baseline(scores, name)
        logger.info(
            "{}{} beat the best baseline on {}/{} windows",
            prefix,
            name,
            wins,
            total,
        )


def score_baseline_models(
    train: TimeSeries,
    actual: TimeSeries,
    horizon: int,
) -> dict[str, dict[str, float]]:
    """Fit seasonal naive (K=7), moving average, and exponential smoothing."""
    forecasts = _baseline_forecasts(train, horizon)
    return {name: forecast_metrics(actual, pred) for name, pred in forecasts.items()}


def forecast_metrics(actual: TimeSeries, pred: TimeSeries) -> dict[str, float]:
    """Point metrics on the overlapping dates. Probabilistic forecasts use the median."""
    point = pred.quantile(0.5) if pred.is_probabilistic else pred
    common_actual = actual.slice_intersect(point)
    common_pred = point.slice_intersect(common_actual)
    if len(common_actual) == 0:
        raise ValueError("Forecast and actuals do not overlap.")
    metrics = BusinessMetrics.standard_metrics(common_actual, common_pred)
    metrics["Direction Accuracy"] = BusinessMetrics.direction_accuracy(
        common_actual, common_pred
    )
    metrics["Peak Capture Rate"] = BusinessMetrics.peak_capture_rate(
        common_actual, common_pred, threshold_percentile=90
    )
    return metrics


def _baseline_forecasts(train: TimeSeries, horizon: int) -> dict[str, TimeSeries]:
    specs = (
        ("seasonal_naive", BaselineModels.get_seasonal_naive(K=7)),
        ("moving_average", BaselineModels.get_moving_average(input_chunk_length=14)),
        ("exponential_smoothing", BaselineModels.get_exponential_smoothing(seasonal_periods=7)),
    )
    forecasts: dict[str, TimeSeries] = {}
    for name, model in specs:
        model.fit(train)
        forecasts[name] = model.predict(horizon)
    return forecasts


def _slice_window(
    covariates: TimeSeries | None,
    train: TimeSeries,
    val: TimeSeries,
) -> TimeSeries | None:
    """Covariates for one window: training history through the validation end.

    Dates after this validation window stay out. A later window's sales are
    not a covariate for this fit.
    """
    if covariates is None:
        return None
    if covariates.end_time() < val.end_time():
        raise ValueError(
            "Covariates end before the validation window: "
            f"{covariates.end_time()} < {val.end_time()}."
        )
    start = train.start_time()
    if covariates.start_time() > start:
        start = covariates.start_time()
    return covariates.slice(start, val.end_time())
