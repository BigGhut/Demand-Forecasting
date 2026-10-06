"""The main evaluation scores several windows and the written baselines."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from darts import TimeSeries

import run_pipeline
from src.evaluation.backtest import (
    evaluate_forecasts,
    score_baseline_models,
    windows_ahead_of_best_baseline,
)
from src.models.routing import ABCSegmenter, training_history
from src.utils.config import get_base_config


def _weekly(n: int) -> TimeSeries:
    values = np.tile(np.arange(1, 8, dtype=float), n // 7 + 1)[:n]
    index = pd.date_range("2020-01-01", periods=n, freq="D")
    return TimeSeries.from_times_and_values(index, values, freq="D")


def test_seasonal_naive_repeats_last_week():
    series = _weekly(28)
    scores = score_baseline_models(series[:-7], series[-7:], horizon=7)
    assert set(scores) == {"seasonal_naive", "moving_average", "exponential_smoothing"}
    assert scores["seasonal_naive"]["MAE"] == pytest.approx(0.0)
    assert scores["moving_average"]["MAE"] > 0.0


def test_evaluate_forecasts_uses_several_windows_and_baselines():
    series = _weekly(51)
    extra_index = pd.date_range(series.end_time() + pd.Timedelta(days=1), periods=30, freq="D")
    cov_index = series.time_index.append(extra_index)
    covariates = TimeSeries.from_times_and_values(
        cov_index, np.ones(len(cov_index)), freq="D"
    )
    seen_ends: list[pd.Timestamp] = []

    def forecast_fn(train, val, future, shifted):
        del train, shifted
        seen_ends.append(future.end_time())
        assert future.end_time() == val.end_time()
        return val

    scores = evaluate_forecasts(
        series,
        horizon=7,
        stride=7,
        n_windows=3,
        forecast_fn=forecast_fn,
        future_covariates=covariates,
    )

    assert set(scores["model"]) == {
        "hybrid",
        "seasonal_naive",
        "moving_average",
        "exponential_smoothing",
    }
    assert scores["window"].nunique() == 3
    hybrid = scores[scores["model"] == "hybrid"]
    assert hybrid["MAE"].max() == pytest.approx(0.0)
    naive = scores[scores["model"] == "seasonal_naive"]
    assert naive["MAE"].max() == pytest.approx(0.0)
    assert all(start > end for start, end in zip(scores["val_start"], scores["train_end"]))
    assert seen_ends[-1] == series.end_time()
    assert covariates.end_time() > series.end_time()


def test_main_pipeline_asks_for_several_series_and_the_cv_config():
    source = inspect.getsource(run_pipeline.main)
    assert "_run_series" in source
    assert "evaluate_forecasts" in inspect.getsource(run_pipeline._run_series)
    assert "series_to_evaluate" in source
    chosen = run_pipeline.series_to_evaluate()
    assert len(chosen) >= 3
    assert (15, "BABY CARE") in chosen
    assert len({store for store, _ in chosen}) >= 2
    assert len({family for _, family in chosen}) >= 2
    pipeline = get_base_config().pipeline
    assert pipeline.cv_windows >= 2
    assert pipeline.residual_min_train_size == 365
    assert "log_baseline_wins" in source
    assert "routed_forecast" in source
    smoke = inspect.getsource(run_pipeline.smoke_test)
    assert "n_windows=1" in smoke
    assert "SMOKE_SERIES" in smoke
    assert run_pipeline.smoke_tail_length(28) == 120 + 3 * 28 + 28


def test_smoke_tail_keeps_the_recent_end():
    series = _weekly(300)
    covariates = TimeSeries.from_times_and_values(
        series.time_index, np.arange(len(series), dtype=float), freq="D"
    )
    tail, future, shifted = run_pipeline.slice_tail(series, covariates, covariates, 40)
    assert len(tail) == 40
    assert tail.end_time() == series.end_time()
    assert future.start_time() == tail.start_time()
    assert shifted.end_time() == tail.end_time()


def test_one_window_evaluation_runs():
    series = _weekly(40)

    def forecast_fn(train, val, future, shifted):
        del train, future, shifted
        return val

    scores = evaluate_forecasts(
        series, horizon=7, stride=7, n_windows=1, forecast_fn=forecast_fn
    )
    assert scores["window"].nunique() == 1
    assert set(scores["model"]) >= {"hybrid", "seasonal_naive"}


def test_win_count_ignores_ties_and_one_lucky_series():
    rows = []
    # Series A wins 1 of 2. Series B wins 2 of 2. A mean would hide that.
    plan = {
        (1, "PRODUCE"): [(10.0, 20.0), (30.0, 10.0)],
        (1, "BABY CARE"): [(5.0, 8.0), (4.0, 4.0)],
    }
    for (store, family), windows in plan.items():
        for window, (model_wape, best_wape) in enumerate(windows):
            for model, wape in (
                ("hybrid", model_wape),
                ("seasonal_naive", best_wape),
                ("moving_average", best_wape + 1.0),
                ("exponential_smoothing", best_wape + 2.0),
            ):
                rows.append(
                    {
                        "store_nbr": store,
                        "family": family,
                        "window": window,
                        "model": model,
                        "WAPE": wape,
                    }
                )
    scores = pd.DataFrame(rows)
    produce_wins, produce_n = windows_ahead_of_best_baseline(
        scores[scores["family"] == "PRODUCE"], "hybrid"
    )
    baby_wins, baby_n = windows_ahead_of_best_baseline(
        scores[scores["family"] == "BABY CARE"], "hybrid"
    )
    assert (produce_wins, produce_n) == (1, 2)
    # The second BABY CARE window ties the best baseline, so it is not a win.
    assert (baby_wins, baby_n) == (1, 2)


def test_abc_class_ignores_the_holdout_tail():
    dates = pd.date_range("2020-01-01", periods=35, freq="D")
    rows: list[dict[str, object]] = []
    for day in dates[:30]:
        rows.extend(
            [
                {"date": day, "store_nbr": 1, "family": "PRODUCE", "sales": 50.0},
                {"date": day, "store_nbr": 2, "family": "PRODUCE", "sales": 30.0},
                {"date": day, "store_nbr": 3, "family": "PRODUCE", "sales": 10.0},
            ]
        )
    for day in dates[30:]:
        rows.extend(
            [
                {"date": day, "store_nbr": 1, "family": "PRODUCE", "sales": 1.0},
                {"date": day, "store_nbr": 2, "family": "PRODUCE", "sales": 1.0},
                {"date": day, "store_nbr": 3, "family": "PRODUCE", "sales": 20000.0},
            ]
        )
    history = pd.DataFrame(rows)
    horizon = 5
    train_only = training_history(history, horizon)

    assert train_only["date"].max() == dates[29]
    assert ABCSegmenter(train_only).get_class(1, "PRODUCE") == "A"
    assert ABCSegmenter(history).get_class(1, "PRODUCE") == "C"

    app = Path(__file__).resolve().parents[1] / "app" / "main.py"
    text = app.read_text(encoding="utf-8")
    assert "training_history(" in text
    assert "abc_horizon" in text
    assert "ABCSegmenter(datasets['train'])" not in text
    assert 'ABCSegmenter(datasets["train"])' not in text
