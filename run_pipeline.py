"""Score the routed model against the baselines on several series and windows.

The windows are the expanding splits from ``TimeSeriesCV`` (``cv_windows`` and
``cv_stride`` in the pipeline config). ``python run_pipeline.py --smoke`` fits
one series on one window of a short tail before that full report.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from loguru import logger

from darts import TimeSeries

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.evaluation.backtest import evaluate_forecasts, log_baseline_wins
from src.features.pipeline import FeatureEngineeringPipeline
from src.models.catboost_model import CatBoostResidualModel, horizon_safe_lags
from src.models.hybrid_pipeline import HybridProphetCatBoost
from src.models.prophet_model import ProphetForecaster
from src.models.routing import (
    ABCSegmenter,
    DirectCatBoostForecaster,
    ForecastRouter,
    history_until,
)
from src.utils.config import get_base_config


# High-volume families sit in class A. BABY CARE is sparse, so the report
# also fits DirectCatBoostForecaster. Store 1 never sells BABY CARE (the
# whole 2013–2017 series is zero). Store 15 sells it from mid-2014, with
# many zero days. BOOKS stays out: its sales start only in late 2016.
EVAL_SERIES: tuple[tuple[int, str], ...] = (
    (1, "PRODUCE"),
    (1, "GROCERY I"),
    (2, "BEVERAGES"),
    (15, "BABY CARE"),
)

# Smoke: one series, one window, enough tail for three residual blocks
# (lag -56 needs more than 56 out-of-sample points) and a short Prophet fit.
SMOKE_SERIES = (15, "BABY CARE")
SMOKE_HISTORY_DAYS = 400
SMOKE_MIN_TRAIN = 120
SMOKE_RESIDUAL_BLOCKS = 3


def series_to_evaluate() -> tuple[tuple[int, str], ...]:
    return EVAL_SERIES


def smoke_tail_length(horizon: int) -> int:
    """Points kept for the smoke run, including the one validation window."""
    return SMOKE_MIN_TRAIN + SMOKE_RESIDUAL_BLOCKS * horizon + horizon


def slice_tail(
    series: TimeSeries,
    future: TimeSeries | None,
    shifted: TimeSeries | None,
    n_points: int,
) -> tuple[TimeSeries, TimeSeries | None, TimeSeries | None]:
    """Keep the last ``n_points`` of the target and the matching covariates."""
    if len(series) < n_points:
        raise ValueError(
            f"Series has {len(series)} points after features; "
            f"the smoke tail needs {n_points}."
        )
    series = series[-n_points:]
    start = series.start_time()
    end = series.end_time()
    return series, _slice_span(future, start, end), _slice_span(shifted, start, end)


def prepare_series(
    datasets: dict,
    store_nbr: int,
    family: str,
    horizon: int,
    history_days: int | None = None,
) -> tuple[TimeSeries, TimeSeries | None, TimeSeries | None]:
    """One store-family series, with covariates that already cover the holdouts.

    The tail is not cut off here. ``TimeSeriesCV`` holds out each window.
    ``history_days`` keeps only the recent raw rows before features are built.
    """
    train_df = datasets["train"]
    subset = train_df[
        (train_df["store_nbr"] == store_nbr) & (train_df["family"] == family)
    ].copy()
    if history_days is not None:
        end = pd.to_datetime(subset["date"]).max()
        start = end - pd.Timedelta(days=history_days - 1)
        subset = subset.loc[pd.to_datetime(subset["date"]) >= start].copy()
    run_datasets = {key: value.copy() for key, value in datasets.items()}
    run_datasets["train"] = subset

    preprocessor = DataPreprocessor(aggregation_level="store_family")
    df_clean = preprocessor.preprocess(run_datasets)
    feature_pipeline = FeatureEngineeringPipeline(forecast_horizon=horizon)
    df_features = feature_pipeline.fit_transform(
        df_clean, target_col="sales", date_col="date"
    )

    prophet_covs = [
        col for col in feature_pipeline.future_covariate_names() if col in df_features.columns
    ]
    shifted_covs = [
        col for col in feature_pipeline.shifted_covariate_names() if col in df_features.columns
    ]
    df_features = df_features.dropna(subset=shifted_covs).reset_index(drop=True)
    df_features = df_features.select_dtypes(exclude=["category", "object"])
    prophet_covs = [col for col in prophet_covs if col in df_features.columns]
    shifted_covs = [col for col in shifted_covs if col in df_features.columns]
    if prophet_covs:
        df_features[prophet_covs] = df_features[prophet_covs].fillna(0)

    sales = TimeSeries.from_dataframe(
        df_features, time_col="date", value_cols="sales", freq="D"
    )
    future = (
        TimeSeries.from_dataframe(
            df_features, time_col="date", value_cols=prophet_covs, freq="D"
        )
        if prophet_covs
        else None
    )
    shifted = (
        TimeSeries.from_dataframe(
            df_features, time_col="date", value_cols=shifted_covs, freq="D"
        )
        if shifted_covs
        else None
    )
    return sales, future, shifted


def evaluation_catalog(datasets: dict) -> pd.DataFrame:
    """Sales of the evaluated series, used only to assign ABC class."""
    train_df = datasets["train"]
    parts: list[pd.DataFrame] = []
    for store_nbr, family in series_to_evaluate():
        part = train_df[
            (train_df["store_nbr"] == store_nbr) & (train_df["family"] == family)
        ]
        parts.append(part.loc[:, ["date", "store_nbr", "family", "sales"]])
    if not parts:
        raise ValueError("No series selected for evaluation.")
    return pd.concat(parts, ignore_index=True)


def predict_point(
    model: HybridProphetCatBoost | DirectCatBoostForecaster,
    train: TimeSeries,
    future_covariates: TimeSeries | None,
    shifted_covariates: TimeSeries | None,
    horizon: int,
    min_train: int | None = None,
) -> TimeSeries:
    """Fit one window and return the median forecast."""
    fit_kwargs: dict = {
        "future_covariates": future_covariates,
        "shifted_covariates": shifted_covariates,
    }
    if isinstance(model, HybridProphetCatBoost):
        fit_kwargs["residual_horizon"] = horizon
        if min_train is not None:
            fit_kwargs["residual_min_train_size"] = min_train
    model.fit(train, **fit_kwargs)
    preds = model.predict(
        n=horizon,
        future_covariates=future_covariates,
        shifted_covariates=shifted_covariates,
        num_samples=100,
    )
    return preds.quantile(0.5) if preds.is_probabilistic else preds


def routed_forecast(catalog: pd.DataFrame, store_nbr: int, family: str, horizon: int):
    """Pick the model from sales on or before this window's training end."""

    def forecast(train, val, future, shifted):
        del val
        history = history_until(catalog, train.end_time())
        model = ForecastRouter(store_nbr, family, ABCSegmenter(history)).get_model(
            forecast_horizon=horizon
        )
        name = "hybrid" if isinstance(model, HybridProphetCatBoost) else "direct_catboost"
        logger.info(
            "store {} {} train end {} -> {}",
            store_nbr,
            family,
            pd.Timestamp(train.end_time()).date(),
            name,
        )
        return predict_point(model, train, future, shifted, horizon), name

    return forecast


def _slice_span(
    covariates: TimeSeries | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> TimeSeries | None:
    if covariates is None:
        return None
    if covariates.end_time() < end or covariates.start_time() > start:
        raise ValueError(
            "Covariates do not cover the smoke tail: "
            f"{covariates.start_time()} .. {covariates.end_time()}, "
            f"tail {start} .. {end}."
        )
    return covariates.slice(start, covariates.end_time())


def _run_series(
    datasets: dict,
    store_nbr: int,
    family: str,
    horizon: int,
    stride: int,
    n_windows: int,
    forecast_fn,
    history_days: int | None = None,
    tail_points: int | None = None,
    model_name: str = "hybrid",
) -> pd.DataFrame:
    sales, future, shifted = prepare_series(
        datasets, store_nbr, family, horizon, history_days=history_days
    )
    if tail_points is not None:
        sales, future, shifted = slice_tail(sales, future, shifted, tail_points)
    scores = evaluate_forecasts(
        sales,
        horizon=horizon,
        stride=stride,
        n_windows=n_windows,
        forecast_fn=forecast_fn,
        future_covariates=future,
        shifted_covariates=shifted,
        model_name=model_name,
    )
    scores["store_nbr"] = store_nbr
    scores["family"] = family
    return scores


def main() -> pd.DataFrame:
    logger.info("Starting Demand Forecasting Hybrid Pipeline")
    pipeline = get_base_config().pipeline
    horizon = pipeline.forecast_horizon
    stride = pipeline.cv_stride
    n_windows = pipeline.cv_windows
    series_list = series_to_evaluate()
    if len(series_list) < 2:
        raise ValueError("The main evaluation needs at least two series.")
    if len({store for store, _ in series_list}) < 2 or len({family for _, family in series_list}) < 2:
        raise ValueError("Evaluate more than one store and more than one family.")
    if n_windows < 2:
        raise ValueError("The main evaluation needs at least two CV windows.")
    if not any(family == "BABY CARE" for _, family in series_list):
        raise ValueError("The evaluation needs a sparse family so class C is fitted.")

    datasets = DataLoader(data_dir=Path("data/raw")).load_all()
    catalog = evaluation_catalog(datasets)
    frames: list[pd.DataFrame] = []
    for store_nbr, family in series_list:
        logger.info("Series store {} / {}", store_nbr, family)
        scores = _run_series(
            datasets,
            store_nbr,
            family,
            horizon,
            stride,
            n_windows,
            forecast_fn=routed_forecast(catalog, store_nbr, family, horizon),
        )
        log_baseline_wins(scores)
        frames.append(scores)

    result = pd.concat(frames, ignore_index=True)
    summary = result.groupby("model")[["WAPE", "MAE", "RMSE"]].mean()
    logger.info("====== MEAN ACROSS SERIES AND WINDOWS ======\n{}", summary.to_string())
    log_baseline_wins(result)
    logger.success("End-to-End pipeline ran successfully!")
    return result


def smoke_test() -> pd.DataFrame:
    """One series, one window, truncated history.

    The hybrid cascade is fit explicitly: a single series is class C under
    ABC, and that would skip Prophet. The same window is then scored with
    the router, which on store 15 BABY CARE is DirectCatBoostForecaster.
    """
    pipeline = get_base_config().pipeline
    horizon = pipeline.forecast_horizon
    store_nbr, family = SMOKE_SERIES
    tail_points = smoke_tail_length(horizon)
    logger.info(
        "Smoke test: store {} {}, one window, last {} feature points "
        "(raw history {} days, residual min train {}).",
        store_nbr,
        family,
        tail_points,
        SMOKE_HISTORY_DAYS,
        SMOKE_MIN_TRAIN,
    )
    datasets = DataLoader(data_dir=Path("data/raw")).load_all()
    catalog = evaluation_catalog(datasets)

    def hybrid_fn(train, val, future, shifted):
        del val
        model = HybridProphetCatBoost(
            prophet_model=ProphetForecaster(),
            catboost_model=CatBoostResidualModel(
                lags=horizon_safe_lags(horizon),
                lags_future_covariates=[0],
            ),
        )
        return predict_point(
            model, train, future, shifted, horizon, min_train=SMOKE_MIN_TRAIN
        )

    hybrid_scores = _run_series(
        datasets,
        store_nbr,
        family,
        horizon,
        stride=pipeline.cv_stride,
        n_windows=1,
        history_days=SMOKE_HISTORY_DAYS,
        tail_points=tail_points,
        forecast_fn=hybrid_fn,
        model_name="hybrid",
    )

    def direct_fn(train, val, future, shifted):
        del val
        history = history_until(catalog, train.end_time())
        model = ForecastRouter(store_nbr, family, ABCSegmenter(history)).get_model(
            forecast_horizon=horizon
        )
        if not isinstance(model, DirectCatBoostForecaster):
            raise RuntimeError(
                f"Store {store_nbr} {family} routed to {type(model).__name__}. "
                "The smoke expects the class C direct model."
            )
        logger.info("Class C route: {}", type(model).__name__)
        return predict_point(model, train, future, shifted, horizon)

    direct_scores = _run_series(
        datasets,
        store_nbr,
        family,
        horizon,
        stride=pipeline.cv_stride,
        n_windows=1,
        history_days=SMOKE_HISTORY_DAYS,
        tail_points=tail_points,
        forecast_fn=direct_fn,
        model_name="direct_catboost",
    )
    result = pd.concat([hybrid_scores, direct_scores], ignore_index=True)
    summary = result.groupby("model")[["WAPE", "MAE", "RMSE"]].mean()
    logger.info("====== SMOKE MEAN ======\n{}", summary.to_string())
    log_baseline_wins(result)
    # The direct pass repeats the same baseline rows. One copy is enough in the log.
    logger.success("Smoke test finished.")
    return result


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        smoke_test()
    else:
        main()
