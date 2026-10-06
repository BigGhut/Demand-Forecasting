import pytest
import pandas as pd
import numpy as np

from src.features.pipeline import FeatureEngineeringPipeline
from src.features.lag_features import LagFeatureGenerator

def test_leakage_detection():
    """Test that the pipeline correctly identifies data leakage."""
    # Forecast horizon is 28, but we pass lag=1, which is illegal.
    # The new behavior filters it out and falls back to [28]
    gen = LagFeatureGenerator(lags=[1, 7, 14], forecast_horizon=28)
    assert gen.lags == [28]

def test_pipeline_valid_lags(monkeypatch):
    """Test that pipeline works with valid lags."""
    from src.utils.config import FeatureConfig
    # Mock config to have valid lags
    mock_config = FeatureConfig(lags=[28, 35])

    # We patch the default get_feature_config to return our mock
    monkeypatch.setattr('src.features.pipeline.get_feature_config', lambda: mock_config)

    pipeline = FeatureEngineeringPipeline(forecast_horizon=28)
    df = pd.DataFrame({"date": [1], "sales": [1]})
    assert pipeline.validate_no_leakage(df, "sales", "date") is True

def test_feature_generation():
    """Horizon 28 must keep only lags that are known at the forecast origin."""
    dates = pd.date_range("2023-01-01", periods=120)
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "date": dates,
        "store_nbr": [1] * 120,
        "family": ["PRODUCE"] * 120,
        "sales": rng.uniform(1, 100, size=120),
        "onpromotion": rng.integers(0, 10, size=120),
        "dcoilwtico": np.linspace(40, 50, 120),
        "transactions": rng.uniform(100, 200, size=120),
    })

    pipeline = FeatureEngineeringPipeline(forecast_horizon=28)
    df_out = pipeline.fit_transform(df, target_col="sales", date_col="date")

    assert "lag_1" not in df_out.columns
    assert "lag_7" not in df_out.columns
    assert "lag_28" in df_out.columns
    assert "oil_lag_1" not in df_out.columns
    assert "oil_lag_28" in df_out.columns
    assert "transactions_lag_1" not in df_out.columns
    assert "transactions_lag_28" in df_out.columns
    assert "rolling_mean_lag28_w7" in df_out.columns
    assert "expanding_mean" in df_out.columns
    assert "day_of_week" in df_out.columns
    assert "is_weekend" in df_out.columns

    safe = df_out["sales"].shift(28).expanding(min_periods=28).mean()
    raw = df_out["sales"].expanding(min_periods=28).mean()
    mask = df_out["expanding_mean"].notna()
    np.testing.assert_allclose(df_out.loc[mask, "expanding_mean"], safe[mask])
    assert not np.allclose(df_out.loc[mask, "expanding_mean"], raw[mask])

    future = pipeline.future_covariate_names()
    shifted = pipeline.shifted_covariate_names()
    assert "day_of_week" in future
    assert "oil_lag_28" in future
    assert not any(name.startswith(("lag_", "rolling_", "expanding_", "transactions_")) for name in future)
    assert "lag_28" in shifted
    assert "expanding_mean" in shifted
    assert "transactions_lag_28" in shifted

def test_short_base_lags_are_replaced_not_dropped():
    from src.features.rolling_features import RollingFeatureGenerator

    gen = RollingFeatureGenerator(base_lags=[1, 7], windows=[7], functions=["mean"], forecast_horizon=28)
    assert gen.base_lags == [28]
    df = pd.DataFrame({"sales": np.arange(40, dtype=float)})
    out = gen.transform(df, target_col="sales")
    assert "rolling_mean_lag28_w7" in out.columns

def test_constant_sales_are_not_flagged_as_an_expanding_leak():
    dates = pd.date_range("2023-01-01", periods=120)
    df = pd.DataFrame({
        "date": dates,
        "store_nbr": [1] * 120,
        "family": ["BABY CARE"] * 120,
        "sales": np.zeros(120),
        "onpromotion": np.zeros(120),
        "dcoilwtico": np.full(120, 40.0),
        "transactions": np.zeros(120),
    })
    pipeline = FeatureEngineeringPipeline(forecast_horizon=28)
    out = pipeline.fit_transform(df, target_col="sales", date_col="date")
    assert "expanding_mean" in out.columns


def test_fit_transform_rejects_failed_validation(monkeypatch):
    pipeline = FeatureEngineeringPipeline(forecast_horizon=28)
    monkeypatch.setattr(pipeline, "validate_no_leakage", lambda *args, **kwargs: False)
    dates = pd.date_range("2023-01-01", periods=40)
    df = pd.DataFrame({"date": dates, "sales": np.arange(40, dtype=float) + 1.0})
    with pytest.raises(ValueError, match="Leakage validation failed"):
        pipeline.fit_transform(df, target_col="sales", date_col="date")
