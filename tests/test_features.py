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
    pipeline.validate_no_leakage(df, "sales", "date")

def test_feature_generation():
    """Test end-to-end feature generation on dummy data."""
    dates = pd.date_range("2023-01-01", periods=100)
    df = pd.DataFrame({
        "date": dates,
        "store_nbr": [1]*100,
        "family": ["PRODUCE"]*100,
        "sales": np.random.randn(100) * 10 + 100,
        "onpromotion": np.random.randint(0, 10, 100),
        "dcoilwtico": np.linspace(40, 50, 100)
    })
    
    pipeline = FeatureEngineeringPipeline(forecast_horizon=1) # safe horizon
    df_out = pipeline.fit_transform(df, target_col="sales", date_col="date")
    
    # Check that lags were generated
    assert "lag_1" in df_out.columns
    assert "lag_7" in df_out.columns
    
    # Check calendar features
    assert "day_of_week" in df_out.columns
    assert "is_weekend" in df_out.columns
