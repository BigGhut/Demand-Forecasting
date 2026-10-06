import pytest
import numpy as np
import pandas as pd
from darts import TimeSeries
from src.evaluation.metrics import BusinessMetrics

def create_ts(values):
    return TimeSeries.from_dataframe(pd.DataFrame({'date': pd.date_range('2023-01-01', periods=len(values)), 'val': values}), time_col='date')

def test_peak_capture_rate():
    y_true = create_ts([10, 20, 100, 10])
    
    y_pred1 = create_ts([15, 15, 100, 15])
    assert BusinessMetrics.peak_capture_rate(y_true, y_pred1, threshold_percentile=90) == 1.0
    
    y_pred2 = create_ts([15, 15, 50, 15])
    # 50 is below the 90th percentile of the actuals, so the peak is missed.
    assert BusinessMetrics.peak_capture_rate(y_true, y_pred2, threshold_percentile=90) == 0.0

def test_mape_skips_zero_actuals():
    y_true = create_ts([0, 100, 0])
    y_pred = create_ts([5, 110, 0])
    metrics = BusinessMetrics.standard_metrics(y_true, y_pred)
    assert metrics["MAPE"] == pytest.approx(10.0)

def test_direction_accuracy():
    y_true = create_ts([10, 20, 100, 10])
    
    y_pred1 = create_ts([15, 20, 100, 10])
    assert BusinessMetrics.direction_accuracy(y_true, y_pred1) == 1.0
    
    y_pred2 = create_ts([15, 10, -70, -60])
    assert BusinessMetrics.direction_accuracy(y_true, y_pred2) == 0.0

def test_standard_metrics():
    y_true = create_ts([100, 200, 300])
    y_pred = create_ts([90, 210, 300])
    
    metrics = BusinessMetrics.standard_metrics(y_true, y_pred)
    
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "MAPE" in metrics
    assert "sMAPE" in metrics
