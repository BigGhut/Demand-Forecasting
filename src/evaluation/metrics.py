from __future__ import annotations

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.metrics import mae, rmse, smape, r2_score

class BusinessMetrics:
    """Custom business-oriented metrics for demand forecasting."""
    
    @staticmethod
    def standard_metrics(actual: TimeSeries, pred: TimeSeries) -> dict[str, float]:
        """Calculate standard forecasting metrics."""
        return {
            "MAE": mae(actual, pred),
            "RMSE": rmse(actual, pred),
            "MAPE": BusinessMetrics.mape(actual, pred),
            "sMAPE": smape(actual, pred),
            "R2": r2_score(actual, pred),
            "WAPE": BusinessMetrics.wape(actual, pred),
            "PBIAS": BusinessMetrics.pbias(actual, pred),
            "Tracking Signal": BusinessMetrics.tracking_signal(actual, pred),
            "CV": BusinessMetrics.coefficient_of_variation(actual)
        }
        
    @staticmethod
    def mape(actual: TimeSeries, pred: TimeSeries) -> float:
        """Mean absolute percentage error, skipping days where the actual is zero.

        Darts ``mape`` raises ``ValueError`` when any actual is zero. The
        preprocessor zero-fills missing days, so that implementation cannot
        be the pipeline metric.
        """
        y_true = actual.values().flatten()
        y_pred = pred.values().flatten()
        mask = y_true != 0
        if not np.any(mask):
            return np.nan
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)

    @staticmethod
    def wape(actual: TimeSeries, pred: TimeSeries) -> float:
        y_true = actual.values().flatten()
        y_pred = pred.values().flatten()
        sum_actual = np.sum(np.abs(y_true))
        if sum_actual == 0:
            return np.nan
        return float(np.sum(np.abs(y_true - y_pred)) / sum_actual) * 100.0
        
    @staticmethod
    def pbias(actual: TimeSeries, pred: TimeSeries) -> float:
        y_true = actual.values().flatten()
        y_pred = pred.values().flatten()
        sum_actual = np.sum(y_true)
        if sum_actual == 0:
            return np.nan
        # Positive PBIAS means over-forecasting, negative means under-forecasting
        return float(np.sum(y_pred - y_true) / sum_actual) * 100.0
        
    @staticmethod
    def tracking_signal(actual: TimeSeries, pred: TimeSeries) -> float:
        y_true = actual.values().flatten()
        y_pred = pred.values().flatten()
        errors = y_true - y_pred
        cfe = np.sum(errors)
        mad = np.mean(np.abs(errors))
        if mad == 0:
            return 0.0
        return float(cfe / mad)
        
    @staticmethod
    def coefficient_of_variation(actual: TimeSeries) -> float:
        y_true = actual.values().flatten()
        mean_val = np.mean(y_true)
        if mean_val == 0:
            return np.nan
        return float(np.std(y_true) / mean_val)
        
    @staticmethod
    def direction_accuracy(actual: TimeSeries, pred: TimeSeries) -> float:
        """Percentage of time the forecast correctly predicts the direction of change."""
        y_true = actual.values().flatten()
        y_pred = pred.values().flatten()
        
        if len(y_true) < 2:
            return np.nan
            
        true_diff = np.diff(y_true)
        pred_diff = np.diff(y_pred)
        
        correct_direction = np.sign(true_diff) == np.sign(pred_diff)
        return float(np.mean(correct_direction))
        
    @staticmethod
    def peak_capture_rate(actual: TimeSeries, pred: TimeSeries, threshold_percentile: float = 90.0) -> float:
        """Percentage of actual peaks that were correctly forecasted as peaks.
        
        A peak is a value above the ``threshold_percentile`` of the actual series.
        The forecast captures that peak when its value on the same day clears
        the same actual threshold.
        """
        y_true = actual.values().flatten()
        y_pred = pred.values().flatten()

        if len(y_true) == 0:
            return np.nan

        threshold = np.percentile(y_true, threshold_percentile)

        actual_peaks_idx = np.where(y_true > threshold)[0]
        if len(actual_peaks_idx) == 0:
            return 1.0  # No peaks to miss

        captured = np.sum(y_pred[actual_peaks_idx] > threshold)

        return float(captured / len(actual_peaks_idx))
