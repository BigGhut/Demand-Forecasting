from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any
from loguru import logger
from darts import TimeSeries
import statsmodels.api as sm

class ResidualDiagnostics:
    """Tools for diagnosing model residuals."""
    
    @staticmethod
    def ljung_box_test(residuals: TimeSeries, lags: int = 14) -> dict[str, Any]:
        """Perform Ljung-Box test for autocorrelation in residuals.
        
        H0: The data are independently distributed (no autocorrelation).
        If p_value < 0.05, we reject H0 (meaning residuals have autocorrelation).
        """
        res_vals = residuals.values().flatten()
        if len(res_vals) <= lags:
            raise ValueError("Not enough data points for the requested number of lags.")
            
        # Drop NaNs just in case
        res_vals = res_vals[~np.isnan(res_vals)]
        
        lb_test = sm.stats.acorr_ljungbox(res_vals, lags=[lags], return_df=True)
        p_val = lb_test['lb_pvalue'].iloc[0]
        stat = lb_test['lb_stat'].iloc[0]
        
        is_white_noise = p_val > 0.05
        
        return {
            "statistic": float(stat),
            "p_value": float(p_val),
            "lags": lags,
            "is_white_noise": is_white_noise
        }
