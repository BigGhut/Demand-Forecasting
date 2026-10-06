from __future__ import annotations

import os

import numpy as np
import pandas as pd
from loguru import logger

from darts import TimeSeries
from darts.models import Prophet

from src.utils.config import ProphetConfig, get_base_config


def prepare_stan_path() -> None:
    """Put CmdStan's TBB directory on PATH before Prophet is constructed.

    CmdStanPy checks for ``tbb.dll`` with ``where.exe``. On a Russian Windows
    the not-found message is cp866, and CmdStanPy reads it as UTF-8, so the
    Stan backend never loads. Finding the DLL first skips that message.
    """
    if os.name != "nt":
        return
    try:
        from cmdstanpy.utils import cmdstan_path

        root = cmdstan_path()
    except Exception:
        return
    tbb = os.path.join(root, "stan", "lib", "stan_math", "lib", "tbb")
    if not os.path.isdir(tbb):
        return
    current = os.environ.get("PATH", "")
    parts = [part.lower() for part in current.split(os.pathsep)]
    if tbb.lower() not in parts:
        os.environ["PATH"] = tbb + os.pathsep + current


def seasonality_mode_for(series: TimeSeries, configured: str) -> str:
    """Multiplicative Prophet seasonality is undefined at zero and below.

    Box-Cox maps an original zero to 0, so a sparse family reaches Prophet
    with zeros even after the +1 shift. Additive seasonality still fits.
    """
    values = np.asarray(series.values(), dtype=float)
    if configured == "multiplicative" and np.any(values <= 0):
        return "additive"
    return configured

class ProphetForecaster:
    """Prophet model wrapper using Darts integration."""
    
    def __init__(self, config: ProphetConfig | None = None) -> None:
        prepare_stan_path()
        self.config = config or get_base_config().prophet
        
        # Build kwargs for Prophet based on our config
        self.prophet_kwargs = {
            "seasonality_mode": self.config.seasonality_mode,
            "changepoint_prior_scale": self.config.changepoint_prior_scale,
            "yearly_seasonality": self.config.yearly_seasonality,
            "weekly_seasonality": self.config.weekly_seasonality,
            "daily_seasonality": self.config.daily_seasonality,
        }
        
        self.model = Prophet(
            add_seasonalities=None, # can be added if custom seasonalities are needed
            country_holidays=self.config.holidays_country,
            suppress_stdout_stderror=True,
            **self.prophet_kwargs
        )
        self._is_fitted = False
        
    def fit(self, series: TimeSeries, future_covariates: TimeSeries | None = None) -> ProphetForecaster:
        """Fit Prophet model on the given target series."""
        prepare_stan_path()
        mode = seasonality_mode_for(series, self.config.seasonality_mode)
        if mode != self.prophet_kwargs["seasonality_mode"]:
            logger.warning(
                "Prophet seasonality_mode={} cannot fit non-positive values. Using {}.",
                self.prophet_kwargs["seasonality_mode"],
                mode,
            )
            self.prophet_kwargs = {**self.prophet_kwargs, "seasonality_mode": mode}
            self.model = Prophet(
                add_seasonalities=None,
                country_holidays=self.config.holidays_country,
                suppress_stdout_stderror=True,
                **self.prophet_kwargs,
            )
        logger.info(f"Fitting Prophet model with config: {self.config}")
        self.model.fit(series, future_covariates=future_covariates)
        self._is_fitted = True
        return self
        
    def predict(self, n: int, future_covariates: TimeSeries | None = None,
                num_samples: int = 1) -> TimeSeries:
        """Forecast the next n periods."""
        if not self._is_fitted:
            raise ValueError("Model is not fitted yet.")
        return self.model.predict(n=n, future_covariates=future_covariates, num_samples=num_samples)
        
    def get_residuals(self, series: TimeSeries, future_covariates: TimeSeries | None = None) -> TimeSeries:
        """In-sample fitted errors on the series this Prophet was just trained on.

        These are smaller and smoother than horizon forecast errors. CatBoost
        in the hybrid pipeline does not train on them; it uses
        ``rolling_origin_residuals``. This method is only a fitted-value check.
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before extracting residuals.")
            
        logger.warning(
            "Computing in-sample Prophet residuals. Do not train CatBoost on these."
        )
        
        prophet_model = self.model.model
        
        # Construct DataFrame for Prophet predict
        df = pd.DataFrame({'ds': series.time_index})
        if future_covariates is not None:
            cov_df = future_covariates.to_dataframe().reset_index()
            # Darts uses 'date' or similar index, ensure we merge on it
            # Rename the time column to 'ds' for Prophet
            time_col = cov_df.columns[0]
            cov_df = cov_df.rename(columns={time_col: 'ds'})
            
            # Flatten multi-index columns if they exist from Darts to_dataframe
            if isinstance(cov_df.columns, pd.MultiIndex):
                cov_df.columns = ['ds' if c[0] == 'ds' else c[0] for c in cov_df.columns]
                
            df = df.merge(cov_df, on='ds', how='left')
            
            if df.isna().any().any():
                logger.error(f"NaNs found after merge! df shape: {df.shape}, cov_df shape: {cov_df.shape}")
                logger.error(f"df ds head: {df['ds'].head().tolist()}")
                logger.error(f"cov ds head: {cov_df['ds'].head().tolist()}")
                logger.error(f"df ds dtype: {df['ds'].dtype}, cov ds dtype: {cov_df['ds'].dtype}")
                logger.error(f"First match? {df['ds'].iloc[0] == cov_df['ds'].iloc[0]}")
                logger.error(f"Are df dates in cov dates? {df['ds'].isin(cov_df['ds']).sum()} matched")
            
        # Predict in-sample
        preds_df = prophet_model.predict(df)
        
        # Create a TimeSeries for the fitted values
        in_sample_pred = TimeSeries.from_dataframe(preds_df, time_col='ds', value_cols='yhat')
        
        # Compute residuals
        actual_sliced = series.slice_intersect(in_sample_pred)
        in_sample_pred = in_sample_pred.slice_intersect(actual_sliced)
        
        residuals = actual_sliced - in_sample_pred
        logger.info("Residual extraction complete.")
        return residuals
