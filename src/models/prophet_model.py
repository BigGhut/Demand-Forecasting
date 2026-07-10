from __future__ import annotations

import pandas as pd
from loguru import logger

from darts import TimeSeries
from darts.models import Prophet

from src.utils.config import ProphetConfig, get_base_config

class ProphetForecaster:
    """Prophet model wrapper using Darts integration."""
    
    def __init__(self, config: ProphetConfig | None = None) -> None:
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
        """Extract residuals for the training series.
        This is critical for the CatBoost residual learning step.
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before extracting residuals.")
            
        logger.info("Computing in-sample residuals using underlying Prophet model.")
        
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
