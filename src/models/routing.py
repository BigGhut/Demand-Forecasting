from __future__ import annotations

import pandas as pd
from loguru import logger
from darts import TimeSeries

from src.models.hybrid_pipeline import (
    HybridProphetCatBoost,
    _as_series,
    inverse_target,
    stack_covariates,
    transform_series,
)
from src.models.catboost_model import CatBoostResidualModel, horizon_safe_lags
from src.models.prophet_model import ProphetForecaster
from src.transforms.target_transforms import TargetTransformer
from src.utils.config import get_base_config


class DirectCatBoostForecaster:
    """Class C model with the same fit/predict signature as the hybrid.

    Prophet is skipped. Calendar covariates and pre-shifted history are both
    future covariates: with output length 1, lag 0 reads the column on the
    date being forecast.
    """

    def __init__(self,
                 catboost_model: CatBoostResidualModel,
                 target_transformer: TargetTransformer | None = None) -> None:
        self.catboost = catboost_model
        if target_transformer is None:
            method = get_base_config().pipeline.target_transform
            target_transformer = TargetTransformer(method=method)
        self.target_transformer = target_transformer
        self._anchor_level: float | None = None
        self._is_fitted = False

    def fit(self,
            series: TimeSeries,
            future_covariates: TimeSeries | None = None,
            shifted_covariates: TimeSeries | None = None) -> DirectCatBoostForecaster:
        self._anchor_level = float(_as_series(series).iloc[-1])
        # New parameters for this training window, including a new Box-Cox λ.
        self.target_transformer = TargetTransformer(method=self.target_transformer.method)
        series_t, future_t, shifted_t = transform_series(
            self.target_transformer, series, future_covariates, shifted_covariates
        )
        self.catboost.fit(
            series_t,
            future_covariates=stack_covariates(future_t, shifted_t),
        )
        self._is_fitted = True
        return self

    def predict(self, n: int,
                future_covariates: TimeSeries | None = None,
                shifted_covariates: TimeSeries | None = None,
                num_samples: int = 100) -> TimeSeries:
        if not self._is_fitted:
            raise ValueError("Model is not fitted yet.")
        pred = self.catboost.predict(
            n,
            future_covariates=stack_covariates(future_covariates, shifted_covariates),
            num_samples=num_samples,
        )
        return inverse_target(self.target_transformer, pred, self._anchor_level)

    def inverse_transform(self, series: TimeSeries) -> TimeSeries:
        return inverse_target(self.target_transformer, series, self._anchor_level)

def history_until(
    df: pd.DataFrame,
    cutoff: pd.Timestamp,
    date_col: str = "date",
) -> pd.DataFrame:
    """Rows on or before ``cutoff``. Later sales stay out of ABC routing."""
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' is not in the frame.")
    dates = pd.to_datetime(df[date_col])
    limit = pd.Timestamp(cutoff)
    kept = df.loc[dates <= limit]
    if kept.empty:
        raise ValueError(f"No rows on or before {limit}.")
    return kept.copy()


def training_history(
    df: pd.DataFrame,
    horizon: int,
    date_col: str = "date",
) -> pd.DataFrame:
    """Rows on or before the forecast origin.

    The last ``horizon`` days are the period the model is about to score.
    ABC shares computed on them leak that period into model routing.
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}.")
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' is not in the frame.")
    dates = pd.to_datetime(df[date_col])
    cutoff = dates.max() - pd.Timedelta(days=horizon)
    return history_until(df, cutoff, date_col)


class ABCSegmenter:
    """Segments time series into A, B, C classes based on Pareto principle (volume)."""
    
    def __init__(self, df_sales: pd.DataFrame, group_cols: list[str] = ['store_nbr', 'family'], target_col: str = 'sales'):
        self.group_cols = group_cols
        self.target_col = target_col
        self.segmentation = self._compute_segments(df_sales)
        
    def _compute_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        # Calculate total sales per group
        total_sales = df.groupby(self.group_cols)[self.target_col].sum().reset_index()
        total_sales = total_sales.sort_values(by=self.target_col, ascending=False)
        
        # Calculate cumulative percentage
        total_sum = total_sales[self.target_col].sum()
        total_sales['cum_perc'] = total_sales[self.target_col].cumsum() / total_sum
        
        # Assign classes
        def get_class(perc):
            if perc <= 0.80:
                return 'A'
            elif perc <= 0.95:
                return 'B'
            else:
                return 'C'
                
        total_sales['abc_class'] = total_sales['cum_perc'].apply(get_class)
        logger.info(f"ABC Segmentation computed: A={len(total_sales[total_sales['abc_class']=='A'])}, "
                    f"B={len(total_sales[total_sales['abc_class']=='B'])}, "
                    f"C={len(total_sales[total_sales['abc_class']=='C'])}")
        return total_sales
        
    def get_class(self, store_nbr: int, family: str) -> str:
        row = self.segmentation[(self.segmentation['store_nbr'] == store_nbr) & (self.segmentation['family'] == family)]
        if len(row) == 0:
            return 'C' # default to C if unknown
        return row['abc_class'].values[0]


class ForecastRouter:
    """Routes the forecasting task to the appropriate model based on ABC class."""
    
    def __init__(self, store_nbr: int, family: str, segmenter: ABCSegmenter):
        self.abc_class = segmenter.get_class(store_nbr, family)
        logger.info(f"Routing Store {store_nbr}, Family {family} -> Class {self.abc_class}")
        
    def get_model(self, forecast_horizon: int = 28):
        safe_lags = horizon_safe_lags(forecast_horizon)
        if self.abc_class in ['A', 'B']:
            logger.info("Using HybridProphetCatBoost for Class A/B")
            prophet = ProphetForecaster()
            catboost = CatBoostResidualModel(
                lags=safe_lags,
                lags_future_covariates=[0],
            )
            return HybridProphetCatBoost(prophet_model=prophet, catboost_model=catboost)
        logger.info("Using DirectCatBoostForecaster for Class C (skipping Prophet)")
        # Lag 0 reads a covariate on the forecast date. Pre-shifted columns
        # are valid there; a raw series would not be.
        catboost = CatBoostResidualModel(
            lags=safe_lags,
            lags_future_covariates=[0],
        )
        return DirectCatBoostForecaster(catboost)
