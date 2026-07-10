from __future__ import annotations

import pandas as pd
from loguru import logger
from darts import TimeSeries

from src.models.hybrid_pipeline import HybridProphetCatBoost
from src.models.catboost_model import CatBoostResidualModel
from src.models.prophet_model import ProphetForecaster
from src.utils.config import get_base_config

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
        
    def get_model(self):
        if self.abc_class in ['A', 'B']:
            logger.info("Using HybridProphetCatBoost for Class A/B")
            prophet = ProphetForecaster()
            catboost = CatBoostResidualModel(lags_past_covariates=28)
            return HybridProphetCatBoost(prophet_model=prophet, catboost_model=catboost)
        else:
            logger.info("Using CatBoostResidualModel directly for Class C (skipping Prophet)")
            # When bypassing Prophet, the residual model just models the raw target
            return CatBoostResidualModel()
