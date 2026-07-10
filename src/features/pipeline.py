from __future__ import annotations

import pandas as pd
from loguru import logger

from src.utils.config import FeatureConfig, get_feature_config
from src.features.calendar_features import CalendarFeatureGenerator
from src.features.external_features import ExternalFeatureGenerator
from src.features.lag_features import LagFeatureGenerator
from src.features.rolling_features import RollingFeatureGenerator
from src.features.expanding_features import ExpandingFeatureGenerator

class FeatureEngineeringPipeline:
    """Orchestrates all feature generators in the correct order.
    
    Order is critical:
    1. Calendar features (no dependencies)
    2. External features (no dependencies on other features)
    3. Lag features (depends on target column)
    4. Rolling features (depends on lag columns)
    5. Expanding features (depends on target column, uses lag approach)
    """
    
    def __init__(self, feature_config: FeatureConfig | None = None,
                 forecast_horizon: int = 28) -> None:
        self.config = feature_config or get_feature_config()
        self.forecast_horizon = forecast_horizon
        
        self.generators = {
            "calendar": CalendarFeatureGenerator(
                features=self.config.calendar.features,
                binary_features=self.config.calendar.binary
            ),
            "external": ExternalFeatureGenerator(
                oil_lags=self.config.external.oil_lags,
                include_promotion=self.config.external.include_promotion,
                include_transactions=self.config.external.include_transactions
            ),
            "lags": LagFeatureGenerator(
                lags=self.config.lags,
                forecast_horizon=self.forecast_horizon
            ),
            "rolling": RollingFeatureGenerator(
                windows=self.config.rolling.windows,
                functions=self.config.rolling.functions,
                base_lags=self.config.rolling.base_lags
            ),
            "expanding": ExpandingFeatureGenerator(
                functions=self.config.expanding.functions,
                min_periods=self.config.expanding.min_periods
            )
        }
        
    def fit_transform(self, df: pd.DataFrame, 
                       target_col: str = 'sales',
                       group_cols: list[str] | None = None,
                       date_col: str = 'date') -> pd.DataFrame:
        """Apply all feature generators and return enriched DataFrame."""
        return self.transform(df, target_col, group_cols, date_col, validate=True)
    
    def transform(self, df: pd.DataFrame, 
                       target_col: str = 'sales',
                       group_cols: list[str] | None = None,
                       date_col: str = 'date',
                       validate: bool = False) -> pd.DataFrame:
        """Transform data through the feature pipeline."""
        df_out = df.copy()
        
        # 1. Calendar
        df_out = self.generators["calendar"].transform(df_out, date_col=date_col)
        
        # 2. External
        df_out = self.generators["external"].transform(df_out, group_cols=group_cols)
        
        # 3. Lags
        df_out = self.generators["lags"].transform(df_out, target_col=target_col, group_cols=group_cols)
        
        # 4. Rolling
        df_out = self.generators["rolling"].transform(df_out, group_cols=group_cols)
        
        # 5. Expanding
        df_out = self.generators["expanding"].transform(df_out, target_col=target_col, group_cols=group_cols)
        
        if validate:
            self.validate_no_leakage(df_out, target_col, date_col)
            
        total_feats = sum(len(g.get_feature_names()) for g in self.generators.values())
        logger.info(f"Pipeline completed: {total_feats} total features generated.")
        
        return df_out
    
    def get_all_feature_names(self) -> dict[str, list[str]]:
        """Return feature names grouped by generator type."""
        return {
            name: gen.get_feature_names()
            for name, gen in self.generators.items()
        }
    
    def validate_no_leakage(self, df: pd.DataFrame, target_col: str,
                             date_col: str) -> bool:
        """Check that no feature at time t contains information from t+1..t+h."""
        # Simple heuristic check: correlation between feature and future target
        # For a robust check in real-world, we'd ensure min lag >= horizon.
        # Here we just verify that all lag-based features have "lag_N" where N >= horizon.
        lags_gen = self.generators["lags"]
        if hasattr(lags_gen, 'lags'):
            min_lag = min(lags_gen.lags)
            if min_lag < self.forecast_horizon:
                logger.error(f"Leakage detected! Min lag ({min_lag}) < forecast_horizon ({self.forecast_horizon}).")
                return False
        logger.info("Leakage validation passed: min lag >= forecast_horizon.")
        return True
    
    def drop_na_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows with NaN values created by lagging/rolling."""
        initial_len = len(df)
        
        feature_cols = [f for feats in self.get_all_feature_names().values() for f in feats]
        check_cols = [c for c in feature_cols if c in df.columns]
        
        df_out = df.dropna(subset=check_cols)
        dropped = initial_len - len(df_out)
        logger.info(f"Dropped {dropped} rows due to NaN values in generated features.")
        return df_out
