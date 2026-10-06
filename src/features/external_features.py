from __future__ import annotations

import pandas as pd
from loguru import logger

class ExternalFeatureGenerator:
    """Generate features from external signals: oil prices, promotions, transactions."""
    
    def __init__(self, oil_lags: list[int] | None = None,
                 include_promotion: bool = True,
                 include_transactions: bool = True,
                 forecast_horizon: int = 28) -> None:
        self.forecast_horizon = forecast_horizon
        requested = list(oil_lags) if oil_lags is not None else [forecast_horizon]
        valid = [lag for lag in requested if lag >= forecast_horizon]
        if not valid:
            logger.warning(
                f"Oil lags {requested} are not known {forecast_horizon} days ahead. "
                f"Using oil_lag_{forecast_horizon}."
            )
            valid = [forecast_horizon]
        elif len(valid) < len(requested):
            skipped = [lag for lag in requested if lag < forecast_horizon]
            logger.warning(
                f"Skipping oil lags {skipped}: oil is not known over a horizon of {forecast_horizon}."
            )
        self.oil_lags = valid
        # Store traffic on the forecast dates does not exist yet. Lag by the
        # full horizon so every forecast step still reads traffic from the origin.
        self.transactions_lag = forecast_horizon
        self.include_promotion = include_promotion
        self.include_transactions = include_transactions
        self._feature_names: list[str] = []

    def transform(self, df: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
        df_out = df.copy()
        self._feature_names = []
        
        # Oil Lags
        if 'dcoilwtico' in df_out.columns:
            for lag in self.oil_lags:
                feat_name = f"oil_lag_{lag}"
                self._feature_names.append(feat_name)
                # Since oil is global (same for all groups), we just shift the column if it's already sorted by date.
                # If grouped, we shift per group.
                if group_cols:
                    df_out[feat_name] = df_out.groupby(group_cols)['dcoilwtico'].shift(lag)
                else:
                    df_out[feat_name] = df_out['dcoilwtico'].shift(lag)
        else:
            logger.warning("dcoilwtico column not found. Skipping oil lags.")
            
        # Promotions
        if self.include_promotion and 'onpromotion' in df_out.columns:
            # Typically known ahead or is a past covariate depending on context.
            # Usually it's known ahead. We just keep it as a feature.
            self._feature_names.append('onpromotion')
            
            # Promo Lead/Lag (Expectation and residual effect)
            if group_cols:
                df_out['onpromotion_lead_1'] = df_out.groupby(group_cols)['onpromotion'].shift(-1)
                df_out['onpromotion_lag_1'] = df_out.groupby(group_cols)['onpromotion'].shift(1)
            else:
                df_out['onpromotion_lead_1'] = df_out['onpromotion'].shift(-1)
                df_out['onpromotion_lag_1'] = df_out['onpromotion'].shift(1)
            
            # Fill NaNs from shift with 0
            df_out['onpromotion_lead_1'] = df_out['onpromotion_lead_1'].fillna(0).astype(df_out['onpromotion'].dtype)
            df_out['onpromotion_lag_1'] = df_out['onpromotion_lag_1'].fillna(0).astype(df_out['onpromotion'].dtype)
            self._feature_names.extend(['onpromotion_lead_1', 'onpromotion_lag_1'])
            
            # Rolling sum of promotions (density over last week)
            if group_cols:
                df_out['onpromotion_rolling_sum_7'] = df_out.groupby(group_cols)['onpromotion'].transform(lambda x: x.rolling(7, min_periods=1).sum())
            else:
                df_out['onpromotion_rolling_sum_7'] = df_out['onpromotion'].rolling(7, min_periods=1).sum()
            self._feature_names.append('onpromotion_rolling_sum_7')

            # Rolling max and EMA for peak capture (extended to 3, 7, 14 days)
            for window in [3, 7, 14]:
                feat_name_max = f"onpromotion_roll_max_{window}"
                feat_name_ema = f"onpromotion_ema_{window}"
                self._feature_names.extend([feat_name_max, feat_name_ema])
                if group_cols:
                    df_out[feat_name_max] = df_out.groupby(group_cols)['onpromotion'].transform(lambda x: x.rolling(window, min_periods=1).max())
                    df_out[feat_name_ema] = df_out.groupby(group_cols)['onpromotion'].transform(lambda x: x.ewm(span=window, min_periods=1).mean())
                else:
                    df_out[feat_name_max] = df_out['onpromotion'].rolling(window, min_periods=1).max()
                    df_out[feat_name_ema] = df_out['onpromotion'].ewm(span=window, min_periods=1).mean()
            
        # Transactions are observed, not planned. Lag by the forecast horizon.
        if self.include_transactions and 'transactions' in df_out.columns:
            feat_name = f"transactions_lag_{self.transactions_lag}"
            self._feature_names.append(feat_name)
            if group_cols:
                df_out[feat_name] = df_out.groupby(group_cols)['transactions'].shift(self.transactions_lag)
            else:
                df_out[feat_name] = df_out['transactions'].shift(self.transactions_lag)
                
        logger.info(f"Generated {len(self._feature_names)} external features.")
        return df_out

    def get_feature_names(self) -> list[str]:
        return self._feature_names
