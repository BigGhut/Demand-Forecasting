import pandas as pd
import numpy as np
from loguru import logger
import optuna
from pathlib import Path
from darts import TimeSeries

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.splitter import TimeSeriesSplitter
from src.features.pipeline import FeatureEngineeringPipeline
from src.models.hybrid_pipeline import HybridProphetCatBoost
from src.models.prophet_model import ProphetForecaster
from src.models.catboost_model import CatBoostResidualModel
from src.evaluation.metrics import BusinessMetrics
from src.utils.config import get_base_config, CatBoostConfig, ProphetConfig

logger.add("logs/tuning.log")

def get_data_and_cv(store_nbr: int, family: str):
    data_dir = Path("data/raw")
    loader = DataLoader(data_dir)
    datasets = loader.load_all()
    
    # Preprocess
    preprocessor = DataPreprocessor(aggregation_level='store_family')
    df = preprocessor.preprocess(datasets)
    
    # Filter
    df = df[(df['store_nbr'] == store_nbr) & (df['family'] == family)].copy()
    if len(df) == 0:
        return None, None, None
        
    cv = BusinessMetrics.coefficient_of_variation(TimeSeries.from_dataframe(df, time_col='date', value_cols='sales'))
    
    # Split
    splitter = TimeSeriesSplitter(forecast_horizon=28, validation_size=28)
    train_df, val_df, test_df = splitter.split(df)
    
    return train_df, val_df, cv

def tune_hybrid(trial, train_df, val_df):
    # Hyperparameters
    prophet_cps = trial.suggest_float("changepoint_prior_scale", 0.001, 0.5, log=True)
    cat_lr = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    cat_depth = trial.suggest_int("depth", 4, 10)
    
    # Both options stay at or beyond the horizon, so the 28-step forecast
    # does not recurse on predicted residuals.
    lag_type = trial.suggest_categorical("lag_type", ["sparse_horizon", "wide_horizon"])
    if lag_type == "sparse_horizon":
        lags = [-28, -56]
    else:
        lags = [-28, -35, -42, -56]

    horizon = 28
    pipeline = FeatureEngineeringPipeline(forecast_horizon=horizon)
    df_features = pipeline.fit_transform(pd.concat([train_df, val_df]))
    df_features = pipeline.drop_na_rows(df_features)

    train_feat_df = df_features[df_features['date'] <= train_df['date'].max()]
    val_feat_df = df_features[df_features['date'] > train_df['date'].max()]

    ts_train = TimeSeries.from_dataframe(train_feat_df, time_col='date', value_cols='sales', freq='D')
    ts_val = TimeSeries.from_dataframe(val_feat_df, time_col='date', value_cols='sales', freq='D')

    future_cols = [c for c in pipeline.future_covariate_names() if c in df_features.columns]
    shifted_cols = [c for c in pipeline.shifted_covariate_names() if c in df_features.columns]
    cov_future = (
        TimeSeries.from_dataframe(df_features, time_col='date', value_cols=future_cols, freq='D')
        if future_cols else None
    )
    cov_shifted = (
        TimeSeries.from_dataframe(df_features, time_col='date', value_cols=shifted_cols, freq='D')
        if shifted_cols else None
    )

    p_config = ProphetConfig(changepoint_prior_scale=prophet_cps, seasonality_mode='multiplicative', yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    c_config = CatBoostConfig(learning_rate=cat_lr, depth=cat_depth, iterations=100, task_type='CPU', early_stopping_rounds=10)

    prophet = ProphetForecaster(config=p_config)
    catboost = CatBoostResidualModel(
        config=c_config, lags=lags, lags_future_covariates=[0]
    )
    hybrid = HybridProphetCatBoost(prophet_model=prophet, catboost_model=catboost)

    hybrid.fit(ts_train, future_covariates=cov_future, shifted_covariates=cov_shifted)
    # One Gaussian sample is not the median. Average the noise out with the
    # sample median so WAPE can be compared across trials.
    preds = hybrid.predict(
        n=horizon,
        future_covariates=cov_future,
        shifted_covariates=cov_shifted,
        num_samples=100,
    )
    point = preds.quantile(0.5) if preds.is_probabilistic else preds
    metrics = BusinessMetrics.standard_metrics(ts_val, point)

    # If Tracking Signal is breached, heavily penalize.
    # WAPE, not MAPE: zero-filled days make MAPE undefined or explosive.
    tracking = metrics.get('Tracking Signal', 0)
    penalty = 0
    if tracking < -4 or tracking > 4:
        penalty = 1000

    return metrics['WAPE'] + penalty, metrics['RMSE'], metrics['WAPE'], metrics['R2']

if __name__ == "__main__":
    train_df, val_df, cv = get_data_and_cv(1, 'PRODUCE')
    logger.info(f"CV for Store 1 PRODUCE: {cv:.4f}")
    
    def objective(trial):
        wape, rmse, _, r2 = tune_hybrid(trial, train_df, val_df)
        return wape
        
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5)
    
    logger.info(f"Best trial WAPE: {study.best_trial.value}")
    logger.info(f"Best params: {study.best_trial.params}")
