import pandas as pd
from pathlib import Path
from loguru import logger

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.features.pipeline import FeatureEngineeringPipeline
from src.models.prophet_model import ProphetForecaster
from src.models.catboost_model import CatBoostResidualModel
from src.models.hybrid_pipeline import HybridProphetCatBoost

from darts import TimeSeries

def main():
    logger.info("Starting Demand Forecasting Hybrid Pipeline")
    
    # 1. Load Data
    data_dir = Path("data/raw")
    loader = DataLoader(data_dir=data_dir)
    datasets = loader.load_all()
    
    # Subset data for rapid testing (Store 1, PRODUCE)
    logger.info("Subsetting data for rapid testing (Store 1, PRODUCE)")
    train_df = datasets['train']
    train_subset = train_df[(train_df['store_nbr'] == 1) & (train_df['family'] == 'PRODUCE')].copy()
    datasets['train'] = train_subset
    
    # 2. Preprocess Data
    preprocessor = DataPreprocessor(aggregation_level='store_family')
    df_clean = preprocessor.preprocess(datasets)
    
    # 3. Feature Engineering
    feature_pipeline = FeatureEngineeringPipeline(forecast_horizon=28) # horizon 28 for testing
    df_features = feature_pipeline.fit_transform(
        df_clean, target_col='sales', date_col='date'
    )
    
    # Get grouped feature names
    feat_dict = feature_pipeline.get_all_feature_names()
    
    # Prophet future covariates = calendar + external
    prophet_covs = feat_dict['calendar'] + feat_dict['external']
    
    # Catboost past covariates = lags + rolling + expanding
    catboost_covs = feat_dict['lags'] + feat_dict['rolling'] + feat_dict['expanding']
    
    # Drop rows where past covariates (lags) are NaN to avoid issues
    df_features = df_features.dropna(subset=catboost_covs).reset_index(drop=True)
    
    # Convert Dtypes to avoid Dart issues
    df_features = df_features.select_dtypes(exclude=['category', 'object'])
    
    # Fill any remaining NaNs in future covariates (like onpromotion)
    df_features[prophet_covs] = df_features[prophet_covs].fillna(0)
    
    # 4. Prepare for Darts (TimeSeries)
    # Since we dropped rows at the beginning, the series is contiguous from that new start date.
    split_date = df_features['date'].max() - pd.Timedelta(days=28)
    train_df = df_features[df_features['date'] <= split_date]
    val_df = df_features[df_features['date'] > split_date]
    
    logger.info(f"Train set: {train_df.shape}, Val set: {val_df.shape}")
    
    # Create TimeSeries objects without fill_missing_dates to strictly enforce our contiguous index
    ts_train = TimeSeries.from_dataframe(train_df, time_col='date', value_cols='sales', freq='D')
    ts_val = TimeSeries.from_dataframe(val_df, time_col='date', value_cols='sales', freq='D')
    
    if len(prophet_covs) > 0:
        cov_future = TimeSeries.from_dataframe(df_features, time_col='date', value_cols=prophet_covs, freq='D')
    else:
        cov_future = None
        
    if len(catboost_covs) > 0:
        cov_past = TimeSeries.from_dataframe(df_features, time_col='date', value_cols=catboost_covs, freq='D')
    else:
        cov_past = None
        
    # 5. Model Training
    prophet = ProphetForecaster()
    catboost = CatBoostResidualModel(lags=7, lags_past_covariates=7)
    
    hybrid = HybridProphetCatBoost(prophet_model=prophet, catboost_model=catboost)
    
    hybrid.fit(ts_train, future_covariates=cov_future, past_covariates=cov_past)
    
    # 6. Predict
    preds = hybrid.predict(n=28, future_covariates=cov_future, past_covariates=cov_past, num_samples=100)
                           
    logger.info(f"Predictions:\n{preds.quantile(0.5).to_dataframe().head()}")
    
    # 7. Evaluate Metrics
    from src.evaluation.metrics import BusinessMetrics
    y_true_ts = ts_val
    y_pred_ts = preds.quantile(0.5) if preds.is_probabilistic else preds
    
    metrics = BusinessMetrics.standard_metrics(y_true_ts, y_pred_ts)
    metrics["Direction Accuracy"] = BusinessMetrics.direction_accuracy(y_true_ts, y_pred_ts)
    metrics["Peak Capture Rate"] = BusinessMetrics.peak_capture_rate(y_true_ts, y_pred_ts, threshold_percentile=90)
    
    logger.info("====== BUSINESS METRICS ======")
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")
    logger.info("==============================")
    
    logger.success("End-to-End pipeline ran successfully!")

if __name__ == "__main__":
    main()
