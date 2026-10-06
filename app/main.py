import streamlit as st
import pandas as pd
from pathlib import Path
import numpy as np

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.features.pipeline import FeatureEngineeringPipeline
from src.models.hybrid_pipeline import HybridProphetCatBoost
from src.visualization.decomposition_plots import plot_hybrid_decomposition
from src.visualization.evaluation_plots import plot_forecast_vs_actuals
from src.evaluation.metrics import BusinessMetrics
from src.models.routing import ABCSegmenter, ForecastRouter
from darts import TimeSeries

st.set_page_config(page_title="Demand Forecasting", layout="wide")

st.title("📈 Demand Forecasting with Prophet + CatBoost")
st.markdown("Interactive dashboard for exploring forecasts and uncertainty.")

@st.cache_data
def load_all_data():
    data_dir = Path("data/raw")
    loader = DataLoader(data_dir=data_dir)
    return loader.load_all()

try:
    datasets = load_all_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

st.sidebar.header("Configuration")
store_nbr = st.sidebar.selectbox("Store Number", [1, 2, 3, 4, 5])
family = st.sidebar.selectbox("Product Family", ["PRODUCE", "GROCERY I", "BEVERAGES"])
horizon = st.sidebar.slider("Forecast Horizon", 7, 90, 28)

# Run Segmentation
if "segmenter" not in st.session_state:
    with st.spinner("Computing ABC Segmentation..."):
        st.session_state.segmenter = ABCSegmenter(datasets['train'])
        
abc_class = st.session_state.segmenter.get_class(store_nbr, family)
st.sidebar.markdown(f"**ABC Class:** {abc_class}")

if st.sidebar.button("Run Forecast Pipeline"):
    with st.spinner("Running preprocessing and feature engineering..."):
        train_df = datasets['train']
        train_subset = train_df[(train_df['store_nbr'] == store_nbr) & (train_df['family'] == family)].copy()
        
        run_datasets = {k: v.copy() for k, v in datasets.items()}
        run_datasets['train'] = train_subset
        
        preprocessor = DataPreprocessor(aggregation_level='store_family')
        df_clean = preprocessor.preprocess(run_datasets)
        
        feature_pipeline = FeatureEngineeringPipeline(forecast_horizon=horizon)
        df_features = feature_pipeline.fit_transform(
            df_clean, target_col='sales', date_col='date'
        )
        
        prophet_covs = [c for c in feature_pipeline.future_covariate_names() if c in df_features.columns]
        catboost_covs = [c for c in feature_pipeline.past_covariate_names() if c in df_features.columns]

        df_features = df_features.dropna(subset=catboost_covs).reset_index(drop=True)
        df_features = df_features.select_dtypes(exclude=['category', 'object'])
        prophet_covs = [c for c in prophet_covs if c in df_features.columns]
        catboost_covs = [c for c in catboost_covs if c in df_features.columns]
        df_features[prophet_covs] = df_features[prophet_covs].fillna(0)
        
        split_date = df_features['date'].max() - pd.Timedelta(days=horizon)
        train_df = df_features[df_features['date'] <= split_date]
        val_df = df_features[df_features['date'] > split_date]
        
        ts_train = TimeSeries.from_dataframe(train_df, time_col='date', value_cols='sales', freq='D')
        ts_val = TimeSeries.from_dataframe(val_df, time_col='date', value_cols='sales', freq='D')
        
        cov_future = TimeSeries.from_dataframe(df_features, time_col='date', value_cols=prophet_covs, freq='D') if prophet_covs else None
        cov_past = TimeSeries.from_dataframe(df_features, time_col='date', value_cols=catboost_covs, freq='D') if catboost_covs else None
        
    with st.spinner(f"Training Model for Class {abc_class}..."):
        router = ForecastRouter(store_nbr, family, st.session_state.segmenter)
        model = router.get_model(forecast_horizon=horizon)
        # The model applies base_config target_transform and inverts it in predict.
        model.fit(ts_train, future_covariates=cov_future, past_covariates=cov_past)

    with st.spinner("Generating Forecast..."):
        preds = model.predict(
            n=horizon,
            future_covariates=cov_future,
            past_covariates=cov_past,
            num_samples=100,
        )
        preds = preds.map(lambda x: np.clip(x, 0, None))
            
        st.success(f"Forecast completed for Store {store_nbr}, {family}!")
        
        # Asymmetric Optimization based on Product Family
        if "PRODUCE" in family:
            target_quantile = 0.40
            quantile_reason = "Asymmetric safety stock for Produce (P40) to minimize waste (overstock)."
        elif "GROCERY" in family:
            target_quantile = 0.75
            quantile_reason = "Asymmetric safety stock for Grocery (P75) to maximize shelf availability (out-of-stock prevention)."
        else:
            target_quantile = 0.50
            quantile_reason = "Standard median forecast (P50) for balanced risk."
            
        st.info(f"🎯 **Optimization Strategy:** Using **P{int(target_quantile*100)}** quantile. *Reason: {quantile_reason}*")
        
        st.subheader("Forecast vs Actuals")
        # History is the last 90 days of training
        fig_eval = plot_forecast_vs_actuals(actuals=ts_val, forecast=preds, history=ts_train[-90:])
        st.plotly_chart(fig_eval, use_container_width=True)
            
        st.subheader("Business Metrics")
        y_true_ts = ts_val
        y_pred_ts = preds.quantile(target_quantile) if preds.is_probabilistic else preds
        
        # Calculate metrics using BusinessMetrics
        metrics = BusinessMetrics.standard_metrics(y_true_ts, y_pred_ts)
        metrics["Direction Accuracy"] = BusinessMetrics.direction_accuracy(y_true_ts, y_pred_ts)
        metrics["Peak Capture Rate"] = BusinessMetrics.peak_capture_rate(y_true_ts, y_pred_ts, threshold_percentile=90)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAPE", f"{metrics.get('MAPE', 0):.2f}%")
        m2.metric("RMSE", f"{metrics.get('RMSE', 0):.2f}")
        m3.metric("Direction Acc", f"{metrics.get('Direction Accuracy', 0):.2%}")
        m4.metric("Peak Capture", f"{metrics.get('Peak Capture Rate', 0):.2%}")
        
    with st.spinner("Decomposing..."):
        if isinstance(model, HybridProphetCatBoost):
            st.subheader("Model Decomposition (Validation Set)")
            
            # Predict with Darts wrapper to get the TimeSeries (in log scale)
            val_prophet_preds_log = model.prophet.model.predict(n=horizon, future_covariates=cov_future)
            
            # To get components, we must use the underlying Prophet model directly
            future_dates = pd.DataFrame({'ds': val_prophet_preds_log.time_index})
            if cov_future is not None:
                cov_df = cov_future.to_dataframe().reset_index()
                time_col = cov_df.columns[0]
                cov_df = cov_df.rename(columns={time_col: 'ds'})
                if isinstance(cov_df.columns, pd.MultiIndex):
                    cov_df.columns = ['ds' if c[0] == 'ds' else c[0] for c in cov_df.columns]
                future_dates = future_dates.merge(cov_df, on='ds', how='left')
                
            # Get detailed Prophet components (still in transformed space)
            components = model.prophet.model.model.predict(future_dates)
            trend_log = TimeSeries.from_dataframe(components, time_col='ds', value_cols='trend')

            val_prophet_preds = model.inverse_transform(val_prophet_preds_log)
            val_prophet_trend = model.inverse_transform(trend_log)
            
            # Calculate Seasonality in original scale as a multiplier (centered around 1.0)
            val_prophet_seasonality = val_prophet_preds / val_prophet_trend
            
            # Fallback if slice_intersect fails due to freq mismatch
            try:
                val_prophet_resid = ts_val - val_prophet_preds
            except Exception:
                val_prophet_resid = ts_val
                
            fig_decomp = plot_hybrid_decomposition(
                actuals=ts_val, 
                prophet_trend=val_prophet_trend, 
                prophet_seasonality=val_prophet_seasonality, 
                residuals=val_prophet_resid
            )
            st.plotly_chart(fig_decomp, use_container_width=True)
        else:
            st.info("Decomposition plot is only available for the Hybrid Prophet model (Class A/B).")
else:
    st.info("Configure parameters in the sidebar and click 'Run Forecast Pipeline'.")
