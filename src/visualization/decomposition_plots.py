import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from darts import TimeSeries

def plot_hybrid_decomposition(actuals: TimeSeries, prophet_trend: TimeSeries, 
                            prophet_seasonality: TimeSeries, residuals: TimeSeries,
                            title: str = "Hybrid Model Decomposition") -> go.Figure:
    """
    Plots the components of the hybrid Prophet-CatBoost model.
    """
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("Actuals vs Prophet Trend", "Prophet Seasonality", 
                       "Residuals (CatBoost Target)", "Residual Distribution")
    )
    
    df_act = actuals.to_dataframe()
    df_trend = prophet_trend.to_dataframe()
    df_season = prophet_seasonality.to_dataframe()
    df_resid = residuals.to_dataframe()
    
    # 1. Actuals vs Trend
    fig.add_trace(
        go.Scatter(x=df_act.index, y=df_act.iloc[:, 0], name="Actuals", 
                  line=dict(color='blue', width=1), opacity=0.6),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_trend.index, y=df_trend.iloc[:, 0], name="Trend", 
                  line=dict(color='red', width=2)),
        row=1, col=1
    )
    
    # 2. Seasonality
    fig.add_trace(
        go.Scatter(x=df_season.index, y=df_season.iloc[:, 0], name="Seasonality",
                  line=dict(color='green', width=1)),
        row=2, col=1
    )
    
    # 3. Residuals (Time Series)
    fig.add_trace(
        go.Scatter(x=df_resid.index, y=df_resid.iloc[:, 0], name="Residuals",
                  mode='markers', marker=dict(color='orange', size=3, opacity=0.5)),
        row=3, col=1
    )
    
    # 4. Residuals (Histogram)
    fig.add_trace(
        go.Histogram(x=df_resid.iloc[:, 0], name="Residual Dist",
                    marker_color='orange', opacity=0.7, nbinsx=50),
        row=4, col=1
    )
    
    fig.update_layout(height=1000, title_text=title, showlegend=True)
    return fig
