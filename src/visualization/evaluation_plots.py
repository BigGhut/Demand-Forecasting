import pandas as pd
import numpy as np
import plotly.graph_objects as go
from darts import TimeSeries

def plot_forecast_vs_actuals(actuals: TimeSeries, forecast: TimeSeries, 
                             history: TimeSeries | None = None,
                             title: str = "Forecast vs Actuals") -> go.Figure:
    """
    Plots the forecast against actuals, including uncertainty intervals if the 
    forecast is probabilistic.
    """
    fig = go.Figure()
    
    # 1. Plot History (if provided)
    if history is not None:
        df_hist = history.to_dataframe()
        fig.add_trace(
            go.Scatter(x=df_hist.index, y=df_hist.iloc[:, 0], name="History",
                      line=dict(color='gray', width=1))
        )
        
    # 2. Plot Actuals
    df_act = actuals.to_dataframe()
    fig.add_trace(
        go.Scatter(x=df_act.index, y=df_act.iloc[:, 0], name="Actuals",
                  line=dict(color='black', width=2))
    )
    
    # 3. Plot Forecast (handle probabilistic vs deterministic)
    if forecast.is_probabilistic:
        # Get quantiles
        df_low = forecast.quantile(0.05).to_dataframe()
        df_high = forecast.quantile(0.95).to_dataframe()
        df_median = forecast.quantile(0.5).to_dataframe()
        
        # Plot Confidence Interval
        fig.add_trace(
            go.Scatter(x=df_high.index.tolist() + df_low.index.tolist()[::-1],
                       y=df_high.iloc[:, 0].tolist() + df_low.iloc[:, 0].tolist()[::-1],
                       fill='toself', fillcolor='rgba(0, 100, 255, 0.2)',
                       line=dict(color='rgba(255,255,255,0)'),
                       hoverinfo="skip", showlegend=True, name="90% Prediction Interval")
        )
        
        # Plot Median
        fig.add_trace(
            go.Scatter(x=df_median.index, y=df_median.iloc[:, 0], name="Forecast (Median)",
                      line=dict(color='blue', width=2, dash='dash'))
        )
    else:
        df_pred = forecast.to_dataframe()
        fig.add_trace(
            go.Scatter(x=df_pred.index, y=df_pred.iloc[:, 0], name="Forecast",
                      line=dict(color='blue', width=2, dash='dash'))
        )
        
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Sales",
                      hovermode="x unified", template="plotly_white")
    return fig
