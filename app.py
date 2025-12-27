#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Electricity Price Forecasting Dashboard

Description:
    This Dash application visualizes and compares two Machine Learning models 
    (Random Forest and Linear Regression) for predicting electricity prices.
    
    It displays:
    1. Key Performance Metrics (R2, RMSE, MAE).
    2. A Time-Series plot of Actual vs. Predicted prices.
    3. A Scatter plot for error analysis.
    4. A Feature Importance chart (excluding lag-variables).

"""
import numpy as np
import pandas as pd

from dash import html, dcc, Input, Output, Dash
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import webbrowser
from threading import Timer
import plotly.graph_objects as go

from model_analysis import rf_reg_model, linear_reg_model, X, y

app = Dash(__name__)
host = 'localhost'
port = 8050


rf_reg, X_test_rf, y_test_rf, y_pred_rf, df_rf, X_train_rf, y_train_rf = rf_reg_model(X, y)
lin_reg, X_test_lin, y_test_lin, y_pred_lin, df_lin, X_train_lin, y_train_lin = linear_reg_model(X, y)

#Results from the models
results_rf_df = pd.DataFrame({
    'Date': df_rf.loc[y_test_rf.index, 'Date'],
    'Actual': y_test_rf,
    'Predicted': y_pred_rf
}).sort_values('Date')

results_lin_df = pd.DataFrame({
    'Date': df_lin.loc[y_test_lin.index, 'Date'],
    'Actual': y_test_lin,
    'Predicted': y_pred_lin
}).sort_values('Date')

#Residuals
results_rf_df["Residual"] = results_rf_df["Actual"] - results_rf_df["Predicted"]
results_lin_df["Residual"] = results_lin_df["Actual"] - results_lin_df["Predicted"]



rf_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_reg.feature_importances_
})

rf_importance_df = rf_importance_df[rf_importance_df['Feature'] != 'Price_lag_1']
rf_importance_df = rf_importance_df[rf_importance_df['Feature'] != 'Price_lag_24']
rf_importance_df = rf_importance_df[rf_importance_df['Feature'] != 'Price_lag_168']

rf_importance_df = rf_importance_df.sort_values(by='Importance', ascending=True)

lin_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': abs(lin_reg.coef_) 
})

lin_importance_df = lin_importance_df[lin_importance_df['Feature'] != 'Price_lag_1']
lin_importance_df = lin_importance_df[lin_importance_df['Feature'] != 'Price_lag_24']
lin_importance_df = lin_importance_df[lin_importance_df['Feature'] != 'Price_lag_168']

lin_importance_df = lin_importance_df.sort_values(by='Importance', ascending=True)

colors = {
    'background': '#ffffff', 
    'text': '#333333',       
}

app.layout = html.Div(children= [
    
    html.H1(
    children = 'Electricity Price Forecasting', 
    style= {'textAlign': 'center','background' : colors['background'], 'color' : colors['text']}),  
    
    html.Div([ 
        html.Label('Choose a Model:', 
                   style={
                       'fontWeight': 'bold',       
                       'fontSize': '20px',        
                       'color': colors['text'],   
                       }),
        dcc.Dropdown(
            id='model-selector', 
            options = [{'label': 'Random Forest', 'value': 'RF'},
                       {'label': 'Linear Regression', 'value': 'LIN'},
                       {'label': 'Both', 'value': 'BOTH'}
                       ],
            value='LIN'
        )
        ]),
    
    html.Div([
        html.Div([
            html.H4('R2 Score'),
            html.P(id='r2-display')
        ]),
        html.Div([
            html.H4('Root Mean Square Error'),
            html.P(id = 'rmse-display')
        ]),
        html.Div([
            html.H4('Mean Absolute Error'),
            html.P(id = 'mae-display')
        ])
    ]),
    
    html.Div([
        dcc.Graph(id='time-series-graph')
    ]),
    
    html.Div([
        dcc.Graph(id="residuals-graph")
    ]),
    
    html.Div(
        children=[
            html.P(
                "Residual interpretation:",
                style={"fontWeight": "bold", "marginBottom": "4px"}
            ),
            html.P(
                "The spikes in the residuals occur primarily during sudden price changes. "
                "This is expected, as electricity prices are strongly dependent on the previous hour’s price. "
                "When rapid price jumps occur, even small forecast delays lead to large residuals. "
                "Random Forest produces smoother residuals, while the linear model reacts more directly to price jumps."
            )
        ],
        style={
            "maxWidth": "900px",
            "margin": "0 auto 30px auto",
            "color": "#333333",
            "fontSize": "15px"
        }
    ),
    
    html.Div(id="top-error-text"),
    
    html.Div(
            html.P(
                "From this, we can conclude that improving the model primarily requires enhancing its ability to predict extreme pricing events."
                ),
        style={
            "maxWidth": "900px",
            "margin": "0 auto 30px auto",
            "color": "#333333",
            "fontSize": "15px"
        }
    ),

    html.Div([ 
        html.Div([
            dcc.Graph(id='scatter-graph')
        ]),
        html.Div([
            dcc.Graph(id='importance-graph')
        ])
   ]),
    
    html.Div(
        children=[
            html.P(
                "Importance clarification:",
                style={"fontWeight": "bold", "marginBottom": "4px"}
            ),
            html.P(
                "We choose not to include the timelag variables as they were very dominating and made it alot harder to compare the other variables."
                )
        ],
        style={
            "maxWidth": "900px",
            "margin": "0 auto 30px auto",
            "color": "#333333",
            "fontSize": "15px"
        }
    ),       
])

def top_percent_error_share(df, top_percent=0.05):
    """
    Returns the share of total squared error coming from the top {top_procent}% highest prices.
    """
    df_sorted = df.sort_values("Actual")

    cutoff_idx = int((1 - top_percent) * len(df_sorted))
    threshold_price = df_sorted.iloc[cutoff_idx]["Actual"]

    top_df = df_sorted[df_sorted["Actual"] >= threshold_price]

    total_se = np.sum(df_sorted["Residual"] ** 2)
    top_se = np.sum(top_df["Residual"] ** 2)

    return top_se / total_se


@app.callback(
    [
        Output('time-series-graph', 'figure'),
        Output('residuals-graph', 'figure'),
        Output('scatter-graph', 'figure'),
        Output('importance-graph', 'figure'),
        Output('r2-display', 'children'),
        Output('rmse-display', 'children'),
        Output('mae-display', 'children'),
        Output('top-error-text', 'children'),
    ],
    [Input('model-selector', 'value')]
)


def update_all_graphs(selected_model):
    """
    Updates all dashboard components based on the user's model selection.

    Args:
        selected_model (str): 'RF' for Random Forest or 'LIN' for Linear Regression.

    Returns:
        tuple: Contains 3 Plotly figures (Time-series, Scatter, Bar) 
               and 3 strings for the metric displays (R2, RMSE, MAE).
    """
    color_lin = '#1abc9c'
    color_rf  = '#1f77b4'

    if selected_model == 'RF':
        df = results_rf_df
        imp_df = rf_importance_df
        main_color = color_rf 
    elif selected_model == 'LIN':
        df = results_lin_df
        imp_df = lin_importance_df
        main_color = color_lin
        
    
    if selected_model == 'BOTH':
        # Linear metrics
        r2_lin = r2_score(results_lin_df['Actual'], results_lin_df['Predicted'])
        rmse_lin = np.sqrt(mean_squared_error(
            results_lin_df['Actual'], results_lin_df['Predicted']
        ))
        mae_lin = mean_absolute_error(
            results_lin_df['Actual'], results_lin_df['Predicted']
        )
    
        # Random Forest metrics
        r2_rf = r2_score(results_rf_df['Actual'], results_rf_df['Predicted'])
        rmse_rf = np.sqrt(mean_squared_error(
            results_rf_df['Actual'], results_rf_df['Predicted']
        ))
        mae_rf = mean_absolute_error(
            results_rf_df['Actual'], results_rf_df['Predicted']
        )
    
        r2_text = f"LIN: {r2_lin:.3f} | RF: {r2_rf:.3f}"
        rmse_text = f"LIN: {rmse_lin:.2f} € | RF: {rmse_rf:.2f} €"
        mae_text = f"LIN: {mae_lin:.2f} € | RF: {mae_rf:.2f} €"
    
    else:
        r2 = r2_score(df['Actual'], df['Predicted'])
        rmse = np.sqrt(mean_squared_error(df['Actual'], df['Predicted']))
        mae = mean_absolute_error(df['Actual'], df['Predicted'])
    
        r2_text = f"{r2:.3f}"
        rmse_text = f"{rmse:.2f} €"
        mae_text = f"{mae:.2f} €"

    model_name = {
        'RF': 'Random Forest',
        'LIN': 'Linear Regression',
        'BOTH': 'Random Forest vs Linear Regression'
    }[selected_model]
    
    fig1 = go.Figure()

    # Actual
    fig1.add_trace(go.Scatter(
        x=results_lin_df['Date'],
        y=results_lin_df['Actual'],
        mode='lines',
        name='Actual',
        line=dict(color='#333', width=1)
    ))
    
    if selected_model in ['LIN', 'BOTH']:
        fig1.add_trace(go.Scatter(
            x=results_lin_df['Date'],
            y=results_lin_df['Predicted'],
            mode='lines',
            name='Predicted (Linear)',
            line=dict(color=color_lin, width=2)
        ))
    
    if selected_model in ['RF', 'BOTH']:
        fig1.add_trace(go.Scatter(
            x=results_rf_df['Date'],
            y=results_rf_df['Predicted'],
            mode='lines',
            name='Predicted (Random Forest)',
            line=dict(color=color_rf, width=2)
        ))
        
    fig1.update_layout(
        title=f"Predictions over time: {model_name}",
        xaxis_title="Date",
        yaxis_title="EUR")

    
    fig_residuals = go.Figure()

    if selected_model in ['LIN', 'BOTH']:
        fig_residuals.add_trace(go.Scatter(
            x=results_lin_df['Date'],
            y=results_lin_df['Residual'],
            mode='lines',
            name='Residuals (Linear)',
            line=dict(color=color_lin, width=1)
        ))
    
    if selected_model in ['RF', 'BOTH']:
        fig_residuals.add_trace(go.Scatter(
            x=results_rf_df['Date'],
            y=results_rf_df['Residual'],
            mode='lines',
            name='Residuals (Random Forest)',
            line=dict(color=color_rf, width=1)
        ))
    
    fig_residuals.add_hline(y=0, line_dash='dash', line_color='red')
    
    fig_residuals.update_layout(
        title="Residuals over time",
        xaxis_title="Date",
        yaxis_title="Residual (EUR)",
        showlegend=True
    )

    if selected_model == 'LIN':
        share = top_percent_error_share(results_lin_df)
        top_error_text = (
            f"Top 5% highest prices account for {share:.1%} of the total squared error. "
            "This indicates that extreme price events dominate the model error."
        )
    
    elif selected_model == 'RF':
        share = top_percent_error_share(results_rf_df)
        top_error_text = (
            f"Top 5% highest prices account for {share:.1%} of the total squared error. "
            "This reflects the difficulty of predicting extreme price spikes and that the large residuals influence our statistical metrics the most."
        )
    
    else:  # BOTH
        share_lin = top_percent_error_share(results_lin_df)
        share_rf = top_percent_error_share(results_rf_df)
    
        top_error_text = (
            f"Top 5% highest prices account for a large share of the total squared error "
            f"(Linear: {share_lin:.1%}, Random Forest: {share_rf:.1%}). "
            "This highlights that model errors are dominated by extreme price events."
        )

    
    fig2 = go.Figure()

    if selected_model in ['LIN', 'BOTH']:
        fig2.add_trace(go.Scatter(
            x=results_lin_df['Actual'],
            y=results_lin_df['Predicted'],
            mode='markers',
            name='Linear Regression',
            marker=dict(color=color_lin, opacity=0.5)
        ))
    
    if selected_model in ['RF', 'BOTH']:
        fig2.add_trace(go.Scatter(
            x=results_rf_df['Actual'],
            y=results_rf_df['Predicted'],
            mode='markers',
            name='Random Forest',
            marker=dict(color=color_rf, opacity=0.5)
        ))

    
    min_val = min(results_lin_df['Actual'].min(), results_rf_df['Actual'].min())
    max_val = max(results_lin_df['Actual'].max(), results_rf_df['Actual'].max())
        
    fig2.add_trace(go.Scatter(
        x=[min_val,
        max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Perfect match'))
    
    fig2.update_layout(
        title="Prediction Accuracy",
        xaxis_title="Actual Price",
        yaxis_title="Predicted Price",
        plot_bgcolor='white'
    )
    fig3 = go.Figure()
    
    if selected_model != 'BOTH':
        fig3.add_trace(go.Bar(
            x=imp_df['Importance'],  
            y=imp_df['Feature'],    
            orientation='h',        
            marker=dict(color=main_color) 
        ))
    
        fig3.update_layout(
            title="Feature Importance (except lag-variables)",
            xaxis_title="Importance",
            plot_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor='#eee') 
        )
    else:
        fig3.update_layout(
            title="Select a single model to view feature importance",
            plot_bgcolor="white"
    )


    return (
        fig1,
        fig_residuals,
        fig2,
        fig3,
        r2_text,
        rmse_text,
        mae_text,
        html.P(
            top_error_text,
            style={
                "maxWidth": "900px",
                "margin": "0 auto 30px auto",
                "fontSize": "15px",
                "color": "#333"
            }
        )
    )


    
    


#%%

def open_browser():
    webbrowser.open_new(f'http://{host}:{port}')

if __name__== '__main__':
    Timer(1, open_browser).start()
    app.run(debug=True, host=host, port=port)
    