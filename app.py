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
                       {'label': 'Linear Regression', 'value': 'LIN'}  
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
        
        html.Div([
            dcc.Graph(id='scatter-graph')
            ]),
        
        html.Div([
            dcc.Graph(id='importance-graph')
        ])
        
        
   ])
    
            
])

@app.callback(
    [Output('time-series-graph', 'figure'),
     Output('scatter-graph', 'figure'),
     Output('importance-graph', 'figure'),
     Output('r2-display', 'children'),
     Output('rmse-display', 'children'),
     Output('mae-display', 'children')],
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
    
    if selected_model == 'RF':
        df = results_rf_df
        imp_df = rf_importance_df
        main_color = '#1f77b4' 
        model_name = "Random Forest"
    else:
        df = results_lin_df
        imp_df = lin_importance_df
        main_color = '#1abc9c' 
        model_name = "Linear Regression"
    
    r2 = r2_score(df['Actual'], df['Predicted'])
    mse = mean_squared_error(df['Actual'], df['Predicted'])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(df['Actual'], df['Predicted'])
    
    r2_text = f"{r2:.3f}"
    rmse_text = f"{rmse:.2f} €"
    mae_text = f"{mae:.2f} €"
    
    fig1=go.Figure()
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['Actual'], mode='lines', name='Actual', line=dict(color='#333', width=1)))
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['Predicted'], mode='lines', name='Predicted', line=dict(color=main_color, width=2)))
    fig1.update_layout(title=f"Predictions over time: {model_name}", plot_bgcolor='white', xaxis_title="Date", yaxis_title="EUR")
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df['Actual'], 
        y=df['Predicted'], 
        mode='markers', 
        marker=dict(color=main_color, opacity=0.5),
        name='Data'))
    
    min_val, max_val = df['Actual'].min(), df['Actual'].max()
    fig2.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', line=dict(color='red', dash='dash'), name='Perfect match'))
    
    fig2.update_layout(
        title="Prediction Accuracy",
        xaxis_title="Actual Price",
        yaxis_title="Predicted Price",
        plot_bgcolor='white'
    )
    
    fig3 = go.Figure()
    
    fig3.add_trace(go.Bar(
        x=imp_df['Importance'],  
        y=imp_df['Feature'],    
        orientation='h',        
        marker=dict(color=main_color) 
    ))
    
    fig3.update_layout(
        title="Feature Importance (except lag-variables)",
        xaxis_title="Importance",
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='#eee') 
    )

    return fig1, fig2, fig3, r2_text, rmse_text, mae_text

    
    


#%%

def open_browser():
    webbrowser.open_new(f'http://{host}:{port}')

if __name__== '__main__':
    Timer(1, open_browser).start()
    app.run(debug=True, host=host, port=port)
