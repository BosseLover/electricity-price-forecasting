#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 25 17:16:36 2025

@author: elingarvare
"""
import numpy as np
import pandas as pd

import plotly.express as px
import matplotlib.pyplot as plt 
from dash import html, dcc, Input, Output, Dash
from sklearn.metrics import mean_squared_error

import webbrowser
from threading import Timer
import plotly.graph_objects as go

from model_analysis import rf_reg_model, X, y

app = Dash(__name__)
host = 'localhost'
port = 8050

def open_browser():
    webbrowser.open_new(f'http://{host}:{port}')

rf_reg, X_test, y_test, y_pred, df_full, X_train, y_train = rf_reg_model(X, y)

if __name__== '__main__':
    Timer(1, open_browser).start()
    app.run(debug=True, host=host, port=port)

results_df = pd.DataFrame({
    'Date': df_full.loc[y_test.index, 'Date'],
    'Actual': y_test,
    'Predicted': y_pred
}).sort_values('Date')

importances = rf_reg.feature_importances_
indices = np.argsort(importances)[::-1]


