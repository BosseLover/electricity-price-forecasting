#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 20 17:01:42 2025

@author: elingarvare
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('project_elc_temp.csv')

# Detta måste vi ändra för det är redan i data_prep men filen i repot gör det 
# inte så den måste sparas ner på nytt
# dvs ta bort denna rad när det är åtgärdat
df['WindSpeed']=df['WindSpeed']/3.6 

#print(df.head())

X = df[['Temperature', 'WindSpeed', 'Hour', 'Month', 'Weekday']]

y=df['PriceEUR']

def plot_test(y_test,y_pred):    
    
    
    plt.scatter(y_test, y_pred, color='blue', label='Predictions')
    
    m, M = y_test.min(), y_test.max() #faktisk linje 
    plt.plot([m, M], [m, M], color='red', linestyle='--', label='Perfect prediction')
    
    plt.xlabel('Correct Price (EUR)')
    plt.ylabel('Model Prediction (EUR)')
    plt.title('How well does the model predict the prices?')
    plt.legend()
    plt.show()
    

def plot_test_hour(hours, y_test, y_pred):
    
    plt.scatter(hours, y_test, color='blue', label='Correct Price')
    plt.scatter(hours, y_pred, color='red', label='Predictions')
    
    plt.xlabel('Time of the Day (0-23)')
    plt.ylabel('Price (EUR)')
    plt.title('Correct results vs Predictions from Chosen Model')
    plt.legend()
    plt.show()
    

def linear_reg_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    lin_reg = LinearRegression().fit(X_train, y_train)
    y_pred = lin_reg.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse =np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f'MSE: {mse}')
    print(f'RMSE {rmse}')
    print(f'Mean price is {df['PriceEUR'].mean()}')
    print(f'R2 {r2}')
    
    X_test_plot = X_test['Hour'] 
    plot_test(y_test,y_pred)
    plot_test_hour(X_test_plot,y_test,y_pred)


linear_reg_model(X,y)
# ganska dåliga resultat -> R2 lågt och snittfel ligger på 41€ medan snittpris 57€ så stort fel
#MSE: 1684.5131774229992
#RMSE 41.04282126539304
#Mean price is 57.33451622038929
#R2 0.2956275990164542

def rf_reg_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_reg=RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
    
    y_pred = rf_reg.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse =np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f'MSE: {mse}')
    print(f'RMSE {rmse}')
    print(f'Mean price is {df['PriceEUR'].mean()}')
    print(f'R2 {r2}')
    
    X_test_plot = X_test['Hour'] 
    plot_test(y_test,y_pred) #wow sån stor skillnad mot innan
    plot_test_hour(X_test_plot,y_test,y_pred) #jättestor förändring här också
    
    

rf_reg_model(X, y)

#Nu mycket bättre prediktioner
#MSE: 888.2003824841694
#RMSE 29.802690859789312
#Mean price is 57.33451622038929
#R2 0.6286025871747887

