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


df = pd.read_csv('project_elc_temp.csv')

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)


#Pris lagningar, 1 timme, 24 timmar, 168 timmar (1 vecka tillbaka)
df["Price_lag_1"]   = df["PriceEUR"].shift(1)
df["Price_lag_24"]  = df["PriceEUR"].shift(24)
df["Price_lag_168"] = df["PriceEUR"].shift(168)

df = df.dropna().reset_index(drop=True)



#jag lägger till den nya paramterna
X = df[
    [
        "Temperature",
        "WindSpeed",
        "Hour",
        "Month",
        "Weekday",
        "Price_lag_1",
        "Price_lag_24",
        "Price_lag_168",
    ]
]
y = df["PriceEUR"]


def plot_test(y_test,y_pred):    
    """
    Scatter plot comparing Actual vs Predicted prices.
     Ideally, points should cluster around the red diagonal line.
    """
    
    plt.scatter(y_test, y_pred, color='blue', label='Predictions')
    
    m, M = y_test.min(), y_test.max() #faktisk linje 
    plt.plot([m, M], [m, M], color='red', linestyle='--', label='Perfect prediction')
    
    plt.xlabel('Correct Price (EUR)')
    plt.ylabel('Model Prediction (EUR)')
    plt.title('How well does the model predict the prices?')
    plt.legend()
    plt.show()
    

def plot_test_hour(hours, y_test, y_pred):
    """
    Visualizes how the model tracks prices throughout the hours of the day.
    Helps to see if the model captures morning/evening peaks.
    """
    
    plt.scatter(hours, y_test, color='blue', label='Correct Price')
    plt.scatter(hours, y_pred, color='red', label='Predictions')
    
    plt.xlabel('Time of the Day (0-23)')
    plt.ylabel('Price (EUR)')
    plt.title('Correct results vs Predictions from Chosen Model')
    plt.legend()
    plt.show()
    

#den linära metoden måste ändras på här om den ska fungera nu med tid aspekten

def linear_reg_model(X, y):
    split_idx = int(0.8 * len(X))
    
    X_train = X.iloc[:split_idx]
    X_test  = X.iloc[split_idx:]
    
    y_train = y.iloc[:split_idx]
    y_test  = y.iloc[split_idx:]
    
    lin_reg = LinearRegression().fit(X_train, y_train)
    y_pred = lin_reg.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse =np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f'MSE: {mse}')
    print(f'RMSE {rmse}')
    print(f"Mean price is {df['PriceEUR'].mean()}")
    print(f'R2 {r2}')
    
    X_test_plot = X_test['Hour'] 
    plot_test(y_test,y_pred)
    plot_test_hour(X_test_plot,y_test,y_pred)


#linear_reg_model(X,y)
# ganska dåliga resultat -> R2 lågt och snittfel ligger på 41€ medan snittpris 57€ så stort fel
#MSE: 1684.5131774229992
#RMSE 41.04282126539304
#Mean price is 57.33451622038929
#R2 0.2956275990164542

def compute_residuals(y_true, y_pred, dates):
    """
    Computes residuals and returns a DataFrame for analysis.

    Args:
        y_true (pd.Series): Actual target values (indexed).
        y_pred (np.ndarray): Model predictions.
        dates (pd.Series): Datetime values aligned with y_true.

    Returns:
        pd.DataFrame: DataFrame with Date, Actual, Predicted, Residual
    """
    residual_df = pd.DataFrame({
        "Date": dates.values,
        "Actual": y_true.values,
        "Predicted": y_pred,
    })

    residual_df["Residual"] = residual_df["Actual"] - residual_df["Predicted"]
    return residual_df


def plot_residuals_over_time(residual_df):
    """
    Plots residuals over time.

    Args:
        residual_df (pd.DataFrame): Output from compute_residuals
    """
    plt.figure(figsize=(12, 4))
    plt.plot(residual_df["Date"], residual_df["Residual"], alpha=0.6)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("Residual (EUR)")
    plt.title("Residuals over time (actual - predicted)")
    plt.show()

def print_residual_summary(residual_df):
    print("\nResidual summary:")
    print(residual_df["Residual"].describe())

    large_errors = residual_df.loc[
        residual_df["Residual"].abs() > residual_df["Residual"].std() * 2
    ]

    print(f"\nNumber of large errors (>2 std): {len(large_errors)}")

def error_contribution_top_percent(residual_df, top_percent=0.05):
    """
    Quantifies how much of the total squared error comes from the top X% prices.

    Args:
        residual_df (pd.DataFrame): Must contain 'Actual' and 'Residual'
        top_percent (float): Fraction of highest prices to analyze (default 5%)

    Returns:
        None (prints results)
    """
    df_sorted = residual_df.sort_values("Actual")

    # Threshold for top X%
    cutoff_idx = int((1 - top_percent) * len(df_sorted))
    threshold_price = df_sorted.iloc[cutoff_idx]["Actual"]

    top_df = df_sorted[df_sorted["Actual"] >= threshold_price]
    rest_df = df_sorted[df_sorted["Actual"] < threshold_price]

    # Squared errors
    total_se = np.sum(residual_df["Residual"] ** 2)
    top_se = np.sum(top_df["Residual"] ** 2)
    rest_se = np.sum(rest_df["Residual"] ** 2)

    print(f"\nTop {int(top_percent*100)}% price threshold: {threshold_price:.2f} EUR")
    print(f"Share of data points: {len(top_df) / len(df_sorted):.1%}")
    print(f"Share of total squared error from top {int(top_percent*100)}% prices: {top_se / total_se:.1%}")
    print(f"Share of total squared error from remaining {int((1-top_percent)*100)}% prices: {rest_se / total_se:.1%}")


def rf_reg_model(X, y):
    split_idx = int(0.8 * len(X))
    
    X_train = X.iloc[:split_idx]
    X_test  = X.iloc[split_idx:]
    
    y_train = y.iloc[:split_idx]
    y_test  = y.iloc[split_idx:]
    
    rf_reg=RandomForestRegressor(n_estimators=200, 
                                 max_depth=10,
                                 min_samples_leaf=10,
#                                 random_state=42,
                                 n_jobs=-1
                                 ).fit(X_train, y_train)
    
    y_pred = rf_reg.predict(X_test)

    # mse = mean_squared_error(y_test, y_pred)
    # rmse =np.sqrt(mse)
    # r2 = r2_score(y_test, y_pred)
    
    # print(f'MSE: {mse}')
    # print(f'RMSE {rmse}')
    # print(f"Mean price is {df['PriceEUR'].mean()}")
    # print(f'R2 {r2}')
    
    # train_pred = rf_reg.predict(X_train)
    # test_pred  = rf_reg.predict(X_test)
    
    # rmse_train = np.sqrt(mean_squared_error(y_train, train_pred))
    # rmse_test  = np.sqrt(mean_squared_error(y_test, test_pred))
    
    # print(f"Train RMSE: {rmse_train}")
    # print(f"Test RMSE:  {rmse_test}")

    
    
    # # Residual analysis
    # residual_df = compute_residuals(
    #     y_true=y_test,
    #     y_pred=y_pred,
    #     dates=df.loc[y_test.index, "Date"]
    # )
    
    # plot_residuals_over_time(residual_df)
    
    # print_residual_summary(residual_df)
    
    # error_contribution_top_percent(residual_df, top_percent=0.05) 
    
    return rf_reg, X_test, y_test, y_pred, df, X_train, y_train

if __name__ == "__main__":

    rf_reg, X_test, y_test, y_pred, df, X_train, y_train = rf_reg_model(X, y)
    
  
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    train_pred = rf_reg.predict(X_train)
    test_pred  = rf_reg.predict(X_test)
    
    rmse_train = np.sqrt(mean_squared_error(y_train, train_pred))
    rmse_test  = np.sqrt(mean_squared_error(y_test, test_pred))
    print(f"Mean price is {df['PriceEUR'].mean()}")
    
    print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')
    print(f'R2: {r2}')
    
    print(f"Train RMSE: {rmse_train}")
    print(f"Test RMSE:  {rmse_test}")
    
    # 3. Plotta (Nu gör vi det här nere, så appen slipper pop-up fönster)
    plot_test(y_test, y_pred)
    
    X_test_plot = X_test['Hour'] 
    plot_test_hour(X_test_plot, y_test, y_pred)
    
    # Residual analys
    residual_df = compute_residuals(y_test, y_pred, df.loc[y_test.index, "Date"])
    plot_residuals_over_time(residual_df)
    print_residual_summary(residual_df)
    error_contribution_top_percent(residual_df)
    
    



#med tidserie tillägget fick jag detta resultat:
#MSE: 531.7507110493614
#RMSE 23.059720532767983
#Mean price is 57.001264460434555
#R2 0.8580644218475701
