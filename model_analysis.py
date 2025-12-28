#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Python Project – Electricity Price Forecasting

This script builds and evaluates two predictive models:
- Linear Regression
- Random Forest Regression

The models are trained on historical electricity prices combined with
weather and time-based features. The script includes:
- Feature engineering with lagged prices
- Model training and evaluation
- Residual analysis
- Analysis of error concentration in extreme price events

Author: Elin Garvare and Anton Holmberg
Created: Sat Dec 20, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# -------------------------------------------------------------------
# Data loading and preprocessing
# -------------------------------------------------------------------

# Load dataset
df = pd.read_csv('project_elc_temp.csv')

# Convert Date column to datetime and ensure chronological order
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Create lagged price features:
# - 1 hour lag
# - 24 hour lag (daily pattern)
# - 168 hour lag (weekly pattern)
df["Price_lag_1"] = df["PriceEUR"].shift(1)
df["Price_lag_24"] = df["PriceEUR"].shift(24)
df["Price_lag_168"] = df["PriceEUR"].shift(168)

# Remove rows with missing values caused by lagging
df = df.dropna().reset_index(drop=True)


# -------------------------------------------------------------------
# Feature selection
# -------------------------------------------------------------------

# Feature matrix including weather, calendar effects, and price lags
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

# Target variable: electricity price
y = df["PriceEUR"]



def plot_test(y_test,y_pred):    
    """
    Creates a scatter plot comparing actual and predicted prices.

    A perfect prediction would place all points along the diagonal
    reference line.

    Parameters
    ----------
    y_test : pd.Series
        True electricity prices from the test set.
    y_pred : np.ndarray
        Predicted prices from the model.
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
    Visualizes model performance across hours of the day.

    Useful for assessing whether the model captures
    intraday price patterns such as morning and evening peaks.

    Parameters
    ----------
    hours : pd.Series
        Hour-of-day values (0–23).
    y_test : pd.Series
        True electricity prices.
    y_pred : np.ndarray
        Predicted electricity prices.
    """
    
    plt.scatter(hours, y_test, color='blue', label='Correct Price')
    plt.scatter(hours, y_pred, color='red', label='Predictions')
    
    plt.xlabel('Time of the Day (0-23)')
    plt.ylabel('Price (EUR)')
    plt.title('Correct results vs Predictions from Chosen Model')
    plt.legend()
    plt.show()
    

def linear_reg_model(X, y):
    """
    Trains and evaluates a linear regression model using a time-based split.

    The first 80% of the data is used for training and the remaining
    20% for testing to avoid look-ahead bias.

    Returns
    -------
    lin_reg : LinearRegression
        Trained linear regression model.
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : pd.Series
        True test target values.
    y_pred : np.ndarray
        Model predictions on the test set.
    df : pd.DataFrame
        Full dataset (for date alignment).
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target values.
    """
    split_idx = int(0.8 * len(X))
    
    X_train = X.iloc[:split_idx]
    X_test  = X.iloc[split_idx:]
    
    y_train = y.iloc[:split_idx]
    y_test  = y.iloc[split_idx:]
    
    lin_reg = LinearRegression().fit(X_train, y_train)
    y_pred = lin_reg.predict(X_test)
    
    return lin_reg, X_test, y_test, y_pred, df,  X_train, y_train

def rf_reg_model(X, y):
    """
    Trains and evaluates a Random Forest regression model
    using a time-based train-test split.

    The model is regularized using limited tree depth and
    minimum leaf size to reduce overfitting.

    Returns
    -------
    rf_reg : RandomForestRegressor
        Trained Random Forest model.
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : pd.Series
        True test target values.
    y_pred : np.ndarray
        Model predictions on the test set.
    df : pd.DataFrame
        Full dataset (for date alignment).
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target values.
    """
    split_idx = int(0.8 * len(X)) 
    
    X_train = X.iloc[:split_idx]
    X_test  = X.iloc[split_idx:]
    
    y_train = y.iloc[:split_idx]
    y_test  = y.iloc[split_idx:]
    
    rf_reg=RandomForestRegressor(n_estimators=200, 
                                 max_depth=10,
                                 min_samples_leaf=10,
#                                random_state=42,
                                 n_jobs=-1
                                 ).fit(X_train, y_train)
    
    y_pred = rf_reg.predict(X_test)
    
    return rf_reg, X_test, y_test, y_pred, df, X_train, y_train


def compute_residuals(y_true, y_pred, dates):
    """
    Computes residuals for model evaluation.

    Residuals are defined as:
        residual = actual - predicted

    Parameters
    ----------
    y_true : pd.Series
        Actual electricity prices.
    y_pred : np.ndarray
        Model predictions.
    dates : pd.Series
        Datetime values aligned with y_true.

    Returns
    -------
    pd.DataFrame
        DataFrame containing Date, Actual, Predicted, and Residual.
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
    Plots residuals over time to identify systematic patterns,
    volatility clustering, or regime changes.

    Parameters
    ----------
    residual_df : pd.DataFrame
        Output from compute_residuals().
    """
    plt.figure(figsize=(12, 4))
    plt.plot(residual_df["Date"], residual_df["Residual"], alpha=0.6)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("Residual (EUR)")
    plt.title("Residuals over time (actual - predicted)")
    plt.show()

def print_residual_summary(residual_df):
    """
    Prints summary statistics of residuals and identifies
    large prediction errors.

    Large errors are defined as residuals exceeding
    two standard deviations.
    """
    print("\nResidual summary:")
    print(residual_df["Residual"].describe())

    large_errors = residual_df.loc[
        residual_df["Residual"].abs() > residual_df["Residual"].std() * 2
    ]

    print(f"\nNumber of large errors (>2 std): {len(large_errors)}")

def error_contribution_top_percent(residual_df, top_percent=0.05):
    """
    Quantifies how much of the total squared error originates
    from the top X% highest electricity prices.

    This analysis highlights whether model errors are dominated
    by extreme price events.

    Parameters
    ----------
    residual_df : pd.DataFrame
        Must contain 'Actual' and 'Residual'.
    top_percent : float, optional
        Fraction of highest prices to analyze (default is 0.05).

    Returns
    -------
    None
        Prints results to the console.
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

if __name__ == "__main__":
    """
    Runs model training, evaluation, and residual diagnostics.

    This block:
    - Trains both models
    - Prints standard performance metrics
    - Visualizes predictions and residuals
    - Analyzes the contribution of extreme price events
    """
    
    rf_reg, X_test, y_test, y_pred, df, X_train, y_train = rf_reg_model(X, y)
    lin_reg, X_test_lin, y_test_lin, y_pred_lin, df_lin, X_train_lin, y_train_lin = linear_reg_model(X, y)

    
  
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    mse_lin = mean_squared_error(y_test_lin, y_pred_lin)
    rmse_lin = np.sqrt(mse_lin)
    r2_lin = r2_score(y_test_lin, y_pred_lin)
    
    train_pred = rf_reg.predict(X_train)
    test_pred  = rf_reg.predict(X_test)
    
    rmse_train = np.sqrt(mean_squared_error(y_train, train_pred))
    rmse_test  = np.sqrt(mean_squared_error(y_test, test_pred))
    print(f"Mean price is {df['PriceEUR'].mean()}")
    
    print(f'MSE RF: {mse}')
    print(f'RMSE RF: {rmse}')
    print(f'R2 RF: {r2}')
    
    print(f'MSE LIN: {mse_lin}')
    print(f'RMSE LIN: {rmse_lin}')
    print(f'R2 LIN: {r2_lin}')
    
    print(f"Train RMSE RF: {rmse_train}")
    print(f"Test RMSE RF:  {rmse_test}")
    
    # 3. Plotta (Nu gör vi det här nere, så appen slipper pop-up fönster)
    plot_test(y_test, y_pred)
    
    X_test_plot = X_test['Hour'] 
    plot_test_hour(X_test_plot, y_test, y_pred)
    
    # Residual analys
    residual_df = compute_residuals(y_test_lin, y_pred_lin, df.loc[y_test.index, "Date"])
    plot_residuals_over_time(residual_df)
    print_residual_summary(residual_df)
    error_contribution_top_percent(residual_df)
    
    residual_df = compute_residuals(y_test, y_pred, df.loc[y_test.index, "Date"])
    plot_residuals_over_time(residual_df)
    print_residual_summary(residual_df)
    error_contribution_top_percent(residual_df)
