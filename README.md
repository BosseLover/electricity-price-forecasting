# Electricity Price Forecasting Dashboard
**Author:** Anton Holmberg & Elin Garvare

##Project Overview:
This project is a Machine Learning application designed to forecast hourly electricity spot prices (EUR) for the **SE4-Region** (Southern Sweden). By combining historical weather data with electriciyt price records, the project compares the performance of a **Linear Regression** and a **Random Forest Regressor**.

The results are presented in a **Dash** web application, allowing users to analyze price trends, residuals and feature importance. 

##Features:
* **Automatic Data Fetching** The 'data_prep.py' script pulls data from **Open-Meteo** and **Energy Data Service** and creates a combined DataFrame with hourly time stamps with start date 2023-01-01 and end date 2024-12-31.
* **Smart Features** To improve the predictions the model uses three "lag variables" in the code. 'Price_lag_1' being the price 1 hour ago, 'Price_lag_24' yesterday, 'Price_lag_168' 1 week ago.
* **Model Comparison** Side-by-side comparison of Linear Regression and Random Forest performance metrics (R2, RMSE, MAE).
*  **Interactive Dashboard:**
    * **Time-Series Visualization:** Zoomable history of Actual vs. Predicted prices.
    * **Residual Analysis:** Plots residuals over time to identify periods of high error.
    * **Error Breakdown:** Quantifies how much extreme price events contribute to the total error.
    * **Accuracy Scatter Plot:** Compares the forecast against reality. The red dashed line shows a "perfect score"—the closer the dots are to this line, the better the model is performing.
    * **Dynamic Feature Importance:** Visualizes which variables drive the model's decisions (excluding lags).

## Project Structure
| File | Description |
| :--- | :--- |
| `data_prep.py` | Fetches raw weather and electricity data from APIs, merges them, cleans the dataset, and saves it to `project_elc_temp.csv`. |
| `model_analysis.py` | Loads the CSV, performs feature engineering (creating lags), trains the models, and calculates performance metrics. |
| `app.py` | The main entry point. Runs the Dash server, handles callbacks, and renders the interactive dashboard. |
| `project_elc_temp.csv` | The processed dataset used in the model.|
