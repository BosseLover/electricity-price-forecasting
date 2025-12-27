# Electricity Price Forecasting Dashboard
**Author:** Anton Holmberg & Elin Garvare

## Project Overview:
This project is a Machine Learning application designed to forecast hourly electricity spot prices (EUR) for the **SE4-Region** (Southern Sweden). By combining historical weather data from Lund with electricity price records. The project compares the performance of a **Linear Regression** and a **Random Forest Regressor**.

The results are presented in a **Dash** web application, allowing users to analyze price trends, residuals and feature importance. 

## Features:
* **Automatic Data Fetching** The 'data_prep.py' script pulls data from **Open-Meteo** and **Energy Data Service** and creates a combined DataFrame with hourly time stamps with start date 2023-01-01 and end date 2024-12-31.
* **Smart Features** To improve prediction accuracy, the models utilize "lag variables":
    * `Price_lag_1` (Price 1 hour ago)
    * `Price_lag_24` (Price yesterday at the same time)
    * `Price_lag_168` (Price 1 week ago)
* **Model Comparison** Side-by-side comparison of Linear Regression and Random Forest performance metrics (R2, RMSE, MAE).
*  **Interactive Dashboard:**
    * **Time-Series Visualization:** History of Actual vs. Predicted prices.
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

## Dashboard Preview
<img width="1440" height="656" alt="first_dash" src="https://github.com/user-attachments/assets/f4cf58c3-91db-4069-a10f-ad2b3df760d6" />
<img width="1440" height="534" alt="sec_dash" src="https://github.com/user-attachments/assets/f304a022-043c-426a-a035-c7575b5f8722" />
<img width="1429" height="359" alt="third_dash" src="https://github.com/user-attachments/assets/a921a90b-e771-4c2e-836c-75e4dfa48a7d" />
<img width="1440" height="398" alt="fourth_dash" src="https://github.com/user-attachments/assets/f4485ddb-2014-498c-8115-f26877a8da83" />
<img width="1440" height="391" alt="fifth_dash" src="https://github.com/user-attachments/assets/aa66397b-565a-44a8-8d5a-4c89194c6623" />

## Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/BosseLover/electricity-price-forecasting.git
    cd electricity-price-forecasting
    ```

2.  **Install dependencies:**
    Run the following command to install all necessary libraries from the requirements file:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
**Run the Dashboard:**
   Launch the application by running:
    ```bash
    python app.py
    ```
