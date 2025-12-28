#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Collection and Preprocessing for Electricity Price Forecasting

This script fetches and processes:
- Hourly weather data (temperature and wind speed) from the Open-Meteo API
- Hourly electricity spot prices from Energidataservice

The datasets are merged on a common datetime index and enriched with
time-based features (hour, month, weekday). The final dataset is prepared
for downstream machine learning analysis.

Author: Elin Garvare and Anton Holmberg
Created: Tue Dec 16, 2025
"""

import requests
import pandas as pd

def fetch_weather_data(start_date, end_date):
    """
    Fetches hourly weather data (temperature and wind speed) from the Open-Meteo API.

    Args:
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A DataFrame containing 'Date', 'Temperature', and 'WindSpeed'.
    """
    # API endpoint with query parameters
    weather_url = (
        "https://archive-api.open-meteo.com/v1/archive"
        "?latitude=55.70584&longitude=13.19321"
        f"&start_date={start_date}&end_date={end_date}"
        "&hourly=temperature_2m,wind_speed_10m"
    )
    
    # Send request to API (fetch data)
    response = requests.get(weather_url)
    weather_json = response.json()

    # Create DataFrame
    df = pd.DataFrame({
        'Date': weather_json['hourly']['time'],
        'Temperature': weather_json['hourly']['temperature_2m'],
        'WindSpeed': weather_json['hourly']['wind_speed_10m']
    })

    # Convert to datetime objects
    df['Date'] = pd.to_datetime(df['Date'])
    
    return df

def fetch_electricity_prices(start_date, end_date, price_area="SE4"):
    """
    Fetches hourly electricity spot prices from Energidataservice.

    Args:
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.
        price_area (str): The price area code (default is "SE4").

    Returns:
        pd.DataFrame: A DataFrame containing 'Date' and 'PriceEUR'.
    """
    # Define URL with parameters
    elec_url = (
        'https://api.energidataservice.dk/dataset/Elspotprices'
        f'?start={start_date}&end={end_date}'
        f'&filter={{"PriceArea":["{price_area}"]}}'
    )

    # Fetch data
    response = requests.get(elec_url)
    elec_json = response.json()

    # Create and clean DataFrame
    df = pd.DataFrame(elec_json['records'])
    df = df[['HourUTC', 'SpotPriceEUR']] # Keep only relevant columns

    # Rename columns to match weather data
    df = df.rename(columns={
        'HourUTC': 'Date',
        'SpotPriceEUR': 'PriceEUR' 
    })

    # Convert to datetime and sort
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')
    
    return df

def process_and_merge_data():
    """
    Main function to fetch, merge, and process weather and electricity datasets.
    
    Steps:
    1. Retrieve weather and electricity datasets via APIs.
    2. Merge datasets on the Date column using an inner join.
    3. Remove missing values and ensure correct time ordering.
    4. Add time-based features (hour, month, weekday).
    5. Convert wind speed from km/h to m/s.
    """
    start = "2023-01-01"
    end = "2024-12-31"

    print("Fetching weather data...")
    df_weather = fetch_weather_data(start, end)

    print("Fetching electricity data...")
    df_elec = fetch_electricity_prices(start, end)

    print("Merging datasets...")
    # Merge datasets (inner join to keep only matching timestamps)
    df_total = pd.merge(df_elec, df_weather, on='Date', how='inner')
    
    # Sort and reset index
    df_total = df_total.sort_values(by='Date').reset_index(drop=True)
    df_total = df_total.dropna()

    # Feature Engineering
    df_total['Hour'] = df_total['Date'].dt.hour
    df_total['Month'] = df_total['Date'].dt.month
    df_total['Weekday'] = df_total['Date'].dt.weekday # 0=Monday, 6=Sunday

    # Convert WindSpeed from km/h to m/s
    df_total['WindSpeed'] = df_total['WindSpeed'] / 3.6

    print("\n--- FINAL DATASET PREVIEW ---")
    print(df_total.head())
    print(f"\nTotal rows ready for analysis: {len(df_total)}")

    # Save to CSV (optional and only needed once)
    #df_total.to_csv("project_elc_temp.csv", index=False) 

# This block ensures the script runs only when executed directly
if __name__ == "__main__":
    """
    Executes the full data collection and preprocessing pipeline
    when the script is run directly.
    """
    process_and_merge_data()