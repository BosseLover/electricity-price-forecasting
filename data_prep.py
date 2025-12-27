#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 16:35:37 2025

@author: elingarvare
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
    # Define URL with parameters
    weather_url = (
        "https://archive-api.open-meteo.com/v1/archive"
        "?latitude=55.70584&longitude=13.19321"
        f"&start_date={start_date}&end_date={end_date}"
        "&hourly=temperature_2m,wind_speed_10m"
    )
    
    # Fetch data
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
    1. Fetches weather and electricity data.
    2. Merges them on 'Date'.
    3. Adds columns for Hour, Month, and Weekday.
    4. Converts wind speed from km/h to m/s.
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

    # Save to CSV 
    #df_total.to_csv("project_elc_temp.csv", index=False)

# This block ensures the script runs only when executed directly
if __name__ == "__main__":
    process_and_merge_data()