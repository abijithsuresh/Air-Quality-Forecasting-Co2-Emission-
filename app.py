import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import pickle

# Load the ARIMA model from the pickle file
arima_model = pickle.load(open('CO2_Forecast_arima_yearly.pkl', 'rb'))

# Function to forecast future values
def forecast_future_values(model, start, end):
    forecast = model.predict(start=start, end=end, typ='levels')
    return forecast

# Function to display the forecast plot
def display_forecast_plot(original_data, forecast_data, title):
    plt.figure(figsize=(12, 6))
    plt.plot(original_data, label='Original')
    plt.plot(forecast_data, label='Forecast')
    plt.title(title)
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('CO2 Levels')
    plt.grid(True)
    st.pyplot()

# Streamlit app
def main():
    st.title('CO2 Emission Forecasting with ARIMA Model')
    st.sidebar.title('Settings')

    # Upload the CO2 data
    uploaded_file = st.sidebar.file_uploader('Upload CSV file', type=['csv'])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write('### Uploaded Data:')
        st.write(data.head())

        # Forecast future values
        st.sidebar.subheader('Forecast Future Values')
        start_year = st.sidebar.number_input('Start Year', min_value=2000, max_value=2030, step=1)
        end_year = st.sidebar.number_input('End Year', min_value=2026, max_value=2035, step=1)

        if start_year >= end_year:
            st.sidebar.error('End year must be greater than start year.')
        else:
            forecast_button = st.sidebar.button('Forecast')

            if forecast_button:
                forecast_start = data.index[-1] + 1
                forecast_end = forecast_start + (end_year - start_year)
                forecast_result = forecast_future_values(arima_model, start=forecast_start, end=forecast_end)
                display_forecast_plot(data['CO2'], forecast_result, 'CO2 Emission Forecast')

if __name__ == "__main__":
    main()
