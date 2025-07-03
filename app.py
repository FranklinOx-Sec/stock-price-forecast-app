import streamlit as st
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# App config
st.set_page_config(page_title="Stock Forecast", layout="centered")
st.title("📈 Stock Price Forecast: Google & Microsoft")

# Sidebar inputs
company = st.sidebar.selectbox("Select Company", ["Google (GOOG)", "Microsoft (MSFT)"])
forecast_days = st.sidebar.slider("Forecast Days", 7, 90, 30)

# Ticker mapping
ticker = "GOOG" if "Google" in company else "MSFT"

# Download data
df = yf.download(ticker, start="2020-01-01", end="2024-12-31")

# ✅ Fix column handling
if df.empty:
    st.error("No data downloaded.")
    st.stop()

# Flatten columns if multi-indexed
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [' '.join(col).strip() for col in df.columns.values]

# Find the 'Close' column
close_col = next((col for col in df.columns if 'close' in col.lower()), None)

if close_col is None:
    st.error("No 'Close' column found in data.")
    st.write("Available columns:", df.columns)
    st.stop()

# Clean up and prepare
df = df[[close_col]].copy()
df.columns = ['Price']
df.index = pd.to_datetime(df.index)

# Show historical data
st.subheader(f"Historical Price of {ticker}")
st.line_chart(df)

# Fit ARIMA model
try:
    model = ARIMA(df['Price'], order=(5, 1, 2))
    fitted_model = model.fit()
    forecast = fitted_model.forecast(steps=forecast_days)
except Exception as e:
    st.error(f"ARIMA model error: {e}")
    st.stop()

# Forecast dates
last_date = df.index[-1]
future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=forecast_days)

# Create forecast DataFrame
forecast_df = pd.DataFrame({'Price': forecast.values}, index=future_dates)

# Combine actual + forecast
combined_df = pd.concat([df, forecast_df])

# Plot final chart
st.subheader(f"{forecast_days}-Day Forecast for {ticker}")
st.line_chart(combined_df)
