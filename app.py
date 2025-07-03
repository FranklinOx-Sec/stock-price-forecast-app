import yfinance as yf
import pandas as pd
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA

# Streamlit page setup
st.set_page_config(page_title="Stock Price Forecast", layout="centered")
st.title("📈 Stock Price Forecast: Google & Microsoft")

# Sidebar input
company = st.sidebar.selectbox("Select a company", ["Google (GOOG)", "Microsoft (MSFT)"])
forecast_days = st.sidebar.selectbox("Forecast days", [7, 14, 30, 60, 90])

# Map company name to ticker
ticker = "GOOG" if company == "Google (GOOG)" else "MSFT"

# Download stock data
df = yf.download(ticker, start="2020-01-01", end="2024-12-31")
df = df[['Close']].dropna()
df.columns = ['Price']  # Rename column to avoid confusion
df.index.name = 'Date'

# Display historical chart
st.subheader(f"Historical Closing Price of {ticker}")
st.line_chart(df)

# ARIMA Forecast
try:
    model = ARIMA(df['Price'], order=(5, 1, 2))
    model_fit = model.fit()

    # Forecast future prices
    forecast = model_fit.forecast(steps=forecast_days)
    last_date = df.index[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)

    forecast_df = pd.DataFrame({'Price': forecast}, index=future_dates)

    # Combine historical and forecast
    combined = pd.concat([df, forecast_df])
    combined['Label'] = ['Actual'] * len(df) + ['Forecast'] * len(forecast_df)

    # Plot combined forecast
    st.subheader(f"{forecast_days}-Day Forecast")
    st.line_chart(combined[['Price']])

except Exception as e:
    st.error(f"Model failed: {e}")
