
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="Stock Forecast App", layout="wide")
st.title("📈 Google & Microsoft Stock Price Forecast")

# Sidebar controls
company = st.sidebar.selectbox("Select Company", ["Google (GOOG)", "Microsoft (MSFT)"])
forecast_days = st.sidebar.slider("Number of days to forecast", 7, 90, 30)

ticker = "GOOG" if company == "Google (GOOG)" else "MSFT"

# Fetch data
df = yf.download(ticker, start="2020-01-01", end="2024-12-31")
close = df["Close"].dropna()

st.write(f"### Historical Close Price for {ticker}")
st.line_chart(close)

# Fit ARIMA model
model = ARIMA(close, order=(5,1,2))
fitted = model.fit()

# Forecast future values
forecast = fitted.forecast(steps=forecast_days)

# Create future index
last_date = close.index[-1]
forecast_index = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)

# Convert to DataFrame for plotting
forecast_df = pd.DataFrame({'Forecast': forecast}, index=forecast_index)

# Combine actual + forecast
combined_df = pd.concat([close, forecast_df['Forecast']])

# Plot
st.write(f"### Forecasted Closing Prices for {forecast_days} Business Days")
st.line_chart(combined_df)
