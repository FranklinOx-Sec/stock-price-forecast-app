
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
forecast = fitted.forecast(steps=forecast_days)

# Plot forecast
forecast_index = pd.date_range(start=close.index[-1], periods=forecast_days, freq='B')
forecast_df = pd.Series(forecast, index=forecast_index)

st.write(f"### {forecast_days}-Day Forecast")
st.line_chart(pd.concat([close, forecast_df]))
