import pandas as pd                                              # Reads the dataset from a CSV file into a Pandas DataFrame.
import numpy as np                                               # Used for numerical operations and handling arrays in a more efficient way.
import matplotlib.pyplot as plt                                  # Used for data visualization.
from statsmodels.tsa.holtwinters import ExponentialSmoothing     # It allows modeling and forecasting time series data with trend and seasonality.
from statsmodels.tsa.seasonal import seasonal_decompose          # It can be applied to analyze the underlying patterns and structures in time series data.

# We used the Holt-Winters Exponential Smoothing method for time series forecasting

data = pd.read_csv("C:\CODING\CSV Files\AirPassengers.csv", parse_dates=True, index_col="Month")

# It creates a plot of the original air passenger data to visualize the trend and seasonality.
plt.figure(figsize=(12, 6))
plt.plot(data)
plt.title("Air Passenger Data")
plt.xlabel("Year")
plt.ylabel("Passengers")
plt.show()

# It decomposes the time series data into trend, seasonality, and residual components using multiplicative decomposition.
decomposition = seasonal_decompose(data, model="multiplicative")
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# It creates a subplot of four charts to visualize the original time series, trend, seasonality, and residuals separately.
plt.subplot(411)
plt.plot(data, label="Original")
plt.legend(loc="best")
plt.subplot(412)
plt.plot(trend, label="Trend")
plt.legend(loc="best")
plt.subplot(413)
plt.plot(seasonal, label="Seasonal")
plt.legend(loc="best")
plt.subplot(414)
plt.plot(residual, label="Residual")
plt.legend(loc="best")
plt.tight_layout()

# It initializes and fits an Exponential Smoothing model to the time series data. The model is configured with additive trend, additive seasonality, and a seasonal period of 12 (assuming monthly seasonality).
model = ExponentialSmoothing(data, trend="add", seasonal="add", seasonal_periods=12)
model_fit = model.fit()

# Make forecasts for the next 12 steps
forecast = model_fit.forecast(steps=12)

# It creates a plot showing both the original air passenger data and the forecasted values, providing a visual comparison of the model's predictions.
plt.figure(figsize=(12, 6))
plt.plot(data, label="Original")
plt.plot(forecast, label="Forecast")
plt.title("Air Passenger Data and Forecast")
plt.xlabel("Year")
plt.ylabel("Passengers")
plt.legend()
plt.show()
