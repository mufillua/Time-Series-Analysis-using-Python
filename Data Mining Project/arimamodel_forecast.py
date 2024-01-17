# Time Series Data Visualization

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Sample dataset
data = pd.read_csv('C:\CODING\CSV Files\Airline-passengers.csv')

# Create a time series plot
plt.figure(figsize=(12, 6))
plt.plot(data['Month'], data['Passengers'])
plt.xlabel('Date')
plt.ylabel('Passengers')
plt.title('Monthly Airline Passengers')

# Rotate x-axis labels for readability
plt.xticks(rotation=45)
plt.show()


# Forecasting using arima model

# Load the "Airline Passengers" dataset
data_url = "F:\Learning_Work\Vs_Work\DM_Project\Airline-passengers.csv"
data = pd.read_csv(data_url)
data['Month'] = pd.to_datetime(data['Month'])   # used to convert the date column to date and time datatype 
data.set_index('Month', inplace=True)           # to set the index of month column accor to data frame 
                                                # and inplace used so it will modyfy the dataframe without returning the dataframe

# Visualize the original time series data
plt.figure(figsize=(12, 6))                 # screen size in width and height
plt.plot(data.index, data['Passengers'])    # passenger column at y axis and month data at x axis
plt.xlabel('Date')
plt.ylabel('Passengers')
plt.title('Monthly Airline Passengers')
plt.show()

# Fit an ARIMA model
model = ARIMA(data, order=(5, 1, 0))        # Example order, you can fine-tune this
model_fit = model.fit()

# Make forecasts
forecast_steps = 12                         # Adjust the number of forecasted months
forecast = model_fit.forecast(steps=forecast_steps)

# Visualize the forecast
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Passengers'], label='Observed')
plt.plot(pd.date_range(start=data.index[-1], periods=forecast_steps, freq='M'), forecast, label='Forecast', color='red')
plt.xlabel('Date')
plt.ylabel('Passengers')
plt.title('Monthly Airline Passengers Forecast')
plt.legend()
plt.show()