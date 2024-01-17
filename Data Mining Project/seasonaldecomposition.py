import pandas as pd                                # Reads the dataset from a CSV file into a Pandas DataFrame.
import matplotlib.pyplot as plt                    # Used for data visualization.
from statsmodels.tsa.seasonal import STL           # Used for performing Seasonal-Trend decomposition using LOESS (Locally Weighted Scatterplot Smoothing).
from sklearn.metrics import mean_squared_error     # Provides the mean_squared_error metric, which is used for evaluating the performance of the model by calculating the RMSE.
from math import sqrt                              # Utilized to calculate the square root of the mean squared error, resulting in the RMSE.

# Load the Air Passenger dataset
data = pd.read_csv("C:\CODING\CSV Files\AirPassengers.csv", parse_dates=True, index_col="Month")
passengers = data["Passengers"]

# It performs Seasonal and Trend decomposition using LOESS (STL) on the passenger data with a seasonal period of 13. The result is stored in the result variable.
stl = STL(passengers, seasonal=13)  # You can experiment with different seasonal values
result = stl.fit()

# Extract the trend and seasonal components from the STL decomposition result.
trend = result.trend
seasonal = result.seasonal

# Forecast future values using the trend component
future_periods = 12  # Forecast 12 months into the future
forecast = trend[-1] + seasonal[-future_periods:]

# Calculate root mean squared error between the actual data for the last 12 periods and the forecasted values.
test = passengers[-future_periods:]
rmse = sqrt(mean_squared_error(test, forecast))
print(f"Test RMSE: {rmse:.2f}")

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(passengers, label="Actual Data")
plt.plot(forecast, label="Forecast")
plt.legend()
plt.title("Air Passenger Forecast with STL Decomposition")
plt.show()
