import numpy as np                                     # Used for numerical operations and handling arrays in a more efficient way.
import pandas as pd                                    # Reads the dataset from a CSV file into a Pandas DataFrame.
import matplotlib.pyplot as plt                        # Used for data visualization.
from sklearn.preprocessing import MinMaxScaler         # For normalizing data to a specific range.
from tensorflow.keras.models import Sequential         # Employed for creating a linear stack of layers.
from tensorflow.keras.layers import Dense              # Represents a fully connected layer in the neural network.
from tensorflow.keras.optimizers import Adam           # Used to optimize the neural network during training.
from tensorflow.keras.callbacks import EarlyStopping   # Employed as a callback to stop training when a monitored metric has stopped improving.
from sklearn.metrics import mean_squared_error         # Provides the mean_squared_error metric, which is used for evaluating the performance of the model by calculating the RMSE.
from math import sqrt                                  # Utilized to calculate the square root of the mean squared error, resulting in the RMSE.

# Load the Air Passenger dataset
data = pd.read_csv("C:\CODING\CSV Files\AirPassengers.csv")
passengers = data["Passengers"].values.astype(float)

# It normalizes the passenger data using Min-Max scaling, transforming the values to be in the range [0, 1].
scaler = MinMaxScaler(feature_range=(0, 1))
passengers = scaler.fit_transform(passengers.reshape(-1, 1))

# It splits the dataset into training and testing sets, with 67% of the data used for training.
train_size = int(len(passengers) * 0.67)
test_size = len(passengers) - train_size
train, test = passengers[0:train_size, :], passengers[train_size:len(passengers), :]

# It defines a function (create_dataset) to create time series data for training and testing, based on a specified look-back window. The function takes the dataset and returns input features (dataX) and corresponding output labels (dataY).
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# It creates a simple artificial neural network (ANN) using Keras with one hidden layer containing 8 neurons and an output layer with 1 neuron. The model is compiled with the mean squared error as the loss function and the Adam optimizer.
model = Sequential()
model.add(Dense(8, input_dim=look_back, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))

# It sets up early stopping to monitor the validation loss and stop training if there is no improvement after 10 epochs.
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

# Train the model
model.fit(trainX, trainY, epochs=100, batch_size=2, verbose=2, validation_data=(testX, testY), callbacks=[early_stopping])

# It uses the trained model to make predictions on both the training and testing datasets.
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Invert predictions to the original scale
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# Calculate root mean squared error
trainScore = sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
print(f"Train Score: {trainScore:.2f} RMSE")
testScore = sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print(f"Test Score: {testScore:.2f} RMSE")

# Plot the results
trainPredictPlot = np.empty_like(passengers)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

testPredictPlot = np.empty_like(passengers)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(passengers) - 1, :] = testPredict

plt.figure(figsize=(12, 6))
plt.plot(scaler.inverse_transform(passengers), label="Actual Passengers")
plt.plot(trainPredictPlot, label="Training Predictions")
plt.plot(testPredictPlot, label="Testing Predictions")
plt.legend()
plt.show()
