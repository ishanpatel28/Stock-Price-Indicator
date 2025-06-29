import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
# Fetch historical stock data
ticker = "AAPL" # you can change this to any stock symbol
data = yf.download(ticker, start="2018-01-01", end="2024-12-31")
data['Tomorrow Close'] = data['Close'].shift(-1)

# Show the first few rows
print(data.head())

# Select the 'Close' price column as the feature
X = data[['Close']]

# Create a target variable - predict next day's Close Price
y=data['Close'].shift(-1) # Shift Close prices up by one day

# Drop the last row with NaN target becuase of the shift
X = X[:-1]
y = y[:-1]

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Drop the last row with NaN target value
data.dropna(inplace=True)

# Features and target
X = data[['Close']]  # today’s close price as feature
y = data['Tomorrow Close']  # next day close price as target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Initialize and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))   # 1. Create a new figure with width 12 inches and height 6 inches
plt.plot(y_test.index, y_test, label='Actual Close Price')  # 2. Plot the actual closing prices from the test set over time
plt.plot(y_test.index, y_pred, label='Predicted Close Price') # 3. Plot the predicted closing prices from the model on the same time axis
plt.xlabel('Date')  # 4. Label the x-axis as "Date"
plt.ylabel('Price') # 5. Label the y-axis as "Price"
plt.title('Actual vs Predicted Close Price')  # 6. Add a title to the plot
plt.legend()  # 7. Show the legend to differentiate actual and predicted lines
plt.show()  # 8. Display the plot window

