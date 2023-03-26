# Section 1: Data Loading and Exploration
import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv('transportation_weather_data.csv')

# Explore the dataset
print(data.head())
print(data.info())
print(data.describe())

# Section 2: Data Cleaning and Preparation
# Check for missing values
print(data.isnull().sum())

# Drop rows with missing values
data.dropna(inplace=True)

# Convert date column to datetime object
data['date'] = pd.to_datetime(data['date'])

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

X = data.drop(['subway', 'bus', 'train'], axis=1)
y_subway = data['subway']
y_bus = data['bus']
y_train = data['train']

X_train, X_test, y_subway_train, y_subway_test = train_test_split(X, y_subway, test_size=0.2, random_state=42)
X_train, X_test, y_bus_train, y_bus_test = train_test_split(X, y_bus, test_size=0.2, random_state=42)
X_train, X_test, y_train_train, y_train_test = train_test_split(X, y_train, test_size=0.2, random_state=42)

# Section 3: Feature Engineering
# Add lagged weather features
X_train['temp_lag1'] = X_train['temp'].shift(1)
X_train['temp_lag2'] = X_train['temp'].shift(2)
X_train['precip_lag1'] = X_train['precip'].shift(1)
X_train['precip_lag2'] = X_train['precip'].shift(2)

X_test['temp_lag1'] = X_test['temp'].shift(1)
X_test['temp_lag2'] = X_test['temp'].shift(2)
X_test['precip_lag1'] = X_test['precip'].shift(1)
X_test['precip_lag2'] = X_test['precip'].shift(2)

# Section 4: Machine Learning Modeling
# Build a random forest model
from sklearn.ensemble import RandomForestRegressor

rf_subway = RandomForestRegressor(n_estimators=100, random_state=42)
rf_subway.fit(X_train, y_subway_train)

rf_bus = RandomForestRegressor(n_estimators=100, random_state=42)
rf_bus.fit(X_train, y_bus_train)

rf_train = RandomForestRegressor(n_estimators=100, random_state=42)
rf_train.fit(X_train, y_train_train)

# Section 5: Model Evaluation
# Evaluate the model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_subway_pred = rf_subway.predict(X_test)
y_bus_pred = rf_bus.predict(X_test)
y_train_pred = rf_train.predict(X_test)

print('Subway Model Performance')
print('MAE:', mean_absolute_error(y_subway_test, y_subway_pred))
print('MSE:', mean_squared_error(y_subway_test, y_subway_pred))
print('R-squared:', r2_score(y_subway_test, y_subway_pred))

print('Bus Model Performance')
print('MAE:', mean_absolute_error(y_bus_test, y_bus_pred))
print('MSE:', mean_squared_error(y_bus_test, y_bus_pred))
print('R-squared:', r2_score(y_bus_test, y_bus_pred))
