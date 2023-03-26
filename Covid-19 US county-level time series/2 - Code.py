# SECTION 1: Import necessary libraries and load data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv('covid_data.csv')

# SECTION 2: Data preprocessing
# Remove unnecessary columns
data = data.drop(['FIPS', 'Admin2', 'Province_State', 'Country_Region', 'Last_Update'], axis=1)
# Fill missing values with 0
data = data.fillna(0)

# SECTION 3: Feature engineering
# Calculate daily cases and deaths
data['Daily_Cases'] = data.groupby(['Combined_Key'])['Confirmed'].diff().fillna(0)
data['Daily_Deaths'] = data.groupby(['Combined_Key'])['Deaths'].diff().fillna(0)
# Drop first row for each county since daily cases and deaths are 0
data = data[data['Daily_Cases'] != 0]

# SECTION 4: Model training
X = data.drop(['Combined_Key', 'Confirmed', 'Deaths', 'Daily_Cases', 'Daily_Deaths'], axis=1)
y_cases = data['Daily_Cases']
y_deaths = data['Daily_Deaths']

X_train_cases, X_test_cases, y_train_cases, y_test_cases = train_test_split(X, y_cases, test_size=0.2, random_state=42)
X_train_deaths, X_test_deaths, y_train_deaths, y_test_deaths = train_test_split(X, y_deaths, test_size=0.2, random_state=42)

# Train models using linear regression
from sklearn.linear_model import LinearRegression

model_cases = LinearRegression()
model_cases.fit(X_train_cases, y_train_cases)

model_deaths = LinearRegression()
model_deaths.fit(X_train_deaths, y_train_deaths)

# SECTION 5: Model evaluation
from sklearn.metrics import mean_squared_error

# Evaluate models on test set
y_pred_cases = model_cases.predict(X_test_cases)
mse_cases = mean_squared_error(y_test_cases, y_pred_cases)

y_pred_deaths = model_deaths.predict(X_test_deaths)
mse_deaths = mean_squared_error(y_test_deaths, y_pred_deaths)

print("MSE for daily cases:", mse_cases)
print("MSE for daily deaths:", mse_deaths)

# SECTION 6: Model deployment
# Use models to make predictions for future dates
future_dates = pd.date_range(start='2021-03-01', end='2021-03-31')
future_data = pd.DataFrame({'Date': future_dates})
future_data['Combined_Key'] = 'Unknown'

future_X = future_data.drop(['Combined_Key', 'Date'], axis=1)

future_cases = model_cases.predict(future_X)
future_deaths = model_deaths.predict(future_X)

future_data['Daily_Cases'] = future_cases
future_data['Daily_Deaths'] = future_deaths

# Visualize predictions
plt.plot(future_dates, future_data['Daily_Cases'], label='Predicted Daily Cases')
plt.plot(future_dates, future_data['Daily_Deaths'], label='Predicted Daily Deaths')
plt.xlabel('Date')
plt.ylabel('Number of cases/deaths')
plt.title('Covid-19 US county-level time series predictions')
plt.legend()
plt.show()
