# Section 1: Import libraries and load dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# Section 2: Data preprocessing and feature engineering
def preprocess_data(df, is_train=True):
    # Fill missing values with 0
    df.fillna(0, inplace=True)
    
    # Convert date to datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Create new features from date
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['WeekOfYear'] = df['Date'].dt.weekofyear
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    
    # If training data, create new label 'FatalitiesRate'
    if is_train:
        df['FatalitiesRate'] = df['Fatalities'] / df['ConfirmedCases']
    
    # Drop unnecessary columns
    df.drop(['Id', 'Province_State', 'Country_Region', 'Date'], axis=1, inplace=True)
    
    return df

# Preprocess training and test data
df_train = preprocess_data(df_train)
df_test = preprocess_data(df_test, False)

# Split training data into X and y
X = df_train.drop(['ConfirmedCases', 'Fatalities', 'FatalitiesRate'], axis=1)
y_cases = df_train['ConfirmedCases']
y_fatalities = df_train['Fatalities']

# Section 3: Train model for predicting confirmed cases
X_train_cases, X_test_cases, y_train_cases, y_test_cases = train_test_split(X, y_cases, test_size=0.2, random_state=42)

model_cases = XGBRegressor(n_estimators=1000, learning_rate=0.05, random_state=42)
model_cases.fit(X_train_cases, y_train_cases)

# Evaluate model for predicting confirmed cases
y_pred_cases = model_cases.predict(X_test_cases)
mse_cases = mean_squared_error(y_test_cases, y_pred_cases)
print('MSE for confirmed cases:', mse_cases)

# Section 4: Train model for predicting fatalities
X_train_fatalities, X_test_fatalities, y_train_fatalities, y_test_fatalities = train_test_split(X, y_fatalities, test_size=0.2, random_state=42)

model_fatalities = XGBRegressor(n_estimators=1000, learning_rate=0.05, random_state=42)
model_fatalities.fit(X_train_fatalities, y_train_fatalities)

# Evaluate model for predicting fatalities
y_pred_fatalities = model_fatalities.predict(X_test_fatalities)
mse_fatalities = mean_squared_error(y_test_fatalities, y_pred_fatalities)
print('MSE for fatalities:', mse_fatalities)

# Section 5: Make predictions for test data
y_pred_cases_test = model_cases.predict(df_test)
y_pred_fatalities_test = model_fatalities.predict(df_test)

# Section 6: Save predictions to submission file
submission = pd.DataFrame({
    'ForecastId': df_test['ForecastId'],
    'ConfirmedCases': y_pred_cases_test,
    'Fatalities': y_pred_fatalities_test
})
submission.to_csv('submission.csv', index=False)

print('Predictions saved to submission.csv')
