import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load the dataset
df = pd.read_csv('Product_demand_forecasting.csv')

# Data cleaning
df.dropna(inplace=True)

# Feature engineering
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Week'] = df['Date'].dt.week
df['Day'] = df['Date'].dt.day

# Encoding categorical variables
le = LabelEncoder()
df['Product_Code'] = le.fit_transform(df['Product_Code'])
df['Warehouse'] = le.fit_transform(df['Warehouse'])
df['Product_Category'] = le.fit_transform(df['Product_Category'])

# Scaling numeric features
scaler = MinMaxScaler()
df[['Order_Demand']] = scaler.fit_transform(df[['Order_Demand']])

# Create lag features
df['Order_Demand_Lag1'] = df['Order_Demand'].shift(1)
df['Order_Demand_Lag7'] = df['Order_Demand'].shift(7)
df['Order_Demand_Lag30'] = df['Order_Demand'].shift(30)

# Create rolling mean features
df['Order_Demand_Rolling7'] = df['Order_Demand'].rolling(7).mean()
df['Order_Demand_Rolling30'] = df['Order_Demand'].rolling(30).mean()

# Create seasonality features
df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
df['Week_Sin'] = np.sin(2 * np.pi * df['Week'] / 52)
df['Week_Cos'] = np.cos(2 * np.pi * df['Week'] / 52)

# Drop unnecessary columns
df.drop(['Date', 'Year', 'Month', 'Week', 'Day'], axis=1, inplace=True)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Split the data into training and testing sets
X = df.drop('Order_Demand', axis=1)
y = df['Order_Demand']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the random forest regressor
rf = RandomForestRegressor()

# Train the model on the training set
rf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = rf.predict(X_test)

# Evaluate the model using mean absolute error (MAE), mean squared error (MSE), and root mean squared error (RMSE)
print('MAE:', mean_absolute_error(y_test, y_pred))
print('MSE:', mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))

#Hyperparameter tuning
param_grid = {
'n_estimators': [100, 200, 300, 400, 500],
'max_depth': [5, 10, 15, 20],
'min_samples_split': [2, 5, 10, 15],
'min_samples_leaf': [1, 2, 4],
'max_features': ['sqrt', 'log2']
}

rf = RandomForestRegressor(random_state=42)

rf_grid = RandomizedSearchCV(estimator=rf, param_distributions=param_grid,
n_iter=50, cv=5, random_state=42, n_jobs=-1)

rf_grid.fit(X_train, y_train)

#Best parameters
best_params = rf_grid.best_params_
print("Best parameters: ", best_params)

#Model evaluation
model = RandomForestRegressor(**best_params, random_state=42)

model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

print("Training RMSE:", train_rmse)
print("Validation RMSE:", val_rmse)