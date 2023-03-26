# Importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Loading the dataset
data = pd.read_csv('train.csv')

#Section 2: Data Exploration and Visualization
# Exploring the dataset
print(data.head())
print(data.info())
print(data.describe())

# Visualizing the dataset
plt.figure(figsize=(12,8))
sns.distplot(data['Rainfall'], bins=100, kde=True)
plt.title('Rainfall Distribution')
plt.xlabel('Rainfall')
plt.ylabel('Density')
plt.show()

#Section 3: Data Preprocessing and Feature Engineering
# Handling missing values
data = data.drop(['Next_Tmin'], axis=1)
data = data.dropna()

# Converting date column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')

# Feature engineering
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day

# Dropping unnecessary columns
data = data.drop(['Date', 'Station'], axis=1)

# Splitting the dataset into training and testing sets
X = data.drop(['Rainfall'], axis=1)
y = data['Rainfall']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Section 4: Building a Linear Regression Model

# Training a Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Evaluating the Linear Regression model
lr_y_pred = lr_model.predict(X_test)
lr_mse = mean_squared_error(y_test, lr_y_pred)
lr_r2 = r2_score(y_test, lr_y_pred)

print('Linear Regression Mean Squared Error:', lr_mse)
print('Linear Regression R^2 Score:', lr_r2)

#Section 5: Building a Random Forest Regression Model

# Training a Random Forest Regression model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluating the Random Forest Regression model
rf_y_pred = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_y_pred)
rf_r2 = r2_score(y_test, rf_y_pred)

print('Random Forest Regression Mean Squared Error:', rf_mse)
print('Random Forest Regression R^2 Score:', rf_r2)

#Section 6: Model Tuning and Evaluation
# Hyperparameter tuning of the Random Forest Regression model
# Section 6: Model Tuning and Evaluation

# Hyperparameter tuning of the Random Forest Regression model
param_grid = {'n_estimators': [50, 100, 150, 200],
              'max_features': ['auto', 'sqrt'],
              'max_depth': [5, 10, 15, 20],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4]}
rf_grid = RandomForestRegressor(random_state=42)
rf_grid_search = GridSearchCV(estimator=rf_grid, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
rf_grid_search.fit(X_train, y_train)

# Best hyperparameters
print("Best hyperparameters:", rf_grid_search.best_params_)

# Evaluation of the model with best hyperparameters
y_train_pred = rf_grid_search.predict(X_train)
y_test_pred = rf_grid_search.predict(X_test)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
print("Train RMSE: {:.4f}".format(train_rmse))
print("Test RMSE: {:.4f}".format(test_rmse))
print("Train R2 Score: {:.4f}".format(train_r2))
print("Test R2 Score: {:.4f}".format(test_r2))

# Plotting the actual vs predicted values
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_test_pred)
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.title("Actual vs Predicted values (Test set)")
plt.show()
