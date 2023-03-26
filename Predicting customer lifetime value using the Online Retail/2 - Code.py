import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('OnlineRetail.csv')

# Remove rows with missing values
df = df.dropna()

# Remove duplicates
df = df.drop_duplicates()

# Convert the InvoiceDate column to a datetime object
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Create a new column for frequency of purchases
df['Frequency'] = df.groupby('CustomerID')['TotalSpend'].count()

# Create a new column for average spend per transaction
df['AvgSpend'] = df['TotalSpend'] / df['Frequency']

# Create a new column for customer lifetime value
df['CLV'] = (df['AvgSpend'] * df['Frequency']) / df['Recency']

# Select relevant columns for training the model
X = df[['TotalSpend', 'Frequency', 'Recency']]
y = df['CLV']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Train a decision tree regression model
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

# Train a random forest regression model
rf_model = RandomForestRegressor


# Remove rows with negative quantity or price
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

# Create a new column for total spend
df['TotalSpend'] = df['Quantity'] * df['UnitPrice']

# Create a new column for recency of purchases
max_date = df['InvoiceDate'].max()
df['Recency'] = (max_date - df['InvoiceDate']).dt.days

# Select relevant columns
df = df[['CustomerID', 'TotalSpend', 'Recency']]

# Group the data by customer ID and aggregate the TotalSpend and Recency columns
df = df.groupby('CustomerID').agg({'TotalSpend': 'sum', 'Recency': 'min'})

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Create a new column for frequency of purchases
df['Frequency'] = df.groupby('CustomerID')['TotalSpend'].count()

# Create a new column for average spend per transaction
df['AvgSpend'] = df['TotalSpend'] / df['Frequency']

# Create a new column for customer lifetime value
df['CLV'] = (df['AvgSpend'] * df['Frequency']) / df['Recency']

# Select relevant columns for training the model
X = df[['TotalSpend', 'Frequency', 'Recency']]
y = df['CLV']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Train a decision tree regression model
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

# Train a random forest regression model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the models on the validation set
lr_pred = lr_model.predict(X_val)
lr_rmse = np.sqrt(mean_squared_error(y_val, lr_pred))
lr_r2 = r2_score(y_val, lr_pred)

dt_pred = dt_model.predict(X_val)
dt_rmse = np.sqrt(mean_squared_error(y_val, dt_pred))
dt_r2 = r2_score(y_val, dt_pred)

rf_pred = rf_model.predict(X_val)
rf_rmse = np.sqrt(mean_squared_error(y_val, rf_pred))
rf_r2 = r2_score(y_val, rf_pred)

# Print the evaluation metrics for each model
print('Linear Regression:')
print('RMSE: {:.2f}'.format(lr_rmse))
print('R^2 Score: {:.2f}'.format(lr_r2))
print()

print('Decision Tree Regression:')
print('RMSE: {:.2f}'.format(dt_rmse))
print('R^2 Score: {:.2f}'.format(dt_r2))
print()

print('Random Forest Regression:')
print('RMSE: {:.2f}'.format(rf_rmse))
print('R^2 Score: {:.2f}'.format(rf_r2))
print()

from sklearn.model_selection import GridSearchCV

# Define the hyperparameter space for the random forest model
param_grid = {
    'n_estimators': [100, 500, 1000],
    'max_depth': [None, 10, 20, 30]
}

# Perform grid search to find the best hyperparameters
grid_search = GridSearchCV(rf_model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters found by grid search
print('Best Hyperparameters:', grid_search.best_params_)
print()

# Train the final random forest model with the best hyperparameters
final_model = RandomForestRegressor(n_estimators=grid_search.best_params_['n_estimators'], 
                                     max_depth=grid_search.best_params_['max_depth'], 
                                     random_state=42)
final_model.fit(X_train, y_train)

# Save the final model to a file
import pickle
with open('final_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)
