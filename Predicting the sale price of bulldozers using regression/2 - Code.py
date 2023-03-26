# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
train_data = pd.read_csv('Train.csv', low_memory=False, parse_dates=['saledate'])
test_data = pd.read_csv('Test.csv', low_memory=False, parse_dates=['saledate'])

# Print the shape of the data
print(f'The training data has {train_data.shape[0]} rows and {train_data.shape[1]} columns.')
print(f'The test data has {test_data.shape[0]} rows and {test_data.shape[1]} columns.')

# Handle missing values
train_data.dropna(subset=['SalePrice'], inplace=True)

# Convert categorical variables to numerical variables
for column in train_data.select_dtypes(include=['object']):
    train_data[column] = train_data[column].astype('category').cat.codes

# Scale the features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_data.iloc[:, :-1] = scaler.fit_transform(train_data.iloc[:, :-1])

# EDA
sns.histplot(train_data['SalePrice'], kde=False)
plt.title('Histogram of Sale Price')
plt.show()

sns.scatterplot(x='YearMade', y='SalePrice', data=train_data)
plt.title('Year Made vs Sale Price')
plt.show()

# Create new features
train_data['Age'] = train_data['saledate'].dt.year - train_data['YearMade']
train_data['Year'] = train_data['saledate'].dt.year
train_data['Month'] = train_data['saledate'].dt.month

# Feature selection
corr = train_data.corr()
sns.heatmap(corr, annot=True)
plt.title('Correlation Matrix')
plt.show()

features = corr['SalePrice'][corr['SalePrice'] > 0.5].index
train_data = train_data[features]

# Split the data into train and validation sets
from sklearn.model_selection import train_test_split

X = train_data.drop('SalePrice', axis=1)
y = train_data['SalePrice']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
from xgboost import XGBRegressor

model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate the model
from sklearn.metrics import mean_absolute_error, mean_squared_error

y_pred = model.predict(X_valid)

mae = mean_absolute_error(y_valid, y_pred)
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))

print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')

# Perform error analysis
residuals = y_valid - y_pred
sns.scatterplot(x=y_valid, y=residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Sale Price')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# Evaluate the model on the test set
# Evaluate the model on the test set
test_data['Age'] = test_data['saledate'].dt.year - test_data['YearMade']
test_data['Year'] = test_data['saledate'].dt.year
test_data['Month'] = test_data['saledate'].dt.month

for column in test_data.select_dtypes(include=['object']):
    test_data[column] = test_data[column].astype('category').cat.codes

test_data.iloc[:, :-1] = scaler.transform(test_data.iloc[:, :-1])
test_data = test_data[X_train.columns]

test_preds = model.predict(test_data)

submission_df = pd.DataFrame({
    'SalesID': test_data['SalesID'],
    'SalePrice': test_preds
})

submission_df.to_csv('submission.csv', index=False)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Load the data
test_data = pd.read_csv('Test.csv', low_memory=False, parse_dates=['saledate'])

# Preprocess the data
test_data['Age'] = test_data['saledate'].dt.year - test_data['YearMade']
test_data['Year'] = test_data['saledate'].dt.year
test_data['Month'] = test_data['saledate'].dt.month

for column in test_data.select_dtypes(include=['object']):
    test_data[column] = test_data[column].astype('category').cat.codes

scaler = StandardScaler()
train_data = pd.read_csv('Train.csv', low_memory=False, parse_dates=['saledate'])
train_data.dropna(subset=['SalePrice'], inplace=True)
for column in train_data.select_dtypes(include=['object']):
    train_data[column] = train_data[column].astype('category').cat.codes
train_data.iloc[:, :-1] = scaler.fit_transform(train_data.iloc[:, :-1])
test_data.iloc[:, :-1] = scaler.transform(test_data.iloc[:, :-1])
features = train_data.corr()['SalePrice'][train_data.corr()['SalePrice'] > 0.5].index
test_data = test_data[features]

# Load the model
model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=-1)
model.fit(train_data[features].drop('SalePrice', axis=1), train_data['SalePrice'])

# Make predictions on the test set
test_preds = model.predict(test_data.drop('SalesID', axis=1))

# Save the predictions to a CSV file
submission_df = pd.DataFrame({
    'SalesID': test_data['SalesID'],
    'SalePrice': test_preds
})
submission_df.to_csv('submission.csv', index=False)
