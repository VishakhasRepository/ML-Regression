# Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Loading the Data
data = pd.read_csv('car_data.csv')

# Removing Duplicates
data.drop_duplicates(keep='first', inplace=True)

# Handling Missing Values
data.dropna(subset=['Make', 'Model', 'Year', 'Engine Fuel Type', 'HP', 'Cylinders', 'Transmission', 'Drivetrain', 'Vehicle Style'], inplace=True)
data['Engine HP'] = data['Engine HP'].fillna(data['Engine HP'].median())
data['Engine Cylinders'] = data['Engine Cylinders'].fillna(data['Engine Cylinders'].median())
data.drop(['Market Category', 'Number of Doors', 'Vehicle Size'], axis=1, inplace=True)

# Encoding Categorical Variables
categorical_columns = ['Make', 'Model', 'Engine Fuel Type', 'Transmission', 'Drivetrain', 'Vehicle Style']
for column in categorical_columns:
    data[column] = data[column].astype('category')
    data[column] = data[column].cat.codes

# Data Visualization
sns.pairplot(data)
plt.show()

# Correlation Matrix
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Histogram of Prices
sns.histplot(data['MSRP'], kde=True)
plt.show()

# Splitting the Data into Training and Testing Sets
from sklearn.model_selection import train_test_split
X = data.drop(['MSRP'], axis=1)
y = data['MSRP']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scaling the Data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training the Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Testing the Model
from sklearn.metrics import r2_score
y_pred = regressor.predict(X_test)
print('R2 Score:', r2_score(y_test, y_pred))

# Feature Selection
from sklearn.feature_selection import SelectKBest, f_regression
selector = SelectKBest(f_regression, k=5)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Tuning Hyperparameters
from sklearn.model_selection import GridSearchCV
params = {'normalize': [True, False], 'copy_X': [True, False]}
grid_search = GridSearchCV(regressor, param_grid=params, cv=5)
grid_search.fit(X_train_selected, y_train)
print('Best Parameters:', grid_search.best_params_)
regressor_best = grid_search.best_estimator_

# Testing the Improved Model
y_pred_best = regressor_best.predict(X_test_selected)
print('R2 Score (Improved Model):', r2_score(y_test, y_pred_best))

# Define function to calculate root mean squared logarithmic error
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

# Evaluate model performance on test set
test_predictions = model.predict(X_test)
mae = mean_absolute_error(Y_test, test_predictions)
mse = mean_squared_error(Y_test, test_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, test_predictions)
rmsle = rmsle(Y_test, test_predictions)

# Print evaluation metrics
print(f"Mean Absolute Error: {mae:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R^2 Score: {r2:.4f}")
print(f"Root Mean Squared Logarithmic Error: {rmsle:.4f}")


# Saving the Trained Model
import joblib
joblib.dump(regressor_best, 'car_price_predictor.pkl')

