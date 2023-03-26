# Section 1: Importing Required Libraries and Loading Data
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the data into a pandas dataframe
energy_df = pd.read_csv('https://raw.githubusercontent.com/ashishpatel26/Fuel/master/Building_Energy_Benchmarking.csv')

# Section 2: Data Cleaning and Preprocessing
# Check for missing values
print(energy_df.isnull().sum())

# Remove columns with more than 30% missing values
missing_values_cols = energy_df.columns[energy_df.isnull().sum() > 0.3*len(energy_df)].tolist()
energy_df.drop(missing_values_cols, axis=1, inplace=True)

# Fill missing values with column mean
energy_df.fillna(energy_df.mean(), inplace=True)

# Remove rows with missing values
energy_df.dropna(inplace=True)

# Section 3: Data Exploration and Visualization
# Correlation matrix
corr_matrix = energy_df.corr()

# Plot the correlation matrix using seaborn heatmap
sns.heatmap(corr_matrix, cmap='coolwarm')

# Distribution plot for Site EUI
sns.distplot(energy_df['Site EUI (kBtu/ft²)'])

# Distribution plot for Energy Star Score
sns.distplot(energy_df['ENERGY STAR Score'])

# Section 4: Model Building
# Split the data into training and testing sets
X = energy_df.drop('Site EUI (kBtu/ft²)', axis=1)
y = energy_df['Site EUI (kBtu/ft²)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regression model
rf_reg = RandomForestRegressor(random_state=42)
rf_reg.fit(X_train, y_train)

# Section 5: Model Evaluation
# Evaluate the model
y_pred = rf_reg.predict(X_test)

print('Model Performance')
print('MAE:', mean_absolute_error(y_test, y_pred))
print('MSE:', mean_squared_error(y_test, y_pred))
print('R-squared:', r2_score(y_test, y_pred))

# Section 6: Feature Importance
# Plot feature importance using seaborn barplot
feat_importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf_reg.feature_importances_})
feat_importance.sort_values(by='Importance', inplace=True, ascending=False)

sns.barplot(x='Importance', y='Feature', data=feat_importance)

