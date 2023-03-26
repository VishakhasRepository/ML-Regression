import pandas as pd

# Load the Flight Delay Prediction dataset
df = pd.read_csv('flight_delay_prediction.csv')

# Remove any invalid or missing values
df.dropna(inplace=True)

# Convert categorical variables to numerical variables
df['origin_airport'] = pd.factorize(df['origin_airport'])[0]
df['destination_airport'] = pd.factorize(df['destination_airport'])[0]

# Split the dataset into features and target variable
X = df.drop('delay', axis=1)
y = df['delay']

import seaborn as sns
import matplotlib.pyplot as plt

# Plot a heatmap of the correlation matrix
sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
plt.show()

# Plot a histogram of the target variable
sns.histplot(y, kde=True)
plt.show()

from sklearn.preprocessing import StandardScaler

# Extract additional features such as weather data and flight duration

# Normalize and standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Select a machine learning model and tune its hyperparameters
model = LogisticRegression()
parameters = {'C': [0.1, 1, 10]}
grid = GridSearchCV(model, parameters, cv=5)
grid.fit(X_train, y_train)

# Evaluate the performance of the model
print('Accuracy:', grid.score(X_test, y_test))
