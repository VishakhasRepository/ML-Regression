# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('Telco-Customer-Churn.csv')

# Get information about the dataset
df.info()

# Check for missing values
df.isnull().sum()

# Data cleaning and preprocessing
df.drop(['customerID'], axis=1, inplace=True)

# Convert categorical variables to dummy variables
cat_vars = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
df = pd.get_dummies(df, columns=cat_vars, drop_first=True)

# Plot distribution of target variable
sns.countplot(x='Churn', data=df)

# Separate features and target variable
X = df.drop(['Churn_Yes'], axis=1)
y = df['Churn_Yes']

# Feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Feature selection
from sklearn.feature_selection import SelectKBest, chi2
selector = SelectKBest(chi2, k=20)
X = selector.fit_transform(X, y)

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Import the classifier and fit the model on the training data
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Evaluate the model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print("Accuracy Score: ", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
