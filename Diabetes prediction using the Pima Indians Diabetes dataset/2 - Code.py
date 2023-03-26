# Step 1: Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Step 2: Load the dataset
diabetes_data = pd.read_csv("diabetes.csv")
print(diabetes_data.head())

# Step 3: Data preprocessing
# Replace 0 values with NaN
diabetes_data[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = diabetes_data[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.nan)

# Drop rows with missing values
diabetes_data = diabetes_data.dropna()

# Split the dataset into features and target variable
X = diabetes_data.drop("Outcome", axis=1)
y = diabetes_data["Outcome"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Feature selection
# No feature selection performed in this code

# Step 5: Model selection
# Create a list of models to test
models = [("Logistic Regression", LogisticRegression()), ("Decision Tree", DecisionTreeClassifier()), ("SVM", SVC())]

# Test the models and select the best one
best_model = None
best_accuracy = 0

for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(name + " Accuracy: ", accuracy)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# Step 6: Model evaluation
# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)
print("Best model accuracy: ", accuracy_score(y_test, y_pred))
print("Best model precision: ", precision_score(y_test, y_pred))
print("Best model recall: ", recall_score(y_test, y_pred))
print("Best model F1 score: ", f1_score(y_test, y_pred))
print("Confusion matrix: \n", confusion_matrix(y_test, y_pred))

# Step 7: Hyperparameter tuning
# Hyperparameter tuning not performed in this code

# Step 8: Final model training
# Train the best model on the entire dataset
X = scaler.fit_transform(X)
best_model.fit(X, y)

# Step 9: Model deployment

# Load the new data to make predictions on
new_data = pd.read_csv("new_data.csv")

# Preprocess the new data using the same steps as the training data
new_data[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = new_data[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.nan)
new_data = new_data.dropna()
X_new = new_data.drop("Outcome", axis=1)
X_new = scaler.transform(X_new)

# Make predictions using the trained model
y_new = best_model.predict(X_new)

# Print the predictions
print("Predictions for new data: ", y_new)

