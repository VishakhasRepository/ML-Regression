# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("loan_data.csv")

# Check for missing values
print(data.isnull().sum())

# Explore the distribution of the target variable
print(data["loan_status"].value_counts())

# Visualize the distribution of each feature
sns.displot(data, x="loan_amnt", hue="loan_status", multiple="stack")
sns.displot(data, x="int_rate", hue="loan_status", multiple="stack")
sns.displot(data, x="annual_inc", hue="loan_status", multiple="stack")
sns.displot(data, x="dti", hue="loan_status", multiple="stack")
sns.displot(data, x="revol_bal", hue="loan_status", multiple="stack")
sns.displot(data, x="total_pymnt", hue="loan_status", multiple="stack")

# Visualize the relationship between features and the target variable
sns.catplot(data=data, x="grade", y="loan_amnt", hue="loan_status", kind="box")
sns.catplot(data=data, x="term", y="loan_amnt", hue="loan_status", kind="box")
sns.catplot(data=data, x="home_ownership", y="loan_amnt", hue="loan_status", kind="box")
sns.catplot(data=data, x="purpose", y="loan_amnt", hue="loan_status", kind="box")

# Section 2: Data Preprocessing (continued)

# Convert categorical variables to dummy variables
dummy_cols = ['term', 'grade', 'sub_grade', 'emp_length', 'home_ownership', 'verification_status', 
              'purpose', 'addr_state']
df = pd.get_dummies(df, columns=dummy_cols, drop_first=True)

# Split the data into training and testing sets
X = df.drop('loan_status', axis=1)
y = df['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Section 3: Model Building and Training

# Define the models to be trained
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC(),
    'Gradient Boosting': GradientBoostingClassifier()
}

# Train the models and calculate their accuracy scores
model_scores = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    model_scores[name] = score
    print(f'{name}: {score:.3f}')

# Plot the accuracy scores of the models
plt.bar(range(len(model_scores)), list(model_scores.values()), align='center')
plt.xticks(range(len(model_scores)), list(model_scores.keys()), rotation=45)
plt.ylim([0.7, 1])
plt.title('Model Accuracy Scores')
plt.show()

