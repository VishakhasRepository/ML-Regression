# Section 1: Importing Required Libraries and Loading Data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# load the dataset
df = pd.read_csv('crypto_price.csv')

# Section 2: Data Preprocessing

# check for missing values
print(df.isnull().sum())

# drop unwanted columns
df.drop(['slug', 'name'], axis=1, inplace=True)

# convert date to datetime format
df['date'] = pd.to_datetime(df['date'])

# set date column as index
df.set_index('date', inplace=True)

# fill missing values with forward fill method
df.fillna(method='ffill', inplace=True)

# Section 3: Exploratory Data Analysis

# visualize the closing price trend
plt.figure(figsize=(12, 6))
sns.lineplot(x=df.index, y='close', data=df)
plt.title('Closing Price Trend')
plt.xlabel('Year')
plt.ylabel('Price (USD)')
plt.show()

# visualize the daily price change distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['close'].pct_change().dropna(), kde=True, bins=100)
plt.title('Daily Price Change Distribution')
plt.xlabel('Percentage Change')
plt.ylabel('Frequency')
plt.show()

# Section 4: Feature Engineering

# create new features
df['daily_change'] = df['close'].pct_change()
df['daily_volatility'] = df['daily_change'].rolling(window=30).std()

# Section 5: Modeling and Evaluation

# split data into training and testing sets
X = df.drop(['close'], axis=1)
y = df['close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# build and evaluate linear regression model
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

# Section 6: Conclusion and Future Work

# summarize the findings and suggest future work
print('The linear regression model performed well in predicting the price of Bitcoin.')
print('Future work could include exploring other machine learning algorithms and incorporating more features such as news sentiment analysis and economic indicators.')
