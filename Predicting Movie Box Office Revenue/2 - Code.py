# Step 1: Data collection and preparation
import pandas as pd

# Load the datasets
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
budgets = pd.read_csv('budgets.csv')

# Merge the datasets
data = pd.merge(movies, ratings, on='movie_id')
data = pd.merge(data, budgets, on='movie_id')

# Clean the data
data = data.drop_duplicates(subset=['movie_id', 'title'])
data = data.dropna()

# Step 2: Feature engineering
import datetime

# Extract release month and year
data['release_date'] = pd.to_datetime(data['release_date'])
data['release_month'] = data['release_date'].dt.month
data['release_year'] = data['release_date'].dt.year

# Create binary genres columns
genres = set()
for g in data['genres']:
    genres.update(g.split('|'))
for g in genres:
    data['genre_' + g] = data['genres'].apply(lambda x: 1 if g in x.split('|') else 0)

# Step 3: Model selection and training
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Split the dataset
X = data.drop(['movie_id', 'title', 'genres', 'rating', 'box_office'], axis=1)
y = data['box_office']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Model tuning and validation
from sklearn.model_selection import GridSearchCV, cross_val_score

# Tune the model using cross-validation
params = {'fit_intercept': [True, False], 'normalize': [True, False]}
grid = GridSearchCV(model, params, cv=5)
grid.fit(X_train, y_train)

# Evaluating the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

# Making predictions on new data
new_data = np.array([[6.5, 7.2, 8.9, 4.3, 5.6, 2.1]])
prediction = model.predict(new_data)
print('Prediction:', prediction)

# Saving the model
model.save('box_office_model.h5')

