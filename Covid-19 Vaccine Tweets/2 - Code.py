import pandas as pd

# read data from CSV file
data = pd.read_csv('all_covid19_vaccine_tweets.csv')

# Alternatively, use Twitter API to collect data
import tweepy

# Define Twitter API credentials
consumer_key = 'your_consumer_key'
consumer_secret = 'your_consumer_secret'
access_token = 'your_access_token'
access_token_secret = 'your_access_token_secret'

# Authenticate with Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Initialize API object
api = tweepy.API(auth)

# Define search query and date range
query = 'covid vaccine'
since_date = '2022-01-01'
until_date = '2022-01-31'

# Collect tweets using search query and date range
tweets = tweepy.Cursor(api.search, q=query, since_id=since_date, until=until_date).items()

# Convert tweets to dataframe
data = pd.DataFrame(columns=['text', 'user_info'])

for tweet in tweets:
    text = tweet.text
    user_info = tweet.user._json
    data = data.append({'text': text, 'user_info': user_info}, ignore_index=True)

import string
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Define function to clean and preprocess text
def clean_text(text):
    # Remove special characters and punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    # Tokenize text into words
    words = word_tokenize(text.lower())
    # Remove stopwords
    words = [word for word in words if word not in stopwords.words('english')]
    # Lemmatize words
    lem = WordNetLemmatizer()
    words = [lem.lemmatize(word) for word in words]
    # Join words back into text
    text = ' '.join(words)
    return text

# Apply function to text data
data['clean_text'] = data['text'].apply(lambda x: clean_text(x))

import matplotlib.pyplot as plt
import seaborn as sns

# Define function to plot word frequency distribution
def plot_word_freq(text):
    # Tokenize text into words
    words = word_tokenize(text.lower())
    # Remove stopwords
    words = [word for word in words if word not in stopwords.words('english')]
    # Plot word frequency distribution
    freq_dist = nltk.FreqDist(words)
    freq_dist.plot(30)

# Plot word frequency distribution for cleaned text
plot_word_freq(data['clean_text'].str.cat(sep=' '))

# 3. Feature Extraction

# Convert the tweet text into numerical features using TF-IDF vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(cleaned_tweets['text'])

# Concatenate the numerical features with the user information and sentiment lexicons
from scipy.sparse import hstack

features = hstack((tfidf, cleaned_tweets[['user_followers_count', 'user_friends_count', 'user_verified']].values))

# Print the shape of the features matrix
print("Shape of features matrix: ", features.shape)

# 4. Model Selection and Training

# split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# initialize models
lr = LogisticRegression(random_state=42)
svm = LinearSVC(random_state=42)
nb = MultinomialNB()

# train models on training data
lr.fit(X_train, y_train)
svm.fit(X_train, y_train)
nb.fit(X_train, y_train)

# evaluate performance on testing data
lr_acc = lr.score(X_test, y_test)
svm_acc = svm.score(X_test, y_test)
nb_acc = nb.score(X_test, y_test)

# print accuracy scores for each model
print(f"Logistic Regression Accuracy: {lr_acc:.2f}")
print(f"Support Vector Machine Accuracy: {svm_acc:.2f}")
print(f"Naive Bayes Accuracy: {nb_acc:.2f}")
