
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# loading the json files in dataframes
new_york_tweets = pd.read_json("new_york.json", lines=True)
london_tweets = pd.read_json("london.json", lines=True)
paris_tweets = pd.read_json("paris.json", lines=True)

# getting a glance at what the data looks like
# print(len(new_york_tweets))
# print(new_york_tweets.columns)
# print(new_york_tweets.loc[12]["text"])

# will be classifying the tweet using language, hence will use a Naive Bayes Classifier
# creating a list of all the tweets
new_york_text = new_york_tweets['text'].tolist()
london_text = london_tweets['text'].tolist()
paris_text = paris_tweets['text'].tolist()

all_tweets = new_york_text + london_text + paris_text

# 0 represents new york, 1 represents london, and 2 represents paris in the labels list
labels = [0]* len(new_york_text) + [1] * len(london_text) + [2] * len(paris_text)

# splitting the data into a 80-20 train test split 
train_data, test_data, train_labels, test_labels = train_test_split(all_tweets, labels, test_size = 0.2, random_state = 1)

# checking to see that the split was done right
# print (len(train_data), len(test_data))

# transforming list of words into count vectors
counter = CountVectorizer()
counter.fit(train_data)
train_counts = counter.transform(train_data)
test_counts = counter.transform(test_data)

# checking to see that the data was transformed correctly
# print(train_data[3], train_counts[3])

# using the CountVectors to train the classifier
classifier = MultinomialNB()
classifier.fit(train_counts, train_labels)

# testing and evaluating the model
predictions = classifier.predict(test_counts)
# evaluations can be done using the accuracy_score function
# print(accuracy_score(test_labels, predictions))

# evaluations can also be conducted using a confusion matrix
# print (confusion_matrix(test_labels, predictions))

# looking at the matrix it is clear that the classifier had trouble telling the difference between new york and london tweets which is understandable, if time zones were taken into account it could be possible to minimize that issue

# the accuracy of this classifier is approximately 68%