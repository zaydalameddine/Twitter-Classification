
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# importing the all_tweets json file
all_tweets = pd.read_json("random_tweets.json", lines=True)

# printing out certain things to understand the file better
# print(len(all_tweets))
# print(all_tweets.columns)
# print(all_tweets.loc[0]['text'])
# print(all_tweets.loc[0]['user'])

# need to define what marks a tweet as viral or not and create labels for the datapoints
# this column will have a 1 if viral and 0 if not
# features to look at: number of retweets
all_tweets['is_viral'] = np.where(all_tweets['retweet_count'] > all_tweets['retweet_count'].median(), 1, 0)
# print(all_tweets['is_viral'].value_counts())

# choosing which features of the data could make a tweet viral or not
all_tweets['tweet_length'] = all_tweets.apply(lambda tweet: len(tweet['text']), axis=1)
all_tweets['followers_count'] = all_tweets.apply(lambda tweet: tweet['user']['followers_count'], axis=1)
all_tweets['friends_count'] = all_tweets.apply(lambda tweet: tweet['user']['friends_count'], axis=1)

# normalizing the data chosen as imporant
# first will specify the data that I will be working with
labels = all_tweets['is_viral']
data = all_tweets[['tweet_length', 'followers_count', 'friends_count']]

# data is normalized
scaled_data = scale(data, axis = 0)
#print(scaled_data[0])

# splitting the data into a 80-20 train test split 
train_data, test_data, train_labels, test_labels = train_test_split(scaled_data, labels, test_size = 0.2, random_state = 1)

# creating and training the classifier
classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(train_data, train_labels)

# testing the accuracy of the model
score = classifier.score(test_data, test_labels)
# print(score)

# accuracy is suboptimal, testing different k values to optimize classifier

scores = []
best_k = 0

for k in range(1, 200):
    classifier = KNeighborsClassifier(n_neighbors = k)
    classifier.fit(train_data, train_labels)
    score = classifier.score(test_data, test_labels)
    scores.append(score)
    best_k = max(best_k, score)

# showing the plot
plt.plot(range(1,200), scores)
plt.show()

print(best_k)

# the best k value for this classifier is found to be approximately 62%