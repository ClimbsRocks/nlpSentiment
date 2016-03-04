from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import RandomizedSearchCV
import nltk
import random
import scipy
import numpy as np
import math
import loadAndProcessData
from sentimentCorpora import nltkMovieReviews

allTweets, allTweetsSentiment = loadAndProcessData.loadDataset('training.1600000.processed.noemoticon.csv')

allTweets = loadAndProcessData.removeStopWords(allTweets)

numWordsToUse = 3000


# unfortunately, the standard nltk movie reviews corpus isn't going to be much use for us today, as it only classifies into positive/negative, and does not have a neutral category
sparseFeatures, reviewsSentiment = nltkMovieReviews.getFeatures(numWordsToUse)

# one easy approach we can take is to simply train on the enormous number of tagged training documents provided in the training dataset
# this has the obvious advantage of being directly applicable to the test data we're working with, as they're both corpora of twitter messages. 
sparseFeatures, reviewsSentiment = stsTwitterMessages.getFeatures(numWordsToUse, allTweets, allTweetsSentiment)

# split our data into training and test datasets
trainReviews, testReviews, trainSentiment, testSentiment = train_test_split(
    sparseFeatures, reviewsSentiment, test_size=0.33, random_state=8)


# train a single random forest on our training data, test on our testing data
classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(trainReviews, trainSentiment)
print classifier.score(testReviews, testSentiment)

# optimize the hyperparameters for our random forest using RandomizedSearchCV
sqrtNum = int( math.log( math.sqrt(numWordsToUse) ) )
# print sqrtNum
parametersToTry = {
    # 'max_features': scipy.stats.randint(1,numWordsToUse/7),
    'min_samples_leaf': scipy.stats.randint(1,30),
    'min_samples_split': scipy.stats.randint(2,30),
    'bootstrap': [True,False]
}
# run on all cores, fail gracefully if a combination of hyperparameters fails to converge, try 100 different combinations of hyperparameters, train on all the training data when finished, and use a third of the dataset for cross-validation while training
searchCV = RandomizedSearchCV(classifier, parametersToTry, n_jobs=-1, error_score=0, n_iter=100, refit=True, cv=3)
# best results out of 10k training runs:
# {'max_features': 51, 'min_samples_split': 19, 'bootstrap': True, 'min_samples_leaf': 4}


searchCV.fit(trainReviews, trainSentiment)
print searchCV.best_params_
print searchCV.best_score_
print searchCV.score(testReviews, testSentiment)



