from sklearn.feature_extraction import DictVectorizer

import loadAndProcessData
import trainClassifiers

from sentimentCorpora import nltkMovieReviews
from sentimentCorpora import stsTwitterMessages

# load the "training" data
trainingTweets, trainingSentiment, allRows = loadAndProcessData.loadDataset('training.1600000.processed.noemoticon.csv', 100)
trainingTweets = loadAndProcessData.removeStopWords(trainingTweets)

# load the test data
testTweets, testSentiment, testRows = loadAndProcessData.loadDataset('testdata.manual.2009.06.14.csv', 1)


# our test data has three categories in it, while the trainingTweets only have two categories. That's why we fit on the test data
# DictVectorizer is explained in more depth inside the files in sentimentCorpora
dv = DictVectorizer(sparse=False)
testSentiment = dv.fit_transform(testSentiment)
# trainingSentiment = dv.transform(trainingSentiment)


# we will only be using the top several thousand most frequent words in each sentiment corpus
numWordsToUse = 30000


###############################################################
# Movie Review Corpus
###############################################################
# get features from corpus
movieReviewFeatures, movieReviewsSentiment = nltkMovieReviews.getFeatures(numWordsToUse)
movieReviewsSentiment = dv.transform(movieReviewsSentiment)

# format test tweets to be compatible with the classifier that will be trained from this corpus
movieReviewTestTweets = nltkMovieReviews.formatTestData(testTweets)

# train a classifier from this corpus and use it to get predictions on our test data
movieReviewPredictions = trainClassifiers.trainClassifier(movieReviewFeatures, movieReviewsSentiment, movieReviewTestTweets, testSentiment)


##############################################################
# Training Data Corpus
###############################################################
# get features from corpus
stsFeatures, stsSentiment = stsTwitterMessages.getFeatures(numWordsToUse, trainingTweets, trainingSentiment)
stsSentiment = dv.transform(stsSentiment)

# format test tweets to be compatible with the classifier that will be trained from this corpus
stsTestTweets = stsTwitterMessages.formatTestData(testTweets)

# train a classifier from this corpus and use it to get predictions on our test data
stsPredictions = trainClassifiers.trainClassifier(stsFeatures, stsSentiment, stsTestTweets, testSentiment)


allPredictions = []
# add header row:
allPredictions.append(['Stanford Twitter Sentiment', 'NLTK Movie Reviews'])
for idx, prediction in enumerate(stsPredictions):
    allPredictions.append([prediction])
    testRows[idx].append(movieReviewPredictions[idx])

loadAndProcessData.writeTestData(allPredictions)
