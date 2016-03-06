# I'm working off a dev version of scikit-learn which is prepping for features in the current production release to be deprecated
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import train_test_split

import loadAndProcessData
import trainClassifiers

from sentimentCorpora import nltkMovieReviews
from sentimentCorpora import stsTwitterMessages
from sentimentCorpora import atcTwitterMessages

# load the "training" data
trainingTweets, trainingSentiment, allRows = loadAndProcessData.loadDataset('training.1600000.processed.noemoticon.csv', 100)
trainingTweets = loadAndProcessData.removeStopWords(trainingTweets)

# load the test data
testTweetsAll, testSentiment, testRows = loadAndProcessData.loadDataset('testdata.manual.2009.06.14.csv', 1)

# we are going to test the models on a subsample of the overall test set that is only positive and negative
# running a bunch of tests shows conclusively that for this particular dataset, the models learn most effectively when trained on only positive and negative sentiment. it makes sense that neutral is hard to figure out, since it is a general catch-all bucket, rather than a specific category to be observed.
# we will go in after and take predictions where the model does not feel strongly about the document being either positive or negative, and transform those into neutral predictions
testTweetsPosNegOnly = []
testSentimentPosNegOnly = []
for row in testRows:
    if row[0] in ('0','4'):
        if row[0] == '4':
            testSentimentPosNegOnly.append(1)
        else:
            testSentimentPosNegOnly.append(0)
        testTweetsPosNegOnly.append(row[5])

cleanedTrainingSentiment = []
for score in trainingSentiment:
    if score == '4':
        cleanedTrainingSentiment.append(1)
    else:
        cleanedTrainingSentiment.append(0)
trainingSentiment = cleanedTrainingSentiment

# split out a third of our "training" dataset to train the ensembler algorithm on at the end
trainingTweets, ensembleTweets, trainingSentiment, ensembleSentiment = train_test_split(trainingTweets, trainingSentiment, test_size=0.33, random_state=8)

# our test data has three categories in it, while the trainingTweets only have two categories. That's why we fit on the test data
# DictVectorizer is explained in more depth inside the files in sentimentCorpora
# dvSentiment = DictVectorizer(sparse=False)
# testSentimentPosNegOnly = dvSentiment.fit_transform(testSentimentPosNegOnly)


# we will only be using the top several thousand most frequent words in each sentiment corpus
numWordsToUse = 2000

##############################################################
# Training Data Corpus
###############################################################
# get features from corpus
stsFeatures, stsSentiment = stsTwitterMessages.getFeatures(numWordsToUse, trainingTweets, trainingSentiment)

# format test tweets to be compatible with the classifier that will be trained from this corpus
stsTestTweetsPosNegOnly = stsTwitterMessages.formatTestData(testTweetsPosNegOnly)
stsTestTweetsAll = stsTwitterMessages.formatTestData(testTweetsAll)

# train a classifier from this corpus and use it to get predictions on our test data
stsPredictions = trainClassifiers.trainClassifier(stsFeatures, stsSentiment, stsTestTweetsPosNegOnly, testSentimentPosNegOnly, stsTestTweetsAll)


###############################################################
# Movie Review Corpus
###############################################################
# get features from corpus
movieReviewFeatures, movieReviewsSentiment = nltkMovieReviews.getFeatures(numWordsToUse)

# format test tweets to be compatible with the classifier that will be trained from this corpus
movieReviewTestTweetsPosNegOnly = nltkMovieReviews.formatTestData(testTweetsPosNegOnly)
movieReviewTestTweetsAll = nltkMovieReviews.formatTestData(testTweetsAll)

# train a classifier from this corpus and use it to get predictions on our test data
movieReviewPredictions = trainClassifiers.trainClassifier(movieReviewFeatures, movieReviewsSentiment, movieReviewTestTweetsPosNegOnly, testSentimentPosNegOnly, movieReviewTestTweetsAll)


##############################################################
# Aggregated Twitter Corpus (ATC)
###############################################################
# get features from corpus
atcFeatures, atcSentiment = atcTwitterMessages.getFeatures(numWordsToUse)

# format test tweets to be compatible with the classifier that will be trained from this corpus
atcTestTweetsPosNegOnly = atcTwitterMessages.formatTestData(testTweetsPosNegOnly)
atcTestTweetsAll = atcTwitterMessages.formatTestData(testTweetsAll)

# train a classifier from this corpus and use it to get predictions on our test data
atcPredictions = trainClassifiers.trainClassifier(atcFeatures, atcSentiment, atcTestTweetsPosNegOnly, testSentimentPosNegOnly, atcTestTweetsAll)


allPredictions = []
# add header row
allPredictions.append(['Stanford Twitter Sentiment', 'NLTK Movie Reviews', 'Aggregated Twitter Corpus'])

# all of our predictions lists will have the same number of items, and in the same order
for idx, prediction in enumerate(atcPredictions):
    # add in a new row with the prediction from the classifier trained on the Aggregated Twitter Corpus
    allPredictions.append([prediction])
    # # to that row, add the NLTK-movie-review-trained classifier's prediction
    allPredictions[idx].append(movieReviewPredictions[idx])
    # # to that row, add the STS-trained classifier's prediction
    # allPredictions[idx].append(movieReviewPredictions[idx])

loadAndProcessData.writeTestData(allPredictions)
