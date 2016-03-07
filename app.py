# I'm working off a dev version of scikit-learn which is prepping for features in the current production release to be deprecated
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import train_test_split

import utils
import trainClassifiers

from sentimentCorpora import nltkMovieReviews
from sentimentCorpora import stsTwitterCorpus
from sentimentCorpora import atcTwitterCorpus
from sentimentCorpora import nltkTwitterCorpus
from sentimentCorpora import nltkTwitterNoEmoticonsCorpus


# load the "training" data
trainingTweets, trainingSentiment, allRows = utils.loadDataset('training.1600000.processed.noemoticon.csv', 10)
trainingTweets, trainingSentiment = utils.tokenize(trainingTweets, trainingSentiment)


# load the test data
testTweetsAll, testSentiment, testRows = utils.loadDataset('testdata.manual.2009.06.14.csv', 1)
testTweetsAll, testSentiment = utils.tokenize(testTweetsAll, testSentiment)


# we are going to test the models on a subsample of the overall test set that is only positive and negative
# running a bunch of tests shows that for this particular dataset, the models learn most effectively when trained on only positive and negative sentiment. it makes sense that neutral is hard to figure out, since it is a general catch-all bucket, rather than a specific category to be observed.
# we will go in after and take predictions where the model does not feel strongly about the document being either positive or negative, and transform those into neutral predictions
# for now, separate out only the positive and negative documents in our testdata
testTweetsPosNegOnly = []
testSentimentPosNegOnly = []
for row in testRows:
    if row[0] in ('0','4'):
        if row[0] == '4':
            testSentimentPosNegOnly.append(1)
        else:
            testSentimentPosNegOnly.append(0)
        testTweetsPosNegOnly.append(row[5])


# instead of predicting two categories ('0', and '4') that the algorithm doesn't inherently understand are mutually exclusive, we will explicitly turn this into a single binary classification problem (0 or 1)
cleanedTrainingSentiment = []
for score in trainingSentiment:
    if score == '4':
        cleanedTrainingSentiment.append(1)
    else:
        cleanedTrainingSentiment.append(0)
trainingSentiment = cleanedTrainingSentiment


# split out a third of our "training" dataset to train the ensembler algorithm on at the end
trainingTweets, ensembleTweets, trainingSentiment, ensembleSentiment = train_test_split(trainingTweets, trainingSentiment, test_size=0.33, random_state=8)


# we will only be using the top several thousand most frequent words in each sentiment corpus
numWordsToUse = 10000


##############################################################
# Stanford Twitter Sentiment (STS)
###############################################################
# this is the original "training" data provided
# get features from corpus
stsFeatures, stsSentiment = stsTwitterCorpus.getFeatures(numWordsToUse, trainingTweets, trainingSentiment)
stsEnsembleTweets = stsTwitterCorpus.formatTestData(ensembleTweets)

# format test tweets to be compatible with the classifier that will be trained from this corpus
stsTestTweetsPosNegOnly = stsTwitterCorpus.formatTestData(testTweetsPosNegOnly)
stsTestTweetsAll = stsTwitterCorpus.formatTestData(testTweetsAll)

# train a classifier from this corpus and use it to get predictions on our test data
stsPredictions, stsEnsemblePredictions = trainClassifiers.trainClassifier(stsFeatures, stsSentiment, stsTestTweetsPosNegOnly, testSentimentPosNegOnly, stsTestTweetsAll, stsEnsembleTweets, ensembleSentiment)


###############################################################
# Movie Review Corpus
###############################################################
# get features from corpus
movieReviewFeatures, movieReviewsSentiment = nltkMovieReviews.getFeatures(numWordsToUse)
movieReviewEnsembleTweets = nltkMovieReviews.formatTestData(ensembleTweets)

# format test tweets to be compatible with the classifier that will be trained from this corpus
movieReviewTestTweetsPosNegOnly = nltkMovieReviews.formatTestData(testTweetsPosNegOnly)
movieReviewTestTweetsAll = nltkMovieReviews.formatTestData(testTweetsAll)

# train a classifier from this corpus and use it to get predictions on our test data
movieReviewPredictions, movieReviewEnsemblePredictions = trainClassifiers.trainClassifier(movieReviewFeatures, movieReviewsSentiment, movieReviewTestTweetsPosNegOnly, testSentimentPosNegOnly, movieReviewTestTweetsAll, movieReviewEnsembleTweets, ensembleSentiment)


##############################################################
# Aggregated Twitter Corpus (ATC)
###############################################################
# get features from corpus
atcFeatures, atcSentiment = atcTwitterCorpus.getFeatures(numWordsToUse)
atcEnsembleTweets = atcTwitterCorpus.formatTestData(ensembleTweets)

# format test tweets to be compatible with the classifier that will be trained from this corpus
atcTestTweetsPosNegOnly = atcTwitterCorpus.formatTestData(testTweetsPosNegOnly)
atcTestTweetsAll = atcTwitterCorpus.formatTestData(testTweetsAll)

# train a classifier from this corpus and use it to get predictions on our test data
atcPredictions, atcEnsemblePredictions = trainClassifiers.trainClassifier(atcFeatures, atcSentiment, atcTestTweetsPosNegOnly, testSentimentPosNegOnly, atcTestTweetsAll, atcEnsembleTweets, ensembleSentiment)


##############################################################
# NLTK Twitter Corpus (NTC)
###############################################################
# get features from corpus
ntcFeatures, ntcSentiment = nltkTwitterCorpus.getFeatures(numWordsToUse)
ntcEnsembleTweets = nltkTwitterCorpus.formatTestData(ensembleTweets)

# format test tweets to be compatible with the classifier that will be trained from this corpus
ntcTestTweetsPosNegOnly = nltkTwitterCorpus.formatTestData(testTweetsPosNegOnly)
ntcTestTweetsAll = nltkTwitterCorpus.formatTestData(testTweetsAll)

# train a classifier from this corpus and use it to get predictions on our test data
ntcPredictions, ntcEnsemblePredictions = trainClassifiers.trainClassifier(ntcFeatures, ntcSentiment, ntcTestTweetsPosNegOnly, testSentimentPosNegOnly, ntcTestTweetsAll, ntcEnsembleTweets, ensembleSentiment)


##############################################################
# NLTK Twitter Corpus without emoticons (ntcNoEmot)
###############################################################
# get features from corpus
ntcNoEmotFeatures, ntcNoEmotSentiment = nltkTwitterNoEmoticonsCorpus.getFeatures(numWordsToUse)
ntcNoEmotEnsembleTweets = nltkTwitterNoEmoticonsCorpus.formatTestData(ensembleTweets)

# format test tweets to be compatible with the classifier that will be trained from this corpus
ntcNoEmotTestTweetsPosNegOnly = nltkTwitterNoEmoticonsCorpus.formatTestData(testTweetsPosNegOnly)
ntcNoEmotTestTweetsAll = nltkTwitterNoEmoticonsCorpus.formatTestData(testTweetsAll)

# train a classifier from this corpus and use it to get predictions on our test data
ntcNoEmotPredictions, ntcNoEmotEnsemblePredictions = trainClassifiers.trainClassifier(ntcNoEmotFeatures, ntcNoEmotSentiment, ntcNoEmotTestTweetsPosNegOnly, testSentimentPosNegOnly, ntcNoEmotTestTweetsAll, ntcNoEmotEnsembleTweets, ensembleSentiment)


# aggregate all our predictions together into a single matrix that we will train our ensemble classifier on
ensembledPredictions = utils.aggregatePredictions(stsEnsemblePredictions, movieReviewEnsemblePredictions, atcEnsemblePredictions, ntcEnsemblePredictions, ntcNoEmotEnsemblePredictions, 'ensembleData.all.predictions.csv')

# do the same for our test data
testPredictions = utils.aggregatePredictions(stsPredictions, movieReviewPredictions, atcPredictions, ntcPredictions, ntcNoEmotPredictions, 'testdata.all.predictions.csv')


# train the ensemble classifier on our ensembleData
# this will return a matrix with all the stage 1 classifiers' predictions on the test data, as well as the final predictions the ensemble algorithm produces
finalPredictions = trainClassifiers.trainEnsembleClassifier(ensembledPredictions, ensembleSentiment, testPredictions)


utils.writeData(finalPredictions, 'testdata.entire.ensembled.predictions.csv')

