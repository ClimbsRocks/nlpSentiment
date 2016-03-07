# I'm working off a dev version of scikit-learn which is preparing for features in the current production release to be deprecated
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
testTweets, testSentiment, testRows = utils.loadDataset('testdata.manual.2009.06.14.csv', 1)
testTweets, testSentiment = utils.tokenize(testTweets, testSentiment)


# instead of predicting two categories ('0', and '4') that the algorithm doesn't inherently understand are mutually exclusive, we will explicitly turn this into a single binary classification problem (0 or 1)
cleanedTrainingSentiment = []
for score in trainingSentiment:
    if score == '4':
        cleanedTrainingSentiment.append(1)
    else:
        cleanedTrainingSentiment.append(0)
trainingSentiment = cleanedTrainingSentiment


# split out most of our "training" dataset to train the ensembler algorithm on at the end
# training the ensembler is relatively quick compared to training the stage1 algorithms, so we will give it the bulk of the data to train on
trainingTweets, ensembleTweets, trainingSentiment, ensembleSentiment = train_test_split(trainingTweets, trainingSentiment, test_size=0.95, random_state=8)


# we will only be using the top several thousand most frequent words in each sentiment corpus
numWordsToUse = 3000


##############################################################
# Stanford Twitter Sentiment (STS)
##############################################################
# this is the original "training" data provided
# get features from corpus
stsFeatures, stsSentiment = stsTwitterCorpus.getFeatures(numWordsToUse, trainingTweets, trainingSentiment)

# format ensemble and test tweets to be compatible with the classifier that will be trained from this corpus
stsTestTweets = stsTwitterCorpus.formatTestData(testTweets)
stsEnsembleTweets = stsTwitterCorpus.formatTestData(ensembleTweets)

# train a classifier from this corpus and use it to get predictions on our test data
stsPredictions, stsEnsemblePredictions = trainClassifiers.trainClassifier(stsFeatures, stsSentiment, stsTestTweets, stsEnsembleTweets, ensembleSentiment)


# each following corpus follows nearly the exact same process
##############################################################
# Movie Review Corpus
##############################################################
movieReviewFeatures, movieReviewsSentiment = nltkMovieReviews.getFeatures(numWordsToUse)

movieReviewTestTweets = nltkMovieReviews.formatTestData(testTweets)
movieReviewEnsembleTweets = nltkMovieReviews.formatTestData(ensembleTweets)

movieReviewPredictions, movieReviewEnsemblePredictions = trainClassifiers.trainClassifier(movieReviewFeatures, movieReviewsSentiment, movieReviewTestTweets, movieReviewEnsembleTweets, ensembleSentiment)


##############################################################
# Aggregated Twitter Corpus (ATC)
##############################################################
atcFeatures, atcSentiment = atcTwitterCorpus.getFeatures(numWordsToUse)

atcTestTweets = atcTwitterCorpus.formatTestData(testTweets)
atcEnsembleTweets = atcTwitterCorpus.formatTestData(ensembleTweets)

atcPredictions, atcEnsemblePredictions = trainClassifiers.trainClassifier(atcFeatures, atcSentiment, atcTestTweets, atcEnsembleTweets, ensembleSentiment)


##############################################################
# NLTK Twitter Corpus (NTC)
##############################################################
ntcFeatures, ntcSentiment = nltkTwitterCorpus.getFeatures(numWordsToUse)

ntcTestTweets = nltkTwitterCorpus.formatTestData(testTweets)
ntcEnsembleTweets = nltkTwitterCorpus.formatTestData(ensembleTweets)

ntcPredictions, ntcEnsemblePredictions = trainClassifiers.trainClassifier(ntcFeatures, ntcSentiment, ntcTestTweets, ntcEnsembleTweets, ensembleSentiment)


##############################################################
# NLTK Twitter Corpus without emoticons (ntcNoEmot)
##############################################################
ntcNoEmotFeatures, ntcNoEmotSentiment = nltkTwitterNoEmoticonsCorpus.getFeatures(numWordsToUse)

ntcNoEmotTestTweets = nltkTwitterNoEmoticonsCorpus.formatTestData(testTweets)
ntcNoEmotEnsembleTweets = nltkTwitterNoEmoticonsCorpus.formatTestData(ensembleTweets)

ntcNoEmotPredictions, ntcNoEmotEnsemblePredictions = trainClassifiers.trainClassifier(ntcNoEmotFeatures, ntcNoEmotSentiment, ntcNoEmotTestTweets, ntcNoEmotEnsembleTweets, ensembleSentiment)



# aggregate all our predictions together into a single matrix that we will train our ensemble classifier on
ensembledPredictions = utils.aggregatePredictions(stsEnsemblePredictions, movieReviewEnsemblePredictions, atcEnsemblePredictions, ntcEnsemblePredictions, ntcNoEmotEnsemblePredictions, 'ensembleData.all.predictions.csv')

# do the same for our test data
testPredictions = utils.aggregatePredictions(stsPredictions, movieReviewPredictions, atcPredictions, ntcPredictions, ntcNoEmotPredictions, 'testdata.all.predictions.csv')


# train the ensemble classifier on our ensembleData
# this will return a matrix with all the stage 1 classifiers' predictions on the test data, as well as the final predictions the ensemble algorithm produces
finalPredictions = trainClassifiers.trainEnsembleClassifier(ensembledPredictions, ensembleSentiment, testPredictions)


utils.writeData(finalPredictions, 'testdata.entire.ensembled.predictions.csv')

