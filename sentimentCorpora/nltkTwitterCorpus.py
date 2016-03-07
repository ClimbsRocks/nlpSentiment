from nltk.corpus import twitter_samples
from sklearn.feature_extraction import DictVectorizer
import utils

dv = DictVectorizer(sparse=True)


def getFeatures(numWordsToUse):

    # NLTK has their own twitter corpus with positive and negative messages
    positiveTweets = twitter_samples.strings('positive_tweets.json')
    negativeTweets = twitter_samples.strings('negative_tweets.json')

    positiveSentiment = [1 for x in positiveTweets]
    negativeSentiment = [0 for x in positiveTweets]

    tweets = positiveTweets + negativeTweets
    sentiment = positiveSentiment + negativeSentiment


    tokenizedTweets, cleanedSentiment = utils.tokenize(tweets, sentiment)

    global popularWords
    formattedTweets, sentiment, popularWords = utils.nlpFeatureEngineering(
            tokenizedTweets, cleanedSentiment, 0, numWordsToUse, 'counts'
        )

    # transform list of dictionaries into a sparse matrix
    sparseFeatures = dv.fit_transform(formattedTweets)

    return sparseFeatures, sentiment


def formatTestData(testTweets):
    formattedTweets = utils.extractFeaturesList(testTweets, popularWords, 'counts')


    sparseFeatures = dv.transform(formattedTweets)
    return sparseFeatures
