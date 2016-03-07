from nltk.corpus import twitter_samples
from sklearn.feature_extraction import DictVectorizer
import utils

dv = DictVectorizer(sparse=True)


def getFeatures(numWordsToUse):

    # these emoticons cover 98.6% of the training data
    emoticons = [':)','(:',': )','( :','=)','(=','= )','( =',':D',': D',':p',': p',':(','):',': (',') :','=(',')=','= (',') =',':D',': D',':p',': p',':-)','(-:',':- )','( -:',':-(',')-:',':- (',') -:']
    emoticons = set(emoticons)
    # NLTK has their own twitter corpus with positive and negative messages
    positiveTweets = twitter_samples.strings('positive_tweets.json')
    negativeTweets = twitter_samples.strings('negative_tweets.json')

    positiveSentiment = [1 for x in positiveTweets]
    negativeSentiment = [0 for x in positiveTweets]

    tweets = positiveTweets + negativeTweets
    sentiment = positiveSentiment + negativeSentiment


    tokenizedTweets, cleanedSentiment = utils.tokenize(tweets, sentiment)


    cleanedTweets = []
    linesCleaned = 0
    for tweet in tokenizedTweets:
        replacedEmoticon = 0
        cleanedTweet = []
        for word in tweet:
            if word not in emoticons:
                cleanedTweet.append(word)
            else:
                replacedEmoticon = 1
        cleanedTweets.append(cleanedTweet)
        linesCleaned += replacedEmoticon

    global popularWords
    formattedTweets, sentiment, popularWords = utils.nlpFeatureEngineering(
            cleanedTweets, cleanedSentiment, 0, numWordsToUse, 'counts'
        )

    # transform list of dictionaries into a sparse matrix
    sparseFeatures = dv.fit_transform(formattedTweets)

    return sparseFeatures, sentiment


def formatTestData(testTweets):
    formattedTweets = utils.extractFeaturesList(testTweets, popularWords, 'counts')


    sparseFeatures = dv.transform(formattedTweets)
    return sparseFeatures
