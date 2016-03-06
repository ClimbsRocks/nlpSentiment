from sklearn.feature_extraction import DictVectorizer
import utils

# putting this in global scope for the module so we can use it on both the training and testing data
dv = DictVectorizer(sparse=True)


def getFeatures(numWordsToUse):

    allTweets, allTweetsSentiment, allRows = utils.loadDataset('twitterCorpus/aggregatedCorpusCleaned.csv',1,2)

    tweets = []
    sentiment = []


    rowCount = 0
    for row in allRows:
        # skip header row
        if rowCount == 0:
            rowCount += 1
        else:
            # the aggregatedCorpus has sentiment scores from 1 - 5, while STS has scores from 0 - 4
            rowSentiment = str( int(row[0]) - 1 )

            # only include the row if this is a fairly extreme sentiment
            if rowSentiment in ('0','4'):
                if rowSentiment == '4':
                    rowSentiment = 1
                else:
                    rowSentiment = 0

                tweets.append(row[2])
                sentiment.append(rowSentiment)


    tokenizedTweets, cleanedSentiment = utils.tokenize(tweets, sentiment)

    global popularWords
    formattedTweets, sentiment, popularWords = utils.nlpFeatureEngineering(
            tokenizedTweets, cleanedSentiment,0,numWordsToUse,'counts'
        )

    # right now we have a data structure roughly equivalent to a dense matrix, except each row is a dictionary
    # DictVectorizer performs two key functions for us:
        # 1. transforms each row from a dictionary into a vector using consistent placing of keys into indexed positions within each vector
        # 2. returns sparse vectors, saving enormous amounts of memory which becomes very useful when training our models
    sparseFeatures = dv.fit_transform(formattedTweets)

    return sparseFeatures, sentiment


def formatTestData(testTweets):

    formattedTweets = utils.extractFeaturesList(testTweets, popularWords, 'counts')


    sparseFeatures = dv.transform(formattedTweets)
    return sparseFeatures
