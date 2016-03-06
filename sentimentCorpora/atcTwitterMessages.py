from sklearn.feature_extraction import DictVectorizer
import nltk
import random
import loadAndProcessData

# putting this in global scope for the module so we can use it on both the training and testing data
dvFeatures = DictVectorizer(sparse=True)
dvSentiment = DictVectorizer(sparse=False)

# this function takes in a document, and then returns a dictionary with a consistent set of keys for every document
# this ensures that we will have consistent features for each document we process, which becomes critical once we start getting predictions on new documents we did not train on
def extractFeatures(doc):
    docWords = set(doc)
    docFeatures = {}

    for word in popularWords:
        docFeatures[word] = word in docWords
    return docFeatures

# extract the count of all the words in each document
def extractFeatureCounts(doc):
    docFeatures = {}
    for word in doc:
        if word in popularWords:
            try:
                docFeatures[word] += 1
            except:
                docFeatures[word] = 1

    return docFeatures


def getFeatures(numWordsToUse):

    allTweets, allTweetsSentiment, allRows = loadAndProcessData.loadDataset('twitterCorpus/aggregatedCorpusCleaned.csv',1,2)

    combined = []
    rowCount = 0
    for row in allRows:
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

                combined.append( (row[2], rowSentiment ) )
            # elif row[1] > 0.85:
            #     # otherwise, if it falls into 'slightly negative' or 'slightly positive', and we have a high degree of confidence in that score, include it as either purely negative or purely positive
            #     if rowSentiment == '1':
            #         combined.append( (row[2], {'sentiment': '0'} ) )
            #     else:
            #         combined.append( (row[2], {'sentiment': '4'} ) )



                
            # # several of the ATC corpora rank sentiment on a 5 point scale, while STS only asks for it on a 3 point scale
            # # here we're aggregating 'slightly negative' into negative and 'slightly positive' into positive
            # if rowSentiment == '3':
            #     rowSentiment = '4'
            # elif rowSentiment == '1':
            #     rowSentiment = '0'


    # TODO:
        # reduce every message's sentiment by 1 to align with STS's grading scale
        # look into whether we have to use an object with a sentiment property to stay super consistent with other datasets and our testTweets
        # ignore slightly positive and slightly negative messages for now
    # for rowIdx, tweet in enumerate(allTweets):
    #     combined.append( (tweet, allTweetsSentiment[rowIdx] ) )
    # our reviews are ordered with all the positive reivews at the end
    # to make sure we're not testing on only the positive reviews, we shuffle them
    # to keep the process consistent during development, we will set the seed
    random.seed(8)
    random.shuffle(combined)

    allWords = []
    for message in combined:
        for word in message[0]:
            allWords.append(word.lower())

    allWords = nltk.FreqDist(allWords)


    # grab the top several thousand words, ignoring the 100 most popular
    # grabbing more words leads to more accurate predictions, at the cost of both memory and compute time
    # ignoring the 100 most popular is an easy method for handling stop words that are specific to this dataset, rather than just the English language overall
    global popularWords
    popularWords = list(allWords.keys())[50:numWordsToUse]

    formattedTweets = []
    tweetsSentiment = []

    for tweet, sentiment in combined:
        tweetFeatures = extractFeatures(tweet)
        formattedTweets.append(tweetFeatures)
        tweetsSentiment.append(sentiment)


    # right now we have a data structure roughly equivalent to a dense matrix, except each row is a dictionary
    # DictVectorizer performs two key functions for us:
        # 1. turns each row form a dictionary into a vector using consistent placing of keys into indexed positions within each vector
        # 2. returns sparse vectors, saving enormous amounts of memory which becomes very useful when training our models
    sparseFeatures = dvFeatures.fit_transform( formattedTweets )
    # tweetsSentiment = dvSentiment.fit_transform( tweetsSentiment )

    return sparseFeatures, tweetsSentiment

def formatTestData(testTweets):
    formattedTweets = []
    for tweet in testTweets:
        tweetFeatures = extractFeatures(tweet)
        formattedTweets.append(tweetFeatures)


    sparseFeatures = dvFeatures.transform(formattedTweets)

    return sparseFeatures
