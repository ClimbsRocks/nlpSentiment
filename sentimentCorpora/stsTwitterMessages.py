from sklearn.feature_extraction import DictVectorizer
import nltk
import random
import loadAndProcessData

# putting this in global scope for the module so we can use it on both the training and testing data
dv = DictVectorizer(sparse=True)

# this function takes in a document, and then returns a dictionary with a consistent set of keys for every document
# this ensures that we will have consistent features for each document we process, which becomes critical once we start getting predictions on new documents we did not train on
def extractFeatures(doc):
    docWords = set(doc)
    docFeatures = {}

    for word in popularWords:
        docFeatures[word] = word in docWords
    return docFeatures


def getFeatures(numWordsToUse, allTweets, allTweetsSentiment):

    combined = []
    for rowIdx, tweet in enumerate(allTweets):
        # speed up dev time by only training on a portion of the remaining dataset
        if random.random() > 0.8:
            combined.append( (tweet, allTweetsSentiment[rowIdx]) )
        else:
            allTweets[rowIdx] = None
            allTweetsSentiment[rowIdx] = None
    print 'created the combined dataset'
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
    print 'allWords has been created'

    # grab the top several thousand words, ignoring the 100 most popular
    # grabbing more words leads to more accurate predictions, at the cost of both memory and compute time
    # ignoring the 100 most popular is an easy method for handling stop words that are specific to this dataset, rather than just the English language overall
    global popularWords
    popularWords = list(allWords.keys())[20:numWordsToUse]

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
    sparseFeatures = dv.fit_transform(formattedTweets)

    return sparseFeatures, tweetsSentiment

def formatTestData(testTweets):
    formattedTweets = []
    for tweet in testTweets:
        tweetFeatures = extractFeatures(tweet)
        formattedTweets.append(tweetFeatures)


    sparseFeatures = dv.transform(formattedTweets)

    return sparseFeatures
