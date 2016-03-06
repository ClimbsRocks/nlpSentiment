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
    # each corpus's getFeatures function is responsible for somehow loading in their own allTweets and allTweetsSentiment data
    # then they have to ensure that data is tokenized (leveraging the modular tokenization functionality in loadAndProcessData)
    # then shuffle the dataset
    # then create the frequency distribution and popularWords
    # then extract features from each tweet, and un-combine the sentiment again


    global popularWords
    formattedTweets, sentiment, popularWords = loadAndProcessData.nlpFeatureEngineering(
            allTweets, allTweetsSentiment,0,numWordsToUse,'counts'
        )

    # right now we have a data structure roughly equivalent to a dense matrix, except each row is a dictionary
    # DictVectorizer performs two key functions for us:
        # 1. turns each row form a dictionary into a vector using consistent placing of keys into indexed positions within each vector
        # 2. returns sparse vectors, saving enormous amounts of memory which becomes very useful when training our models
    sparseFeatures = dv.fit_transform(formattedTweets)

    return sparseFeatures, sentiment

def formatTestData(testTweets):
    formattedTweets = loadAndProcessData.extractFeaturesList(testTweets, popularWords, 'counts')


    sparseFeatures = dv.transform(formattedTweets)

    return sparseFeatures
