from nltk.corpus import movie_reviews
import nltk
import random
from sklearn.feature_extraction import DictVectorizer


def getFeatures(numWordsToUse):
    # create a reviews list that holds two pieces of informatin on each reivew:
        # the words of the review
        # the category (pos or neg) for that review
    reviews = []
    for category in movie_reviews.categories():
        for fileid in movie_reviews.fileids(category):
            reviewWords = list(movie_reviews.words(fileid))
            reviews.append( (reviewWords, category) )

    # our reviews are ordered with all the positive reivews at the end
    # to make sure we're not testing on only the positive reviews, we shuffle them
    # to keep the process consistent during development, we will set the seed
    random.seed(8)
    random.shuffle(reviews)


    allWords = []
    for word in movie_reviews.words():
        allWords.append(word.lower())

    allWords = nltk.FreqDist(allWords)


    # grab the top several thousand words, ignoring the 100 most popular
    # grabbing more words leads to more accurate predictions, at the cost of both memory and compute time
    # ignoring the 100 most popular is an easy method for handling stop words that are specific to this dataset, rather than just the English language overall
    popularWords = list(allWords.keys())[100:numWordsToUse]


    # this function takes in a document, and then returns a dictionary with a consistent set of keys for every document
    # this ensures that we will have consistent features for each document we process, which becomes critical once we start getting predictions on new documents we did not train on
    def extractFeatures(doc):
        docWords = set(doc)
        docFeatures = {}

        for word in popularWords:
            docFeatures[word] = word in docWords
        return docFeatures


    formattedReviews = []
    reviewsSentiment = []

    for review, category in reviews:
        reviewFeatures = extractFeatures(review)
        formattedReviews.append(reviewFeatures)
        reviewsSentiment.append(str(category))


    # right now we have a data structure roughly equivalent to a dense matrix, except each row is a dictionary
    # DictVectorizer performs two key functions for us:
        # 1. turns each row form a dictionary into a vector using consistent placing of keys into indexed positions within each vector
        # 2. returns sparse vectors, saving enormous amounts of memory which becomes very useful when training our models
    dv = DictVectorizer(sparse=True)
    sparseFeatures = dv.fit_transform(formattedReviews)

    return sparseFeatures, reviewsSentiment
