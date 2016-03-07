from nltk.corpus import movie_reviews, stopwords
from sklearn.feature_extraction import DictVectorizer
import utils

dv = DictVectorizer(sparse=True)


def getFeatures(numWordsToUse):
    # stopwords are common words that occur so frequently as to be useless for NLP
    stopWords = set(stopwords.words('english'))


    # read in all the words of each movie review, and it's associated sentiment
    reviewDocuments = []
    sentiment = []

    for category in movie_reviews.categories():
        for fileid in movie_reviews.fileids(category):
            reviewWords = movie_reviews.words(fileid)

            cleanedReview = []
            for word in reviewWords:
                if word not in stopWords:
                    cleanedReview.append(word)

            reviewDocuments.append(cleanedReview)
            if category == 'pos':
                sentiment.append(1)
            elif category == 'neg':
                sentiment.append(0)
            else:
                print 'We are not sure what this category is: ' + category

    global popularWords
    formattedReviews, sentiment, popularWords = utils.nlpFeatureEngineering(
            reviewDocuments, sentiment, 50, numWordsToUse, 'counts'
        )


    # transform list of dictionaries into a sparse matrix
    sparseFeatures = dv.fit_transform(formattedReviews)

    return sparseFeatures, sentiment


def formatTestData(testTweets):
    formattedTweets = utils.extractFeaturesList(testTweets, popularWords, 'counts')

    sparseFeatures = dv.transform(formattedTweets)
    
    return sparseFeatures
