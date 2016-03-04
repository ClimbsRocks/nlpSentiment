from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, movie_reviews
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.grid_search import RandomizedSearchCV
import nltk
import csv
import random
import scipy
import numpy as np
import math

with open('training.1600000.processed.noemoticon.csv', 'rU') as trainingInput:
    # detect the "dialect" of this type of csv file
    try:
        dialect = csv.Sniffer().sniff(trainingInput.read(1024))
    except:
        # if we fail to detect the dialect, defautl to Microsoft Excel
        dialect = 'excel'
    trainingInput.seek(0)
    trainingRows = csv.reader(trainingInput, dialect)

    rowCount = 0

    allTweets = []
    allTweetSentiments = []
    for row in trainingRows:
        rowCount += 1
        # TODO: once we're ready for production, use all the data
        if rowCount % 100 == 0:
            # csv only gives us an iterable, not the data itself
            # the message of the tweet is at index position 5
            allTweets.append(row[5])
            allTweetSentiments.append(row[0])



# stopwords are super common words that occur so frequently as to be useless for ML
stopWords = set(stopwords.words('english'))

rowCount = 0
asciiIssues = 0
for tweet in allTweets:
    rowCount += 1
    try:
        tokenizedWords = word_tokenize(tweet)
        filteredWords = []

        # we will be removing all Twitter handles as well
        # they may occasionally be useful, but will also take up a lot of space
        isHandle = False

        for word in tokenizedWords:
            word = word.lower()

            if word == '@':
                isHandle = True

            elif isHandle:
                # if the previous word was '@', this word is a user's handle
                # we want to skip this word, and clear the slate for the next word
                isHandle = False

            elif word not in stopWords:
                filteredWords.append(word)

    except:
        # there are some weird ascii encoding issues present in a small part of our dataset. 
        # they represent < 1% of our dataset
        # for MVP, i'm going to ignore them to focus on the 99% use case
        asciiIssues += 1    

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
numWordsToUse = 3000
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


# split our data into training and test datasets
trainReviews, testReviews, trainSentiment, testSentiment = train_test_split(
    sparseFeatures, reviewsSentiment, test_size=0.33, random_state=8)


# train a single random forest on our training data, test on our testing data
classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(trainReviews, trainSentiment)
print classifier.score(testReviews, testSentiment)

# optimize the hyperparameters for our random forest using RandomizedSearchCV
sqrtNum = int( math.log( math.sqrt(numWordsToUse) ) )
# print sqrtNum
parametersToTry = {
    # 'max_features': scipy.stats.randint(1,numWordsToUse/7),
    'min_samples_leaf': scipy.stats.randint(1,30),
    'min_samples_split': scipy.stats.randint(2,30),
    'bootstrap': [True,False]
}
# run on all cores, fail gracefully if a combination of hyperparameters fails to converge, try 10 different combinations of hyperparameters, train on all the training data when finished, and use a third of the dataset for cross-validation while training
searchCV = RandomizedSearchCV(classifier, parametersToTry, n_jobs=-1, error_score=0, n_iter=1000, refit=True, cv=3)
# best results out of 10k training runs:
# {'max_features': 51, 'min_samples_split': 19, 'bootstrap': True, 'min_samples_leaf': 4}


searchCV.fit(trainReviews, trainSentiment)
print searchCV.best_params_
print searchCV.best_score_
print searchCV.score(testReviews, testSentiment)



