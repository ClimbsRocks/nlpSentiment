from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, movie_reviews
import nltk
import csv
import random

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
    for row in trainingRows:
        rowCount += 1
        # TODO: once we're ready for production, use all the data
        if rowCount % 100 == 0:
            # csv only gives us an iterable, not the data itself
            # the message of the tweet is at index position 5
            allTweets.append(row[5])

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

    # if rowCount % 5000 == 0:
    #     print filteredWords

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

# print str(reviews[1])
# print reviews[100]

allWords = []
for word in movie_reviews.words():
    allWords.append(word.lower())

allWords = nltk.FreqDist(allWords)

# grab the top 5000 words, ignoring the 100 most popular
# ignoring the 100 most popular is an easy method for handling stop words that are specific to this dataset, rather than just the English language overall
popularWords = list(allWords.keys())[100:5000]

# this function takes in a document, and then returns a dictionary with a consistent set of keys for every document
# ultimately, this is creating a dense matrix of our documents
# obviously, using sparse matrices is an easy optimization to make from here, but it's unnecessary complexity at this stage
def extractFeatures(doc):
    docWords = set(doc)
    docFeatures = {}
    for word in popularWords:
        docFeatures[word] = word in docWords
    return docFeatures

allFeatures = []
# print reviews
for review, category in reviews:
    reviewFeatures = extractFeatures(review)
    allFeatures.append( (reviewFeatures, category) )


