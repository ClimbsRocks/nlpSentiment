import csv
import random
import nltk

from nltk.tokenize import word_tokenize, TweetTokenizer
from nltk.corpus import stopwords

# splitNum is used to take a subsample of our dataset
def loadDataset(fileName, splitNum, tweetColumn=5):
    with open(fileName, 'rU') as trainingInput:
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
        allRows = []
        for row in trainingRows:
            rowCount += 1
            # TODO: once we're ready for production, use all the data
            if rowCount % splitNum == 0:
                # csv only gives us an iterable, not the data itself
                # the message of the tweet is at index position 5
                allTweets.append(row[tweetColumn])
                allTweetSentiments.append( row[0] )
                allRows.append(row)

    return allTweets, allTweetSentiments, allRows


# tokenizing means splitting a document into individual words. while it might seem like splitting on spaces is enough, there are a lot of edge cases, which make a built-in tool like NLTK's tokenizers very useful
def tokenize(tweets, sentiment):
    # stopwords are super common words that occur so frequently as to be useless for ML
    stopWords = set(stopwords.words('english'))

    # NLTK has a tokenizer built out specifically for short messaging data
    # here we will use some of it's features to:
     # turn all words to lowercase,
    # reduce the length of repeated characters ('hiiiiiiiii' and 'hiiiii' both become 'hiii' with three repeats of the 'i'),
    # and get rid of any handles that might exist in the message
    tokenizer = TweetTokenizer(preserve_case=False,reduce_len=True,strip_handles=True)

    tokenizedTweets = []
    cleanedSentiment = []

    asciiIssues = 0
    for rowIdx, tweet in enumerate(tweets):
        try:
            tokenizedWords = tokenizer.tokenize(tweet)
            # filteredWords = []

            # for word in tokenizedWords:
            #     if word not in stopWords:
            #         filteredWords.append(word)

            # tokenizedTweets.append(filteredWords)
            tokenizedTweets.append(tokenizedWords)
            cleanedSentiment.append(sentiment[rowIdx])

        except:
            # there are some weird ascii encoding issues present in a small part of our dataset. 
            # they represent < 1% of our dataset
            # for MVP, i'm going to ignore them to focus on the 99% use case
            asciiIssues += 1  

    return tokenizedTweets, cleanedSentiment


# some algorithms do not train well on ordered data. This function shuffles our data so we don't have one big block of positive documents followed by another large block of negative documents
def shuffleOrder(tweets, sentiment):
    combined = []
    for rowIdx, tweet in enumerate(tweets):
        combined.append( (tweet, sentiment[rowIdx]) )
    # our tweets are oftentimes ordered with all the positive reivews at the end
    # to make sure we're not testing on only the positive reviews, we shuffle them
    # to keep the process consistent during development, we will set the seed
    random.seed(8)
    random.shuffle(combined)
    return combined


# parses through a dataset to extract the most popular words
# can limit this selection to exclude highly popular or rarely-used words using lowerBound and upperBound as indices while slicing
def createPopularWords(combined, lowerBound, upperBound):
    allWords = []
    for message in combined:
        # print message
        for word in message[0]:
            # print word
            allWords.append(word)

    allWords = nltk.FreqDist(allWords)

    # grab the top several thousand words, ignoring the 100 most popular
    # grabbing more words leads to more accurate predictions, at the cost of both memory and compute time
    # ignoring the 100 most popular is an easy method for handling stop words that are specific to this dataset, rather than just the English language overall
    popularWords = []
    wordsToUse = allWords.most_common(upperBound)[lowerBound:upperBound]
    for pair in wordsToUse:
        popularWords.append(pair[0])
        # print word
        # print allWords[word]
    # print popularWords
    return popularWords


# extract features from a single document in a consistent manner for all documents in a corpus
# simply returns whether a given word in popularWords is included in the document
def extractFeaturesDoc(doc, popularWords):
    docWords = set(doc)
    docFeatures = {}

    for word in popularWords:
        docFeatures[word] = word in docWords
    return docFeatures


# same as extractFeaturesDoc, but extracts counts of all words in the document
def extractFeatureCountsDoc(doc, popularWords):
    docFeatures = {}
    for word in doc:
        if word in popularWords:
            try:
                docFeatures[word] += 1
            except:
                docFeatures[word] = 1
    return docFeatures


# extract features from all documents in a corpus, allowing for easy testing of whether feature counts or feature inclusions are more useful
def extractFeaturesList(tweets, popularWords, inclusionOrCounts='counts'):
    if inclusionOrCounts == 'counts':
        popularWords = set(popularWords)
        featureExtractor = extractFeatureCountsDoc
    else:
        featureExtractor = extractFeaturesDoc

    formattedTweets = []
    for tweet in tweets:
        formattedTweet = featureExtractor(tweet, popularWords)
        formattedTweets.append(formattedTweet)

    return formattedTweets


# this will perform the entire standardized portion of our NLP feature engineering process 
def nlpFeatureEngineering(tweets, sentiment, lowerBound=0, upperBound=3000, inclusionOrCounts='counts'):
    combined = shuffleOrder(tweets, sentiment)


    popularWords = createPopularWords(combined, lowerBound, upperBound)


    tweets, sentiment = zip(*combined)


    formattedTweets = extractFeaturesList(tweets, popularWords, 'counts')

    
    return formattedTweets, sentiment, popularWords


# aggregate together predictions made from different classifiers we trained on different corpora
# this will allow us to create an effective ensembler algorithm that picks through the results of all these stage 1 predictions
def aggregatePredictions(stsPredictions, movieReviewPredictions, atcPredictions, fileName):
    allPredictions = []

    # all of our predictions lists will have the same number of items, and in the same order
    for idx, prediction in enumerate(stsPredictions):
        # add in a new row with the prediction from the classifier trained on the Stanford Twitter Sentiment training data
        allPredictions.append([prediction])
        # # to that row, add the NLTK-movie-review-trained classifier's prediction
        allPredictions[idx].append(movieReviewPredictions[idx])
        # # to that row, add the Aggregated Twitter Corpus's-trained classifier's prediction
        allPredictions[idx].append(atcPredictions[idx])

    # add header row
    allPredictions.insert(0,['Stanford Twitter Sentiment', 'NLTK Movie Reviews', 'Aggregated Twitter Corpus'])

    writeData(allPredictions, fileName)
    return allPredictions


# write data to a file
def writeData(testData, fileName):

    with open(fileName, 'wb+') as writeFile:
        csvWriter = csv.writer(writeFile, dialect='excel')
        for row in testData:
            csvWriter.writerow(row)



