import csv
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def loadDataset(fileName, splitNum):
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
                allTweets.append(row[5])
                allTweetSentiments.append( {'sentiment': row[0]} )
                allRows.append(row)

    return allTweets, allTweetSentiments, allRows

def removeStopWords(allTweets):
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
    return allTweets

def writeTestData(testData):

    with open('testdata.all.results.csv', 'wb+') as writeFile:
        csvWriter = csv.writer(writeFile, dialect='excel')
        for row in testData:
            csvWriter.writerow(row)



