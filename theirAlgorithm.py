import csv

def loadDataset(fileName):
    with open(fileName, 'rU') as trainingInput:
        # detect the "dialect" of this type of csv file
        try:
            dialect = csv.Sniffer().sniff(trainingInput.read(1024))
        except:
            # if we fail to detect the dialect, defautl to Microsoft Excel
            dialect = 'excel'
        trainingInput.seek(0)
        trainingRows = csv.reader(trainingInput, dialect)

        allTweets = []
        allTweetSentiments = []
        entireDataset = []
        for row in trainingRows:
            # csv only gives us an iterable, not the data itself
            entireDataset.append(row)

    return entireDataset


# testingTweets, testingSentiment = utils.loadDataset('testdata.manual.2009.06.14.csv', 1)
entireDataset = loadDataset('testdata.manual.updated.csv')

emoticons = {
    '4': [':)','(:',': )','( :','=)','(=','= )','( =',':D',': D',':p',': p',':-)','(-:',':- )','( -:'],
    '0': [':(','):',': (',') :','=(',')=','= (',') =',':D',': D',':p',': p',':-(',')-:',':- (',') -:']
}

print entireDataset
theirAlgosPredictions = []

for row in entireDataset:
    tweet = row[5]

    # set default sentiment score to 2 (neutral)
    sentimentScore = 2

    # try to find a positive emoticon
    for emoticon in emoticons['4']:
        if emoticon in tweet:
            sentimentScore = 4

    # try to find a negative emoticon
    for emoticon in emoticons['0']:
        if emoticon in tweet:
            # if we have already found a positive emoticon for this message, call it neutral
            if sentimentScore == 4:
                sentimentScore = 2
            else:
                sentimentScore = 0

    # add their algorithms prediction to each row
    row.insert(0, sentimentScore)

print entireDataset

utils.writeData(entireDataset)
