import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import RandomizedSearchCV
import scipy

def trainClassifier(X, y, testTweetsPosNegOnly, testSentimentPosNegOnly, testTweetsAll):

    # split our data into training and test datasets
    xTrain, xTest, yTrain, yTest = train_test_split(
        X, y, test_size=0.33, random_state=8)

    classifier = RandomForestClassifier(n_estimators=50)

    # for simplicity's sake, we could train a single random forest:
    # classifier.fit(xTrain, yTrain)
    # print classifier.score(xTest, yTest)


    # for more fun, we will optimize the hyperparameters for our random forest using RandomizedSearchCV
    # parametersToTry = {
    #     # 'max_features': scipy.stats.randint(1,numWordsToUse/7),
    #     'criterion': ['gini','entropy'],
    #     'min_samples_leaf': scipy.stats.randint(1,30),
    #     'min_samples_split': scipy.stats.randint(2,30),
    #     'bootstrap': [True,False]
    # }

    # for ATC twitter data:
    parametersToTry = {
        # 'max_features': ['sqrt','log2'],
        'criterion': ['entropy'],
        'min_samples_leaf': [1],
        'min_samples_split': scipy.stats.randint(2,30),
        'bootstrap': [True,False]
    }


    # run on all cores, fail gracefully if a combination of hyperparameters fails to converge, try 100 different combinations of hyperparameters, train on all the training data when finished, and use a third of the dataset for cross-validation while training
    searchCV = RandomizedSearchCV(classifier, parametersToTry, n_jobs=-1, error_score=0, n_iter=50, refit=True, cv=3)
    # best results out of 10k training runs:
    # {'max_features': 51, 'min_samples_split': 19, 'bootstrap': True, 'min_samples_leaf': 4}

    print xTrain.shape
    print len(yTrain)
    searchCV.fit(xTrain, yTrain)
    print searchCV.best_params_
    print 'best score from hyperparameter search is: ' + str(searchCV.best_score_)
    print 'score on the holdout portion of the training set: ' + str( searchCV.score(xTest, yTest) )
    print 'score on the test data: ' + str( searchCV.score(testTweetsPosNegOnly, testSentimentPosNegOnly) )


    testPredictions = searchCV.predict_proba(testTweetsAll)

    # print 'testPredictions:'
    # print testPredictions

    # for corpora that only predict positive and negative, we can assume that messages where the algorithm doesn't find any strong polarity are neutral
    def addNeutral(predictions):
        cleanedPredictions = []

        # each prediction row contains the probability for negative and positive
        for row in predictions:
            print row
            if row[0] < .45:
                cleanedPredictions.append('0')
            elif row[0] > .45 and row[0] < .55:
                cleanedPredictions.append('2')
            else:
                cleanedPredictions.append('4')
        return cleanedPredictions

    # testPredictions = addNeutral(testPredictions)
    singlePrediction = []
    for row in testPredictions:
        singlePrediction.append(row[0])
    testPredictions = singlePrediction

    return testPredictions
