import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import RandomizedSearchCV
import scipy

def trainClassifier(X, y, testTweetsAll, ensembleTweets, ensembleSentiment):

    # split our data into training and test datasets
    xTrain, xTest, yTrain, yTest = train_test_split(
        X, y, test_size=0.33, random_state=8)


    classifier = RandomForestClassifier(n_estimators=20, n_jobs=-1)

    # for simplicity's sake, we could train a single random forest:
    # classifier.fit(xTrain, yTrain)
    # print classifier.score(xTest, yTest)


    # for more fun, we will optimize the hyperparameters for our random forest using RandomizedSearchCV
    parametersToTry = {
        'max_features': ['sqrt','log2',None,.01,.1,.2,.3],
        'criterion': ['gini','entropy'],
        'min_samples_leaf': [1],
        'min_samples_split': scipy.stats.randint(2,30),
        'bootstrap': [True,False]
    }

    # RandomizedSearchCV will optimize our hyperparameters for us in a way that is much more efficient and comprehensive than GridSearchCV.
    # run on all cores, fail gracefully if a combination of hyperparameters fails to converge, try 10 different combinations of hyperparameters, train on all the training data when finished, and use a third of the dataset for cross-validation while searching for the best hyperparameters
    searchCV = RandomizedSearchCV(classifier, parametersToTry, n_jobs=-1, error_score=0, n_iter=10, refit=True, cv=3)


    print 'shape of this training data set:'
    print xTrain.shape
    searchCV.fit(xTrain, yTrain)
    print 'the best hyperparameters from this search are:'
    print searchCV.best_params_
    print 'best score from hyperparameter search is: ' + str(searchCV.best_score_)
    print 'score on the holdout portion of the training set: ' + str( searchCV.score(xTest, yTest) )
    print 'score on the ensemble data: ' + str( searchCV.score(ensembleTweets, ensembleSentiment) ) + '\n\n'


    testPredictions = searchCV.predict_proba(testTweetsAll)
    ensemblePredictions = searchCV.predict_proba(ensembleTweets)


    def singlePrediction(predictions):
        cleanedPredictions = []
        for predictionRow in predictions:
            cleanedPredictions.append(predictionRow[1])
        return cleanedPredictions

    # the classifier gives us a predicted probability for both the 0 and the 1 case. Given that they're mutually exclusive, we can simplify down to a single number (the predicted probability of the 1 case)
    testPredictions = singlePrediction(testPredictions)
    ensemblePredictions = singlePrediction(ensemblePredictions)

    return testPredictions, ensemblePredictions


def trainEnsembleClassifier(ensemblePredictions, ensembleSentiment, testPredictions):
    headerRow = ensemblePredictions.pop(0)
    testPredictions.pop(0)


    classifier = RandomForestClassifier(n_estimators=200,n_jobs=-1)
    classifier.fit(ensemblePredictions, ensembleSentiment)

    print 'the ensemble classifier\'s predictions on the test tweets are saved to "testdata.entire.ensembled.predictions.csv"'
    finalTestPredictions = classifier.predict_proba(testPredictions)


    # for corpora that only predict positive and negative, we can assume that messages where the algorithm doesn't find any strong polarity are neutral
    def addNeutral(predictions):
        cleanedPredictions = []

        # each prediction row contains the probability for negative and positive
        for row in predictions:
            if row[1] <= .45:
                cleanedPredictions.append('0')
            elif row[1] > .45 and row[1] < .55:
                cleanedPredictions.append('2')
            else:
                cleanedPredictions.append('4')
        return cleanedPredictions

    testPredictionsWithNeutral = addNeutral(finalTestPredictions)


    for rowIdx, prediction in enumerate(finalTestPredictions):
        testPredictions[rowIdx].append(prediction[1])
        testPredictions[rowIdx].append( testPredictionsWithNeutral[rowIdx] )


    headerRow.append('Ensembled Probability')
    headerRow.append('Final Ensembled Prediction')

    testPredictions.insert(0, headerRow)


    return testPredictions
