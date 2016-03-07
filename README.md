# nlpPredictions
> Predicting the sentiment of tweets using NLP. 

## Installation

From the command line

1. `git clone https://github.com/ClimbsRocks/nlpSentiment.git`
1. `cd nlpSentiment`
1. `pip install -r requirements.txt`
1. If this fails to install scikit-learn properly, you may have to `pip install numpy` and `pip install scipy`
1. Open python on the command line
1. `import nltk`
1. `nltk.download()` This will open a GUI.
1. Follow prompts to download everything. This will download 1.8GB of material.
1. Download the [testing and training data](http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip). The easiest way, in my opinion, is `curl -O http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip` from within the `nlpSentiment` directory. 
1. Unzip those files
1. Move those two .csv files, with the names they already have, to the `nlpSentiment` directory
1. Back on the command line inside the nlpSentiment directory: `python app.py`

I have included a copy of the results this project produces in .csv format. Per usual, there is a trade-off between computing power and accuracy. The version of the code pushed up to GitHub is biased towards being able to run quickly to demonstrate the process. The results I have incldued in the .csv files come from a longer run of the data, and doubtless could be improved even more if allowed to run overnight. 

### Results
Several of the classifiers were individually able to achieve accuracy levels around 50% on the three categories in the training data, handily besting the researchers' original algorithm's 34% accuracy. 

The ensembled classifier had the best score of all, around 53%. 


### Reverse engineering their algorithm
The test dataset was manually scored by a group of researchers before they published the data.

They did not include the predictions of their algorithm on the test dataset, only on the training dataset. 

I went through and made predictions on the test dataset according to the algorithm they described (all the variants of happy and frowny emoticons I could find searching through the dataset for a couple minutes). The script for this can be found in [theirAlgorithm.py](https://github.com/ClimbsRocks/nlpSentiment/blob/master/theirAlgorithm.py). The predictions created by this script can be found in [testdata.with.their.algos.predictions.csv](https://github.com/ClimbsRocks/nlpSentiment/blob/master/testdata.with.their.algos.predictions.csv)

#### Their algorithm's results

Overall, their algorithm predicted:

- 15  Negative tweets
- 461 Neutral tweets
- 22  Positive tweets

This lead to an overall accuracy score of 170 correctly predicted messages, or 34.1%.

Breaking this down further, here is how they did for each sentiment category, in terms of how many they accurately predicted in that category, out of total tweets in that category:

- Negative: 14  / 164 (8.5%)
- Neutral:  139 / 139 (100%)
- Positive: 17  / 166 (10.2%)


### Corpora Used

#### Stanford Twitter Sentiment
This is the original training data provided. Being the most closely related to our test data, this is the dataset the ensemble data is taken from as well.

#### NLTK Movie Reviews
The classic sentiment corpus, 2000 movie reviews already gathered by NLTK.

#### Assembling a custom Twitter sentiment corpus
[CrowdFlower](http://www.crowdflower.com/data-for-everyone) hosts a number of Twitter corpora that have already been graded for sentiment by panels of humans. 

I aggregated together 6 of their corpora into a single, aggregated and cleaned corpus, with consistent scoring labels across the entire corpus. The cleaned corpus contains over 45,000 documents, with sentiment graded on a 5 point scale outlined below:

- 1 is negative
- 3 is neutral
- 5 is positive

I then trained a sentiment classifier on this aggregated corpus, and used it to get predictions on the test dataset. 

#### NLTK's tweet data
NLTK has their own tweet data of 5,000 positive and 5,000 negative tweets.

#### NLTK's tweet data without emoticons
NLTK's Twitter corpus also appears to grade sentiment based solely on emoticons. While this is useful, it also allowed the algorithm to be lazy, just learning emoticons, not any of the other words. To address this, I built another version of that corpus that removed all the emoticons. Predictably, this one ended up generalizing much better to our testing data. 


### Performance of Different Corpora- Training
The models were all able to pick up on the trends in their own training corpus rather nicely.

- STS Training Corpus: 76.6%
- Movie Reviews Corpus: 78.9%
- Aggregated Twitter Corpus: 86.3%
- NLTK Twitter Corpus: 99.9% (they used purely emoticon-based sentiment scoring, which is easy for an ML - model to pick up on)
- NLTK Twitter w/o Emoticons Corpus: 78.4%


These scores all come from a holdout portion of their respective training corpus that the model was not trained on. They very closely mirror the models' cross-validation scores from the hyperparameter search, as we would expect. 


### Performance of Different Corpora- Predicting
The models' ability to generalize to the test dataset aligned pretty closely to what you would instinctively expect. A recent run produced these results.

- STS Training Corpus: 51%
- Movie Reviews Corpus: 35%
- Aggregated Twitter Corpus: 46%
- NLTK Twitter Corpus: 33%
- NLTK Twitter w/o Emoticons Corpus: 46%
- Ensembled Predictions: 53%


### Modeling Methodology
I built 5 corpora, and trained a unique model on each one. 

To train the model, I ran a hyperparameter search using RandomizedSearchCV, which heavily leverages cross-validation, over a portion of that model's respective corpus. The model's performance was then evaluated against the holdout portion of it's corpus, and the test data. 

Once the model's performance is validated, we use it to get predictions on an ensembleData portion of the original STS dataset. Every model uses this same ensembleData subset of documents. We also have each model make predictions on the test data. 

Finally, we train an ensembler model on the ensembleData, making sense of the predictions from all five stage 1 models. Once we have trained this ensembler model on the collected ensembleData, we use it to make our final predictions on the test data. 


