# nlpPredictions
> Predicting the sentiment of tweets using NLP.

## Installation

1. `pip install -r requirements.txt`
1. If this fails to install scikit-learn properly, you may have to `pip install numpy` and `pip install scipy`
1. Open python on the command line
1. `import nltk`
1. `ntlk.download()` This will open a GUI.
1. Follow prompts to download everything. This will download 1.8GB of material.
1. 


### Reverse engineering their algorithm
The test dataset was manually scored by a group of researchers before they published the data.

Oddly, they did not include the predictions of their algorithm on the test dataset, only on the training dataset. 

I went through and made predictions on the test dataset according to the algorithm they described (all the variants of happy and frowny emoticons I could find searching through the dataset for a couple minutes).

### Their algorithm's results

Overall, their algorithm predicted:

- 14  Negative tweets
- 463 Neutral tweets
- 21  Positive tweets

This lead to an overall accuracy score of 168 correctly predicted messages, or 33.7%.

Breaking this down further, here is how they did for each sentiment category:

Negative: 13  / 164 (7.9%)
Neutral:  139 / 139 (100%)
Positive: 16  / 166 (9.6%)


