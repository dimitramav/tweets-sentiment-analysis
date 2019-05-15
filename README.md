# Tweets sentiment analysis
## Introduction

The assignment is about tweets' classification in three categories (positive, neutral, negative).
Firstly, train set and test set are stripped and tokenized and then, are vectorized and classified based on their sentiment.

## Strip tweets
Function clean_corpus() removes links and symbols i.e '@' and '#' and convert upper case characters to lower case.

## WordCloud
Most common words in "positive" tweets are sentimentally neutral words i.e. "tomorrow", "day", "get" and also, obviously "positive" words
like "love", "happy", "excite", "birthday", "hope" and "good".
Correspondingly, in "negative" tweets most used words are neutral but there are also many words with negative context like "don't", "can't", "fuck", "shit", "sad", "hate", "bad" and "lose".
In neutral tweets we  exclusively come across with words with neutral sentiment. 

Most used words in total dataset is a result of the most popular words of the subsets ( positive, negative, neutral). 
As it is mentioned above, predominant words are "tommorow", "go", "day", "night", "may", "come", "get". 

The popularity of neutral words and their existence in every subset is logical. In our everyday life and in our speech we use many 
words to describe situations and events and not to express feelings.

## Vectorization
We use **Bag-of-words**, **TF-IDF**, **Word2Vec** and **Doc2Vec** vectorizers to create a single vector from each tweet. The produced
vectors are combined with extra characteristics from vocabularies such as average valence, minimum and maximum valence and number of
words in each tweet. 

## Classification
We use **SVM**, **KNN** and **Round-Robin** algorithms to classify the tweets that have been produced from the mentioned vectorizers.

### Round Robin
Firstly, the train set is divide in three unique subset (positive/negative, positive/neutral, negative/neutral). Those sets will train
three Nearest Neighbor Classifiers. 
Afterwords, we produce the posteriors of trainset and testset which will be the input of KNN Classifer as train and test data respectively.

## Conclusion
The accuracy of the classifier with input the trained data is compared with the accuracy of the classifier with input test data to
ensure that overfitting does not happen. 

![alt text](https://github.com/dimitramav/tweets-sentiment-analysis/blob/master/scr/accuracy_table.png)

* SVM is more accurate than KNN, regardless the number of neighbors ( in KNN) and the vectorizer.
* SVM is faster than KNN.
* KNN with more neighbors (>10) is more efficient than with less neighbors. The number of neighbors is equivalent to the execution 
algorithm time. However, it is not recommended to increase the number of neighbors too much because the accuracy remains steady or 
even decreases, but the execution time is augmented.
* TF-IDF is more efficient than Bag of words, due to the fact that the produced vectors are decimal and more custom. Although TF-IDF
is slower.
* Added characteristics to embeddings improves importantly the accuracy of the classifiers ( 11%-12%)
* Word2Vec and Doc2Vec produce very similar results. Specifically , Word2Vec accuracy is slightly increased in KNN even if Doc2Vec 
has more accurate results in SVM. The current dataset is not appropriate to define which vectorizer is "better".
* The use of enriched Doc2Vec and Word2Vec embeddings confirms the above conclusion.

1. trainset: train2017.tsv
2. testset: test2017.tsv

Pickle files are also part of the project. The creation of these files can be skipped to save execution time.  

Collaborators: Ioannis Charamis (https://github.com/charamis) Dimitra Mavroforaki (https://github.com/dimitramav) 
