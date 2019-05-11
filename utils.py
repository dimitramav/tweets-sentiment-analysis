import re
import string
import pickle
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import  WordNetLemmatizer
from nltk import word_tokenize
import numpy
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import svm

def clean_corpus(corpus):
	translate_table = dict((ord(char), None) for char in string.punctuation)
	cleaned_corpus = []
	for tweet in corpus:
		tweet = re.sub(r"http\S+", "", tweet)  # remove link
		tweet = ' '.join([word for word in tweet.split(' ') if not word.startswith('@')])
		tweet = tweet.translate(translate_table)  # remove symbols
		tweet = tweet.lower()
		cleaned_corpus.append(tweet)
	return cleaned_corpus

def tokenize(cleaned_corpus):
	tokens = [word_tokenize(tweet) for tweet in cleaned_corpus]
	return tokens

def get_wordnet_pos(word):
	# Map POS tag to first character lemmatize() accepts
	tag = pos_tag([word])[0][1][0].upper()
	tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
	return tag_dict.get(tag, wordnet.NOUN)

def lemmatize(tokens):
	lemmatizer = WordNetLemmatizer()
	tweets = []
	for token_list in tokens:
		lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in token_list]
		tweets.append(lemmatized)
	return tweets

def save_to_pickle(filename, data):
	with open(filename+'.pickle','wb') as handle:
		pickle.dump(data,handle,protocol = pickle.HIGHEST_PROTOCOL)
	#na dw ti ginetai se periptwsi sfalmatos

def load_from_pickle(filename):
	with open(filename+'.pickle','rb') as handle:
		return pickle.load(handle)
	#na dw ti ginetai se periptwsi sfalmatos

def embeddings(tweets, model):
	word_embeddings = []
	for tweet in tweets:
		vec = numpy.array(model[tweet[0]])
		for word in tweet[1:]:
			vec = vec + numpy.array(model[word])
			vec = vec / len(tweet) #na dw an len(tweet)==0 einai dynato
			word_embeddings.append(vec)
	return word_embeddings

def knn_classification(X_train, X_test, Y_train, Y_test):
	knn = KNeighborsClassifier(n_neighbors=10)
	knn.fit(X_train, Y_train)
	Y_pred = knn.predict(X_test)
	return metrics.accuracy_score(Y_test, Y_pred)

def svm_classification(X_train, X_test, Y_train, Y_test):
	clf = svm.SVC(gamma='scale')
	clf.fit(X_train, Y_train)
	Y_pred = clf.predict(X_test)
	return metrics.accuracy_score(Y_test, Y_pred)