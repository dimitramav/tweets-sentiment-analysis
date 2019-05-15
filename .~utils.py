import re
import string
import pickle
import numpy
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import  WordNetLemmatizer
from nltk import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn import svm
from statistics import mean

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

def create_word_embeddings(tweets, model):
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

def create_doc_embeddings(corpus):
	tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(corpus)]
	model = Doc2Vec(size=200, min_alpha=0.00025, min_count=1, dm =1)
	model.build_vocab(tagged_data)
	max_epoch = 20
	for epoch in range(max_epoch):
		model.train(tagged_data, total_examples=model.corpus_count, epochs=model.iter)
	return model.docvecs

def add_characteristics(lexica, tweets):
	characteristics = [[] for i in range(len(tweets))]
	for tweet in range(len(tweets)):   #for each tweet
		characteristics[tweet].append(len(tweets[tweet])) #length of each tweet
		for lexicon in lexica:
			tweet_sentiments = [lexicon.get(token,[0])[0] for token in tweets[tweet]]
			max_valence = max(tweet_sentiments)
			min_valence = min(tweet_sentiments)
			average = mean(tweet_sentiments)
			if len(tweet_sentiments) > 1:
				average_half1 = mean(tweet_sentiments[:len(tweet_sentiments)//2])
				average_half2 = mean(tweet_sentiments[len(tweet_sentiments)//2:])
			else:
				average_half1 = len(tweet_sentiments)
				average_half2 = 0
			characteristics[tweet].extend((max_valence, min_valence, average, average_half1, average_half2))
	return characteristics

def clean_tweets(tweets):
	cleaned_tweets = lemmatize(tokenize(clean_corpus(tweets)))
	return [" ".join(str(word) for word in tweet) for tweet in cleaned_tweets]

def create_posteriors(train_tweets, train_as_test_tweets, test_tweets, labels, n_neighbors):
	trainset = clean_tweets(train_tweets)
	testset = clean_tweets(test_tweets)
	train_as_test_set = clean_tweets(train_as_test_tweets)

	vectorizer = CountVectorizer()
	trainset_BOW = vectorizer.fit_transform(trainset)
	testset_BOW = vectorizer.transform(testset)
	train_test_set_BOW = vectorizer.transform(train_as_test_set)
	knn = KNeighborsClassifier(n_neighbors)
	knn.fit(trainset_BOW, labels)

	train_posteriors = knn.predict_proba(train_test_set_BOW)
	test_posteriors = knn.predict_proba(testset_BOW)

	return {'train':train_posteriors, 'test':test_posteriors}