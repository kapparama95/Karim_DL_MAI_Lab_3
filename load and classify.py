import pandas
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import resample
import re
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import SimpleRNN, GRU, LSTM, BatchNormalization
from keras.optimizers import RMSprop, SGD
from keras.utils import np_utils
from collections import Counter
import argparse
import time
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def text_to_words(raw_tweet):
	"""
	Only keeps ascii characters in the tweet and discards @words

	:param raw_tweet:
	:return:
	"""
	try:
		letters_only = re.sub("[^a-zA-Z]", " ", raw_tweet)
		words = letters_only.lower().split()
	except:
		print(raw_tweet)
		return ""
	return " ".join(words)


def resampleDataframe(data):
	data4 = data[data.sentiment == 4]
	nsamples = len(data[data.sentiment == 0])

	df = resample(data4,
				  replace=False,  # sample without replacement
				  n_samples=nsamples,  # to match minority class
				  random_state=123)
	for i in range(0, 4):
		ds = data[data.sentiment == i]
		downsampled = resample(ds,
							   replace=False,  # sample without replacement
							   n_samples=nsamples,  # to match minority class
							   random_state=123)
		df = pandas.concat([df, downsampled])
	return df


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--verbose', help="Verbose output (enables Keras verbose output)", action='store_true',
						default=False)
	parser.add_argument('--gpu', help="Use LSTM/GRU gpu implementation", action='store_true', default=False)
	args = parser.parse_args()

	verbose = 1 if args.verbose else 0
	impl = 2 if args.gpu else 0

	print("Starting:", time.ctime())

	############################################
	# Data

	# Tweet = pandas.read_csv("Airlines.csv")
	data = pandas.read_csv("kindle_reviews.csv", nrows=30e3, encoding='utf8')
	# Pre-process the tweet and store in a separate column
	data['clean_text'] = data['reviewText'].apply(lambda x: text_to_words(x))
	# Convert sentiment to binary
	data['sentiment'] = data['overall'].apply(lambda x: x - 1)
	data['n_words'] = data['clean_text'].apply(lambda x:len(x.split()))
	data = data[data.n_words < 70]
	print(data['sentiment'].value_counts())
	print('new mean: ' + str(np.mean(data['n_words'])))

	# resampling
	#data = resampleDataframe(data)
	print("new df")
	print(data['sentiment'].value_counts())

	# Join all the words in review to build a corpus
	all_text = ' '.join(data['clean_text'])
	words = all_text.split()

	# Convert words to integers
	counts = Counter(words)

	numwords = 8000  # Limit the number of words to use
	vocab = sorted(counts, key=counts.get, reverse=True)[:numwords]
	numwords = len(vocab)
	vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

	tweet_ints = []
	for each in data['clean_text']:
		tweet_ints.append([vocab_to_int[word] for word in each.split() if word in vocab_to_int])
	# Create a list of labels
	labels = np.array(data['sentiment'])

	# Find the number of tweets with zero length after the data pre-processing
	text_len = Counter([len(x) for x in tweet_ints])
	print("Zero-length reviews: {}".format(text_len[0]))
	print("Maximum tweet length: {}".format(max(text_len)))

	# Remove those tweets with zero length and its corresponding label
	tweet_idx = [idx for idx, tweet in enumerate(tweet_ints) if len(tweet) > 0]
	labels = labels[tweet_idx]
	data = data.iloc[tweet_idx]
	tweet_ints = [tweet for tweet in tweet_ints if len(tweet) > 0]

	seq_len = max(text_len)
	features = np.zeros((len(tweet_ints), seq_len), dtype=int)
	for i, row in enumerate(tweet_ints):
		features[i, -len(row):] = np.array(row)[:seq_len]

	train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.2)

	print("\t\t\tFeature Shapes:")
	print("Train set: \t\t{}".format(train_x.shape),
		  "\nTest set: \t\t{}".format(test_x.shape))

	print("Train set: \t\t{}".format(train_y.shape),
		  "\nTest set: \t\t{}".format(test_y.shape))

	############################################

	############################################
	# Load model
	from keras.models import model_from_json

	# Loading model and weights
	json_file = open('nn.json', 'r')
	nn_json = json_file.read()
	json_file.close()
	model = model_from_json(nn_json)
	model.load_weights("weights.hdf5")

	############################################
	# Results
	# Confusion Matrix
	from sklearn.metrics import classification_report, confusion_matrix
	import numpy as np

	# Compute probabilities
	Y_pred = model.predict(test_x)
	# Assign most probable label
	y_pred = np.argmax(Y_pred, axis=1)

	# Plot statistics
	print('Analysis of results')
	target_names = ['1', '2', '3', '4', '5']
	print(classification_report(test_y, y_pred, target_names=target_names))
	print(confusion_matrix(test_y, y_pred))

