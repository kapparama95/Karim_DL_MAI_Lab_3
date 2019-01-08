import pandas
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import resample
import re
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import RNN, GRU, LSTM, BatchNormalization
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
	data1 = data[data.sentiment == 1]
	data2 = data[data.sentiment == 0]
	data1 = resample(data1,
					 replace=False,  # sample without replacement
					 n_samples=len(data2),  # to match minority class
					 random_state=123)
	df = pandas.concat([data1, data2])
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
	data = pandas.read_csv("kindle_reviews.csv", nrows=120e3)

	# Pre-process the tweet and store in a separate column
	data['clean_tweet'] = data['summary'].apply(lambda x: text_to_words(x))
	# Convert sentiment to binary
	data['sentiment'] = data['overall'].apply(lambda x: 0 if x < 3 else 3 if x == 3 else 1)
	# remove middle instance
	data = data[data.sentiment != 3]
	data['n_words'] = data['clean_tweet'].apply(lambda x: len(x.split()))
	data = data[data.n_words < 70]
	print(data['sentiment'].value_counts())

	# resampling
	data = resampleDataframe(data)
	print("new df")
	print(data['sentiment'].value_counts())

	# Join all the words in review to build a corpus
	all_text = ' '.join(data['clean_tweet'])
	words = all_text.split()

	# Convert words to integers
	counts = Counter(words)

	# numwords = 5000  # Limit the number of words to use
	vocab = sorted(counts, key=counts.get, reverse=True)  # [:numwords]
	numwords = len(vocab)
	vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

	tweet_ints = []
	for each in data['clean_tweet']:
		tweet_ints.append([vocab_to_int[word] for word in each.split() if word in vocab_to_int])

	# Create a list of labels
	labels = np.array(data['sentiment'])

	# Find the number of tweets with zero length after the data pre-processing
	tweet_len = Counter([len(x) for x in tweet_ints])
	print("Zero-length reviews: {}".format(tweet_len[0]))
	print("Maximum tweet length: {}".format(max(tweet_len)))

	# Remove those tweets with zero length and its corresponding label
	tweet_idx = [idx for idx, tweet in enumerate(tweet_ints) if len(tweet) > 0]
	labels = labels[tweet_idx]
	data = data.iloc[tweet_idx]
	tweet_ints = [tweet for tweet in tweet_ints if len(tweet) > 0]

	seq_len = max(tweet_len)
	features = np.zeros((len(tweet_ints), seq_len), dtype=int)
	for i, row in enumerate(tweet_ints):
		features[i, -len(row):] = np.array(row)[:seq_len]

	train_x, testVal_x, train_y, testVal_y = train_test_split(features, labels, test_size=0.2)
	val_x, test_x, val_y, test_y = train_test_split(testVal_x, testVal_y, test_size=0.5)

	print("\t\t\tFeature Shapes:")
	print("Train set: \t\t{}".format(train_x.shape),
		  "\nValidation set: \t{}".format(val_x.shape),
		  "\nTest set: \t\t{}".format(test_x.shape))

	print("Train set: \t\t{}".format(train_y.shape),
		  "\nValidation set: \t{}".format(val_y.shape),
		  "\nTest set: \t\t{}".format(test_y.shape))

	############################################
	# Model
	drop = 0.0
	nlayers = 2  # >= 1
	RNN = GRU  # GRU

	neurons = 256
	embedding = 40

	model = Sequential()
	model.add(Embedding(numwords + 1, embedding, input_length=seq_len))

	if nlayers == 1:
		model.add(RNN(neurons, implementation=impl, recurrent_dropout=drop))
	else:
		model.add(RNN(neurons, implementation=impl, recurrent_dropout=drop, return_sequences=True))
		for i in range(1, nlayers - 1):
			model.add(RNN(neurons, recurrent_dropout=drop, implementation=impl, return_sequences=True))
		model.add(RNN(neurons, recurrent_dropout=drop, implementation=impl))

	model.add(Dense(1))
	model.add(Activation('sigmoid'))
	print(model.summary())
	############################################
	# Training

	learning_rate = 0.01
	optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

	epochs = 120
	batch_size = 128

	history = model.fit(train_x, train_y,
						batch_size=batch_size,
						epochs=epochs,
						validation_data=(val_x, val_y),
						verbose=verbose)

	############################################
	# Results
	# Confusion Matrix
	from sklearn.metrics import classification_report, confusion_matrix
	import numpy as np

	# Compute probabilities
	Y_pred = model.predict(test_x)
	# Assign most probable label

	y_pred = list()
	for v in range(0, len(Y_pred)):
		if Y_pred[v] < 0.5:
			y_pred.append(0)
		else:
			y_pred.append(1)
	# Plot statistics
	print('Analysis of results')
	print(classification_report(test_y, y_pred))
	print(confusion_matrix(test_y, y_pred))

	# Accuracy plot
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	# No validation loss in this example
	plt.legend(['train', 'val'], loc='upper left')
	plt.savefig('output/model_accuracy.pdf')
	plt.close()
	# Loss plot
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'val'], loc='upper left')
	plt.savefig('output/model_loss.pdf')

	from keras.models import model_from_json

	nn_json = model.to_json()
	with open('output/nn.json', 'w') as json_file:
		json_file.write(nn_json)
	weights_file = "output/weights" + ".hdf5"
	model.save_weights(weights_file, overwrite=True)