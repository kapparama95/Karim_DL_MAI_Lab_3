
import pandas
from sklearn.metrics import confusion_matrix, classification_report
import re
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM
from keras.optimizers import RMSprop, SGD
from keras.utils import np_utils
from collections import Counter
import argparse
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import resample




def tweet_to_words(raw_tweet):
	"""
	Only keeps ascii characters in the tweet and discards @words

	:param raw_tweet:
	:return:
	"""
	letters_only = re.sub("[^a-zA-Z]", " ", raw_tweet)
	words = letters_only.lower().split()
	return " ".join(words)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--verbose', help="Verbose output (enables Keras verbose output)", action='store_true', default=False)
	parser.add_argument('--gpu', help="Use LSTM/GRU gpu implementation", action='store_true', default=False)
	args = parser.parse_args()

	verbose = 1 if args.verbose else 0
	impl = 2 if args.gpu else 0

	print("Starting:", time.ctime())

	############################################
	# Data

	#Tweet = pandas.read_csv("Airlines.csv")
	data = pandas.read_csv("amazon_alexa.tsv", delimiter="\t")

	# Pre-process the tweet and store in a separate column
	data['clean_tweet'] = data['verified_reviews'].apply(lambda x: tweet_to_words(x))
	# Convert sentiment to binary
	data['sentiment'] = data['rating'].apply(lambda x: x - 1)
	print(data['sentiment'].value_counts())

	# resampling
	data4 = data[data.sentiment == 4]
	df = resample(data4,
				  replace=False,  # sample without replacement
				  n_samples=1000,  # to match minority class
				  random_state=123)
	for i in range(0, 4):
		ds = data[data.sentiment == i]
		downsampled = resample(ds,
							   replace=True,  # sample without replacement
							   n_samples=1000,  # to match minority class
							   random_state=123)
		df = pandas.concat([df, downsampled])
	# Join all the words in review to build a corpus
	data = df
	all_text = ' '.join(data['clean_tweet'])
	words = all_text.split()

	# Convert words to integers
	counts = Counter(words)

	#numwords = 5000  # Limit the number of words to use
	vocab = sorted(counts, key=counts.get, reverse=True)#[:numwords]
	numwords=len(vocab)
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

	train_x, testVal_x,train_y, testVal_y = train_test_split(features,labels,test_size=0.3)
	val_x, test_x, val_y, test_y  = train_test_split(testVal_x,testVal_y,test_size=0.5)

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
	nlayers = 3  # >= 1
	RNN = LSTM  # GRU

	neurons = 128
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

	model.add(Dense(5))
	model.add(Activation('softmax'))

	############################################
	# Training

	learning_rate = 0.01
	optimizer = SGD(lr=learning_rate)
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

	epochs = 10
	batch_size = 100

	train_y_c = np_utils.to_categorical(train_y, 5)
	val_y_c = np_utils.to_categorical(val_y, 5)

	class_weight = {0: 1.,
					1: 1.,
					2: 1.,
					3: 1.,
					4: 1.}

	history = model.fit(train_x, train_y_c,
			  batch_size=batch_size,
			  epochs=epochs,
			  validation_data=(val_x, val_y_c),
			  verbose=verbose, class_weight=class_weight)

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
	with open('nn.json', 'w') as json_file:
		json_file.write(nn_json)
	weights_file = "weights" + ".hdf5"
	model.save_weights(weights_file, overwrite=True)