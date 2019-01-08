from string import punctuation
from os import listdir
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Flatten, LSTM
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import re
import csv
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import re
from keras.optimizers import RMSprop, SGD
from keras.utils import np_utils
import argparse
import time

# load all docs in a directory

def load_text(filename):
	file = open(filename, 'r', encoding="cp1252")
	text = file.read()
	file.close()
	return text

# turn a doc into clean tokens
def clean_doc(doc, vocab):
	doc = doc.lower()
	letters_only = re.sub("[^a-zA-Z]", " ", doc)
	# split into tokens by white space
	tokens = letters_only.split()
	# filter out tokens not in vocab
	tokens = [w for w in tokens if w in vocab]
	tokens = ' '.join(tokens)
	return tokens

# load embedding as a dict
def load_embedding(filename):
	# load embedding into memory, skip first line
	file = open(filename, 'r', encoding="UTF8")
	lines = file.readlines()
	file.close()
	# create a map of words to vectors
	embedding = dict()
	for line in lines:
		parts = line.split()
		# key is string word, value is numpy array for vector
		embedding[parts[0]] = np.asarray(parts[1:], dtype='float32')
	return embedding

# create a weight matrix for the Embedding layer from a loaded embedding
def get_weight_matrix(embedding, vocab):
	# total vocabulary size plus 0 for unknown words
	vocab_size = len(vocab) + 1
	# define weight matrix dimensions with all 0
	weight_matrix = np.zeros((vocab_size, embeddingSize))
	# step vocab, store vectors using the Tokenizer's integer mapping
	for word, i in vocab.items():
		vector = embedding.get(word)
		if vector is not None:
			weight_matrix[i] = vector
	return weight_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--verbose', help="Verbose output (enables Keras verbose output)", action='store_true', default=False)
parser.add_argument('--gpu', help="Use LSTM/GRU gpu implementation", action='store_true', default=False)
args = parser.parse_args()

verbose = 1 if args.verbose else 0
impl = 2 if args.gpu else 0

data = pd.read_csv('amazon_alexa.tsv', sep = '\t')

vocab = load_text('alexaVocab.txt');

data['clean_reviews'] = data['verified_reviews'].apply(lambda x: clean_doc(x,vocab))

tokenizer = Tokenizer()
# fit the tokenizer on the documents
tokenizer.fit_on_texts(data['clean_reviews'])
#drop empty rows
data.dropna(axis=0, how='any')

data['tokens'] = tokenizer.texts_to_sequences(data['clean_reviews'])
labels = data['rating'].apply(lambda x: x-1)
tokens = np.array(data['tokens'])

# Remove those tweets with zero length and its corresponding label
index = [idx for idx, row in enumerate(tokens) if len(row) > 0]
tokens = tokens[index]
labels = np.array(labels[index])
categoricalLabels = np_utils.to_categorical(labels, 5)
max_length = max([len(s) for s in tokens])
padded_docs = pad_sequences(tokens, maxlen=max_length, padding='post')
print(len(padded_docs))
Xtrain, Xtest, Ytrain, Ytest = train_test_split(padded_docs, labels, test_size=0.1)

# define vocabulary size (largest integer value)
vocab_size = len(tokenizer.word_index) + 1
# load embedding from file
embeddingSize = 50
raw_embedding = load_embedding('glove.6B.' + str(embeddingSize) + 'd.txt')
# get vectors in the right order
embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index)
############################################
# Model
drop = 0.0
nlayers = 2  # >= 1
RNN = LSTM  # GRU

neurons = 128

model = Sequential()
embedding_layer = Embedding(vocab_size, embeddingSize, weights=[embedding_vectors], input_length=max_length,
							trainable=False)
model.add(embedding_layer)
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
optimizer = SGD(lr=learning_rate, momentum=0.95)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

epochs = 30
batch_size = 100

yTrain_c = np_utils.to_categorical(Ytrain, 5)

history = model.fit(Xtrain, yTrain_c,
					batch_size=batch_size,
					epochs=epochs,
					validation_split=0.2,
					verbose=verbose)

############################################
# Results

##Store Plots
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

score, acc = model.evaluate(Xtest, Ytest,
							batch_size=batch_size,
							verbose=verbose)
print()
print('Test ACC=', acc)

test_pred = model.predict_classes(Xtest, verbose=verbose)

print()
print('Confusion Matrix')
print('-' * 20)
print(confusion_matrix(np.argmax(Ytest,axis=1), test_pred))
print()
print('Classification Report')
print('-' * 40)
print(classification_report(np.argmax(Ytest,axis=1), test_pred))
print()
print("Ending:", time.ctime())
