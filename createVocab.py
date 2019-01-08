from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
import re
import csv

# load doc into memory
def process_docs(filename, vocab):
	with open(filename,encoding='utf8') as f:
		reader = csv.reader(f, delimiter='\t')
		next(reader)
		for row in reader:
			vocab.update(clean_doc(row[3]))

# turn a doc into clean tokens
def clean_doc(doc):
	# split into tokens by white space
	doc = doc.lower()
	letters_only = re.sub("[^a-zA-Z]", " ", doc)
	# split into tokens by white space
	tokens = letters_only.split()
	#filter out stopwords
	stop_words = set(stopwords.words('english'))
	tokens = [w for w in tokens if not w in stop_words]

	# filter out short tokens
	tokens = [word for word in tokens if len(word) > 2]
	return tokens



# define vocab
vocab = Counter()
# add all docs to vocab
process_docs('amazon_alexa.tsv', vocab)

# print the size of the vocab
print(len(vocab))
# print the top words in the vocab
print(vocab.most_common(50))
# keep tokens with a min occurrence
min_occurane = 2
tokens = [k for k,c in vocab.items() if c >= min_occurane]
print(len(tokens))
# save list to file
def save_list(lines, filename):
	# convert lines to a single blob of text
	data = '\n'.join(lines)
	# open file
	file = open(filename, 'w')
	# write text
	file.write(data)
	# close file
	file.close()

# save tokens to a vocabulary file
save_list(tokens, 'alexaVocab.txt')