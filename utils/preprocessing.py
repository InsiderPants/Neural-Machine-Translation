import os
import numpy as np
from utils.config import MAX_SAMPLES, MAX_NUM_WORDS, EMBEDDING_DIM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def read_english(path = 'data/small_vocab_en'):
	"""
		Arguments:
			path: the path of directory of english
		
		Returns: list of english sentences
		
	"""
	english = []	#for input text
	i = 0	# temp variable
	print('Reading English Lines')
	for lines in open(path).read().split('\n'):	# Reading lines
		i+=1
		if i > MAX_SAMPLES:
			break
		english.append(lines)
	return english


def read_french(path = 'data/small_vocab_fr'):
	"""
		Arguments:
			path: the path of directory of french
		
		Returns: list of french sentence
		
	"""
	french = []		#for french output text
	french_inputs = []	# for input for decoder
	i = 0
	print('Reading French Lines')
	for lines in open(path).read().split('\n'):
		i+=1
		if i > MAX_SAMPLES:
			break
		french.append(lines + ' <eos>')	# adding end of string token 
		french_inputs.append('<sos> ' + lines)	# adding start of stinng token
	return french, french_inputs


def tokenize_english(english):	
	"""
		This function is use to tokenize english sentences and return tokenized list and dictionary of mapping
		
		Arguments:
			english: the list of english sentences
		
		Returns:
			input_sequence: list of tokenized english words that is to enter in encoder model
			word2idx_english: dictionary of maping
		
	"""
	print('Tokenizing English Texts')
	tokenizer_input = Tokenizer(num_words = MAX_NUM_WORDS)
	tokenizer_input.fit_on_texts(english)
	input_sequence = tokenizer_input.texts_to_sequences(english)
	word2idx_english = tokenizer_input.word_index
	print('Found %s unique english tokens' % len(word2idx_english))
	return input_sequence, word2idx_english


def tokenize_french(a, b):
	"""
		This function is use to tokenize french sentences and return tokenized list and dictionary of mapping
		
		Arguments:
			a: the list of french sentences for encoder model
			b: the list of french sentences for decoder model
		
		Returns:
			target_sequence: list of tokenized french words that is to enter in encoder model
			taregt_sequences_inputs: list of tokenized french words that is to enter in decoder model
			word2idx_french: dictionary of maping
		
	"""
	print('Tokenizing French Texts')
	tokenizer_output = Tokenizer(num_words = MAX_NUM_WORDS, filters = '')
	tokenizer_output.fit_on_texts(a)
	tokenizer_output.fit_on_texts(b)
	target_sequence = tokenizer_output.texts_to_sequences(a)
	target_sequence_inputs = tokenizer_output.texts_to_sequences(b)
	word2idx_french = tokenizer_output.word_index
	print('Found %s unique french tokens' % len(word2idx_french))
	return target_sequence, target_sequence_inputs, word2idx_french


def padding(a, b, c, len_input, len_target):
	"""
		This is a function to padd all the sentences.
		
		Arguments:
			a, b, c: Three lists of english, french and french_inputs(for decoder) tokenised sentences
			len_input: maximum length of sentence in english
			len_target: maximum length of sentence in french
		
		Returns:
			encoder_inputs: padded inputs for encoder model
			decoder_inputs: padded inputs for decoder model
			decoder_targets: padded targets for decoder model
			
	"""
	print('Padding..')
	encoder_inputs = pad_sequences(a, maxlen = len_input)
	decoder_inputs = pad_sequences(c, maxlen = len_target, padding='post')
	decoder_targets = pad_sequences(b, maxlen = len_target, padding='post')
	return encoder_inputs, decoder_inputs, decoder_targets


def glove_embedding(eng_dict, path='data/glove.6B.100d.txt'):
	"""
		This is a function to load GloVe Embeedding and making Embedding Matrix
		
		Arguments:
			path: path of glove txt file
			eng_dict: Dictionary of english words
		
		Returns:
			word2vec: dictionary of word to vector by GloVe
			embedding_matrix: Embedding Matrix
			
	"""
	print('Loading GloVe word embedding')
	word2vec = {}
	with open(path, encoding = 'utf-8', errors = 'ignore') as f:
		for line in f:
			values = line.split()
			word = values[0]
			vec = np.asarray(values[1:], dtype = 'float32')
			word2vec[word] = vec
	print('Found %s word vectors' % len(word2vec))
	print('Filling pre-trained embeddings...')
	num_words = min(MAX_NUM_WORDS, len(eng_dict) + 1)
	embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
	for word, i in eng_dict.items():
		if i<MAX_NUM_WORDS:
			embedding_vector = word2vec.get(word)
			if embedding_vector is not None:
				embedding_matrix[i] = embedding_vector
	return word2vec, embedding_matrix

