from keras.models import Model
from keras.layers import Input, Dense, Embedding, Bidirectional, RepeatVector, Concatenate, Dot, Lambda
import keras.backend as K
from keras.layers import CuDNNLSTM as LSTM
from utils.config import EMBEDDING_DIM, LATENT_DIM, LATENT_DIM_DECODER
import numpy as np

class model:
	
	def __init__(self, num_words, embedding_matrix, max_len_input, max_len_target, num_words_output):
		self.num_words = num_words
		self.embedding_matrix= embedding_matrix
		self.max_len_input = max_len_input
		self.max_len_target = max_len_target
		self.num_words_output = num_words_output
		self.attn_repeat_layer = RepeatVector(self.max_len_input)
		self.attn_concat_layer = Concatenate(axis=-1)
		self.attn_dense1 = Dense(10, activation='tanh')
		self.attn_dense2 = Dense(1, activation=self.softmax_over_time)
		self.attn_dot = Dot(axes=1) # to perform the weighted sum of alpha[t] * h[t]
		# Encoder
		# Embedding layer
		self.embedding_layer = Embedding(num_words,	
									     EMBEDDING_DIM,
										 weights=[embedding_matrix],
										 input_length=max_len_input,
										 trainable = False
										 )	
		# Input Layer of shape (max_len_input, None)  #if default it is (102, None)
		self.encoder_inputs_placeholder = Input(shape=(max_len_input,))
		self.x = self.embedding_layer(self.encoder_inputs_placeholder)
		# encoder layer, Bidirectional LSTM
		self.encoder = Bidirectional(LSTM(LATENT_DIM, return_sequences=True))
		self.encoder_output = self.encoder(self.x)
		self.encoder_model = Model(self.encoder_inputs_placeholder, self.encoder_output)
		self.decoder_embedding = Embedding(num_words_output, EMBEDDING_DIM)
		self.context_last_word_concat_layer = Concatenate(axis=2)
		self.decoder_lstm = LSTM(LATENT_DIM_DECODER, return_state=True)
		self.decoder_dense = Dense(num_words_output, activation='softmax')
		self.initial_s = Input(shape=(LATENT_DIM_DECODER,), name='s0')
		self.initial_c = Input(shape=(LATENT_DIM_DECODER,), name='c0')
		
	def softmax_over_time(self, x):
		assert(K.ndim(x) > 2)
		e = K.exp(x - K.max(x, axis=1, keepdims=True))
		s = K.sum(e, axis=1, keepdims=True)
		return e / s

	
	def one_step_attention(self, h, st_1):
		# h is sequence from each LSTM cell of encoder
		# st_1 is input for each cell in decoder
		st_1 = self.attn_repeat_layer(st_1)
		x = self.attn_concat_layer([h, st_1])
		x = self.attn_dense1(x)
		alphas = self.attn_dense2(x)
		context = self.attn_dot([alphas, h])
		return context
	
	def Seq2SeqModel(self):
		"""
			This is the actual Seq2Seq model with attention
			
			Arguments:
				
			Return: 
				model: the actual seq2seq model
		"""
		
		# Decoder
		# Input layer of shape (max_len_target, None) # if default it will be (22, None)
		decoder_inputs_placeholder = Input(shape=(self.max_len_target,))
		# Embedding Layer for french word 
		decoder_inputs_x = self.decoder_embedding(decoder_inputs_placeholder)
	
		
		# s & c will change at every LSTM cell 
		s = self.initial_s # s is the input in each lstm cell
		c = self.initial_c # c is the state comming from last lstm cell
		
		# defining output
		outputs = []
		for t in range(self.max_len_target):
			# getting context vector from attention layer
			context = self.one_step_attention(self.encoder_output, s)
			# adding custom layer to get correct input in decoder
			selector = Lambda(lambda x: x[:, t:t+1])
			xt = selector(decoder_inputs_x)		  
			# combine 
			decoder_lstm_input = self.context_last_word_concat_layer([context, xt])		
			# passing concatenated layers into lstm and egtting output
			# o is the actual output
			# s is the input in each lstm cell
			# c is the state of previous lstm cell
			o, s, c = self.decoder_lstm(decoder_lstm_input, initial_state=[s, c])		
			# final dense layer to get next word prediction
			decoder_outputs = self.decoder_dense(o)
			outputs.append(decoder_outputs)
		
		
		# 'outputs' is now a list of length Ty
		# each element is of shape (batch size, output vocab size)
		# therefore if we simply stack all the outputs into 1 tensor
		# it would be of shape T x N x D
		# we would like it to be of shape N x T x D
		
		def stack_and_transpose(x):
			# x is a list of length T, each element is a batch_size x output_vocab_size tensor
			x = K.stack(x) # is now T x batch_size x output_vocab_size tensor
			x = K.permute_dimensions(x, pattern=(1, 0, 2)) # is now batch_size x T x output_vocab_size
			return x
		
		# make it a layer
		stacker = Lambda(stack_and_transpose)
		outputs = stacker(outputs)
		
		# create the model
		model = Model(
		  inputs=[
		    self.encoder_inputs_placeholder,
		    decoder_inputs_placeholder,
		    self.initial_s, 
		    self.initial_c,
		  ],
		  outputs=outputs
		)
		return model
	
	
	def prediction(self):
		"""
			This is a function to make predictions
			
			Return:
				decoder_model: decoder model
		"""
		encoder_outputs_as_input = Input(shape=(self.max_len_input, LATENT_DIM * 2,))
		decoder_inputs_single = Input(shape=(1,))
		decoder_inputs_single_x = self.decoder_embedding(decoder_inputs_single)
		
		# no need to loop over attention steps this time because there is only one step
		context = self.one_step_attention(encoder_outputs_as_input, self.initial_s)
		
		# combine context with last word
		decoder_lstm_input = self.context_last_word_concat_layer([context, decoder_inputs_single_x])
		
		# lstm and final dense
		o, s, c = self.decoder_lstm(decoder_lstm_input, initial_state=[self.initial_s, self.initial_c])
		decoder_outputs = self.decoder_dense(o)
		
		
		# note: we don't really need the final stack and tranpose
		# because there's only 1 output
		# it is already of size N x D
		# no need to make it 1 x N x D --> N x 1 x D
		
		
		
		# create the model object
		decoder_model = Model(
		  inputs=[
		    decoder_inputs_single,
		    encoder_outputs_as_input,
		    self.initial_s, 
		    self.initial_c
		  ],
		  outputs=[decoder_outputs, s, c]
		)
		return decoder_model
	
	
	def decode_sequence(self,input_seq, fr_dict, decoder_model, inv_dict_fr):
		# Encode the input as state vectors.
		enc_out = self.encoder_model.predict(input_seq)
	
		# Generate empty target sequence of length 1.
		target_seq = np.zeros((1, 1))
	  
	    # Populate the first character of target sequence with the start character.
		target_seq[0, 0] = fr_dict['<sos>']
	
		# if we get this we break
		eos = fr_dict['<eos>']
		
		# [s, c] will be updated in each loop iteration
		s = np.zeros((1, LATENT_DIM_DECODER))
		c = np.zeros((1, LATENT_DIM_DECODER))
	
	
		# Create the translation
		output_sentence = []
		for _ in range(self.max_len_target):
			o, s, c = decoder_model.predict([target_seq, enc_out, s, c])
	        
	
			# Get next word
			idx = np.argmax(o.flatten())	
			# End sentence of EOS
			if eos == idx:
				break
	
			word = ''
			if idx > 0:
				word = inv_dict_fr[idx]
				output_sentence.append(word)
	
			# Update the decoder input
			# which is just the word just generated
			target_seq[0, 0] = idx
			
		return ' '.join(output_sentence)

