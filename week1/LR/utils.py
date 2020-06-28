import numpy as np 
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer 

tokenizer = Tokenizer()


def load_dataset():
	"""Outputs the training and test sets in form of an numpy array"""

	train = np.array(pd.read_csv("train.csv").fillna("NotAvailable"))
	test = np.array(pd.read_csv("test.csv").fillna("NotAvailable"))
	x_train = train[:,1:-1]
	y_train = train[:,-1]
	x_test = test[:,1:]

	return x_train, y_train, x_test

def preprocess_dataset(data, test_data):
	"""
	pre-processes the data, all words are converted to tokens and the data is normalized
	Arguments : data -> numpy array 
	Returns : Normalised data
				word_index -> dict of all tokens

	"""
	c = 1

	tokenised_data = []
	for i in range(data.shape[1]):
		l = np.array(data[:,i])
		try:
			tokenizer.fit_on_texts(l)
		except:
			print(end="")

	word_index = tokenizer.word_index
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			try:
				data[i][j] = word_index[data[i][j].lower()]
			except:
				print(end = "")
	for i in range(data.shape[1]):
		try:
			mean = data[:,i].mean()
			std = data[:,i].std()
			data[:,i] = (data[:,i] - mean)/std
		except:
			print(end = "")

	for i in range(test_data.shape[0]):
		for j in range(test_data.shape[1]):
			try:
				test_data[i][j] = word_index[test_data[i][j].lower()]
			except:
				print(end = "")
	for i in range(test_data.shape[1]):
		try:
			mean = test_data[:,i].mean()
			std = test_data[:,i].std()
			test_data[:,i] = (test_data[:,i] - mean)/std
		except:
			print(end = "")


	return data, test_data, word_index



