import pandas as pd 
import numpy as np 

labels = ["Survived","Pclass","Sex","Age","SibSp","Parch","Fare"]	

def load_datasets():
	df_train = pd.read_csv("train.csv", usecols = labels)
	df_test = pd.read_csv("test.csv", usecols = ["Pclass","Sex","Age","SibSp","Parch","Fare"])
	mean_age_train = df_train['Age'].mean()
	mean_age_test = df_test['Age'].mean()
	df_train = df_train.fillna(mean_age_train)
	df_test = df_test.fillna(mean_age_test)
	X_train = np.array(df_train[["Pclass","Sex","Age","SibSp","Parch","Fare"]]).T
	Y_train = np.array(df_train[["Survived"]]).T
	Y_train = np.squeeze(Y_train)
	X_test = np.array(df_test[["Pclass","Sex","Age","SibSp","Parch","Fare"]]).T
	
	return X_train, Y_train, X_test


def pre_process_data(X_train, X_test):
	for i in range(X_train.shape[1]):
		if(X_train[1][i] == "male"):
			X_train[1][i] = 0
		else:
			X_train[1][i] = 1

	for i in range(X_test.shape[1]):
		if(X_test[1][i] == "male"):
			X_test[1][i] = 0
		else:
			X_test[1][i] = 1

	for i in range(X_train.shape[0]):
		mean = X_train[i, :].mean()
		std =  X_train[i, :].std()
		X_train[i,:] = (X_train[i,:] - mean)/std
	for i in range(X_test.shape[0]):
		mean = X_test[i, :].mean()
		std =  X_test[i, :].std()
		X_test[i,:] = (X_test[i,:] - mean)/std

	return X_train, X_test

def sigmoid(z):
    s = 1/(1 + np.exp(-z))
    return s