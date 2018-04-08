from keras.layers import Input, Dense
from keras.models import Model
from keras import optimizers
from keras.callbacks import LearningRateScheduler
from keras.callbacks import EarlyStopping
import glob
from sklearn.preprocessing import MinMaxScaler


import keras

import numpy as np
import sys
import os
import random
from sklearn import preprocessing
from random import randint
#from cop_kmeans import cop_kmeans, l2_distance
import math


def step_decay(epoch):
	if epoch < 100:
		#print "epoch %d learning rate %f" % (epoch, 0.005)
		return 0.0005
	else:
		#print "epoch %d learning rate %f" % (epoch, 0.0001)
		return 0.0001

def deepSSAEMulti(n_dim, n_hidden1, n_hidden2, n_classes):
	input_layer = Input(shape=(n_dim,))
	encoded = Dense(n_hidden1, activation='relu')(input_layer)
	encoded = Dense(n_hidden2, activation='relu', name="low_dim_features")(encoded)
	decoded = Dense(n_hidden1, activation='relu')(encoded)
	decoded = Dense(n_dim, activation='sigmoid')(decoded)
	
	classifier = Dense(n_classes, activation='softmax')(encoded)
	
	adamDino = optimizers.RMSprop(lr=0.0005)
	adamDino1 = optimizers.RMSprop(lr=0.0005)
	autoencoder = Model(inputs=[input_layer], outputs=[decoded])
	autoencoder.compile(optimizer=adamDino, loss=['mse'])
	
	ssautoencoder = Model(inputs=[input_layer], outputs=[decoded, classifier])
	ssautoencoder.compile(optimizer=adamDino1, loss=['mse','categorical_crossentropy'], loss_weights=[1., 1.])
	return [autoencoder, ssautoencoder]
	

def feature_extraction(model, data, layer_name):
	feat_extr = Model(inputs= model.input, outputs= model.get_layer(layer_name).output)
	return feat_extr.predict(data)

def learn_SingleReprSS(X_tot, idx_train, Y_train):
	n_classes = len(np.unique(Y_train))
	idx_train = idx_train.astype("int")
	X_train = X_tot[idx_train]
	encoded_Y_train = keras.utils.to_categorical(Y_train, n_classes)
	n_row, n_col = X_tot.shape
		
	n_feat = math.ceil( n_col -1)
	n_feat_2 = math.ceil( n_col * 0.5)
	n_feat_4 = math.ceil( n_col * 0.25)
	
	#print n_feat
	#print n_feat_2
	#print n_feat_4
	
	n_hidden1 = randint(n_feat_2, n_feat)
	n_hidden2 = randint(n_feat_4, n_feat_2-1)
		
	ae, ssae = deepSSAEMulti(n_col, n_hidden1, n_hidden2, n_classes)
	for i in range(200):	
		ae.fit(X_tot, X_tot, epochs=1, batch_size=16, shuffle=True, verbose=1)
		ssae.fit(X_train, [X_train, encoded_Y_train], epochs=1, batch_size=8, shuffle=True, verbose=1)			
	new_train_feat = feature_extraction(ae, X_tot, "low_dim_features")
	return new_train_feat


def learn_representationSS(X_tot, idx_train, Y_train, ens_size):
	intermediate_reprs = np.array([])
	for l in range(ens_size): 
		print "learn representation %d" % l
		embeddings = learn_SingleReprSS(X_tot, idx_train, Y_train)
		if intermediate_reprs.size == 0:
			intermediate_reprs = embeddings
		else:
			intermediate_reprs = np.column_stack((intermediate_reprs, embeddings))
	return intermediate_reprs
		
	
def normData(data):
	X = np.array(data).astype("float32")
	scaler = MinMaxScaler()
	scaler.fit(X)
	return scaler.transform(X)

#name of the .npy containing the data examples
dataset_name = sys.argv[1]

#name of the .npy containing a two column matriw with as many examples as the number of labeled examples and
#two columns: example_id, class_value
class_name = sys.argv[2]

#ensemble size
ens_size = int(sys.argv[3])

dataset = np.load(dataset_name)
dataset = normData(dataset)

idx_train_cl_val = np.load(class_name)
idx_train = idx_train_cl_val[:,0]
Y_train = idx_train_cl_val[:,1]
new_feat_ssae = learn_representationSS(dataset, idx_train, Y_train, ens_size)
outFileName = "representation.npy"
np.save(outFileName, new_feat_ssae)


