#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 12:08:35 2017

@author: steffen
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import edward as ed
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense, RepeatVector
from keras.layers import LSTM , Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from edward.models import Bernoulli, Normal
from edward.util import Progbar
from tensorflow.python.framework import ops      #resets TF Graph to set the seed
ops.reset_default_graph()
sess = tf.InteractiveSession()


# Variables
run=1
np.random.seed(7)
plt.style.use(['seaborn-darkgrid'])
batch_size = 4
look_back = 9   #timesteps
featurenumber=1 #for multivariat time series
epochs=100
np.random.seed(7)
d=5  # latent dim

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

# Choose dataset
if run==1:
   dataset = np.cos(np.arange(1000)*(20*np.pi/1000))[:,None]
   plt.plot(dataset)
   #scaler = MinMaxScaler(feature_range=(0, 1))
#   dataset = scaler.fit_transform(dataset)
   print "Dataset", run
elif run==2:
   dataset = read_csv('international-airline-passengers.csv', usecols=[1],
   engine='python', skipfooter=3)
   dataset = dataset.values
   dataset = dataset.astype('float32')
   print "Dataset", run
   plt.plot(dataset)
   scaler = MinMaxScaler(feature_range=(0, 1))
   dataset = scaler.fit_transform(dataset)


train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainY= np.reshape(trainY, (trainY.shape[0], 1))
testY=np.reshape(testY, (testY.shape[0], 1))

trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], featurenumber))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], featurenumber))


# INFERENCE MODEL
# Define a subgraph of the variational model, corresponding to a
# minibatch of size M.
x_ph = tf.placeholder(tf.int32, [batch_size, look_back, featurenumber])

Inference_Model = LSTM(4, batch_input_shape=(batch_size, look_back, featurenumber), stateful=True, unroll=True)(tf.cast(x_ph, tf.float32))

qz = Normal(loc=Dense(d)(Inference_Model),
            scale=Dense(d, activation='softplus')(Inference_Model))

# RECOGNITION MODEL
# Define a subgraph of the full model, corresponding to a minibatch of
# size M.
z     = Normal(loc=tf.zeros([batch_size, d]), scale=tf.ones([batch_size, d]))
z_rep = RepeatVector(look_back)(z)
x_i   = LSTM(4, batch_input_shape=(batch_size, look_back, d), stateful=True, unroll=True)(z_rep)
x   = Dense(1)(x_i)
#x = Bernoulli(logits=Dense(look_back)(hidden1))



# Bind p(x, z) and q(z | x) to the same TensorFlow placeholder for x.
inference = ed.KLqp({z: qz}, data={x: x_ph})
optimizer = tf.train.RMSPropOptimizer(0.01, epsilon=1.0)
inference.initialize(optimizer=optimizer)



