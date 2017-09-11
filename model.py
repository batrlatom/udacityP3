import keras
import keras.models as models

from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Lambda
from keras.layers import BatchNormalization,Input
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.layers.convolutional import Convolution2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Convolution2D
from keras.callbacks import ModelCheckpoint
from utils import INPUT_SHAPE, batch_generator, resize_normalize, pandas_split
from keras.layers import Cropping2D
from keras.models import load_model
from keras import metrics


import sklearn.metrics as metrics

import cv2
import numpy as np
import json

import math
import h5py
import glob
from tqdm import tqdm
import scipy
from scipy import misc
import argparse, os
import pandas as pd

import matplotlib.pyplot as plt
import sklearn
plt.ion()


data_file_name = 'driving_log.csv'
batch_size = 32
nb_epoch = 1000
data_df = None

######################################################################################
#	load training data from all data directories under the main data dir
######################################################################################
def load_training_data(args):

	first = True 
	for data_dir in (next(os.walk(args.data_dir))[1]):
		if first:
			data_df = pd.read_csv(os.path.join(os.path.join(args.data_dir, data_dir), data_file_name), sep=',', skipinitialspace=True, names = ["center", "left", "right", "steering", "throtle", "break", "speed"])
			first = False
		else:
			data_df_holder = pd.read_csv(os.path.join(os.path.join(args.data_dir, data_dir), data_file_name), sep=',', skipinitialspace=True, names = ["center", "left", "right", "steering", "throtle", "break", "speed"])
		
	#load image paths from csv and load all images accordingly
	X = data_df[['center', 'left', 'right']].values
	y = data_df['steering'].values

	# return splitted dataset ( 0.8/0.2 )
	return pandas_split(X, y)
	



######################################################################################
#	Will train dataset, provide 2 batch generators - for training and validation
#	Each epoch is validated and if validation_acc is better than last best, it will save model into model.h5 file 
######################################################################################
def train_model(model, args, X_train, y_train, X_valid, y_valid):
	#model = load_model('model.h5')
	checkpoint = ModelCheckpoint('model.h5', monitor='val_acc', verbose=1, save_best_only = True, mode='auto')
	callbacks_list = [checkpoint]

	model.compile(loss = 'mean_squared_error', optimizer = Adam(lr=1e-4), metrics=['acc'])

	
	# there are two nested generators, one is for training data and second is for validation data
	model.fit_generator(batch_generator(args.data_dir, X_train, y_train, batch_size, True),
                        len(X_train)/batch_size,
                        nb_epoch,
                        max_q_size=1,
                        validation_data=batch_generator(args.data_dir, X_valid, y_valid, batch_size, False),
                        nb_val_samples=len(X_valid)/batch_size,
                        callbacks=[checkpoint],
                        verbose=1)
	

#####################################################################################
#	create NVIDIA Drivenet model as provided at https://arxiv.org/pdf/1604.07316.pdf
#	we will change it slightly and add elu activation - should work little better
#####################################################################################
def create_model(args):
	model = Sequential()
	model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape = INPUT_SHAPE))
	model.add(Lambda(resize_normalize, output_shape=(66, 200, 3)))
	model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='elu'))
	model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='elu'))
	model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='elu'))
	model.add(Convolution2D(64, 3, 3, activation='elu'))
	model.add(Convolution2D(64, 3, 3, activation='elu'))
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(100, activation = 'elu'))
	model.add(Dense(50, activation = 'elu'))
	model.add(Dense(10, activation = 'elu'))
	model.add(Dense(1, activation = 'tanh'))
	model.summary()

	return model

######################################################################################
# main function
######################################################################################
if __name__ == '__main__':

	# get parameters
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("data_dir", help="Path to the folder of run images",
	                    type=str)
	
	args = parser.parse_args()


	# load training data and split them to train and validation dataset
	X_train, X_valid, y_train, y_valid = load_training_data(args)

	# create and train model
	model = create_model(args)
	train_model(model, args, X_train, y_train, X_valid, y_valid)
	

