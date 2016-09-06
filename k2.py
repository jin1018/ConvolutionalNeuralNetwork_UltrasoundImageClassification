#Sources Used : https://keras.io (Keras website)
from __future__ import print_function
from PIL import Image # Python Imaging Library (PIL)

import sys
import os
import time
import copy

import numpy as np
np.set_printoptions(threshold='nan')#otherwise prints the truncated numpy array [0 0 0 ..., 0 0 0]

import theano
import theano.tensor as T

import keras as keras
#from keras.datasets import mnist
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

#to use tensorflow backend instead of theano
#useTensorflow = 'KERAS_BACKEND=tensorflow python -c "from keras import backend; print(backend._BACKEND)"'
#os.system(useTensorflow)

def load_dataset():
	X_all_data=[]
	Y_all_data=[]

	"""ex) MNIST
	X_train.shape is (50000, 1, 28, 28), 
	to be interpreted as: 50,000 images of 1 channel, 28 rows and 28 columns each. 

	y_train.shape is simply (50000,): a vector same length of X_train
	"""

	#I have total 1367*2 images: one from trainYes folder, other from trainNO folder

	#X_all_data
	for subject in range(1,48):#48
		for imageNum in range(1,121):#121
			#first add the images that are neck cells. Add 1 to y
			try:
				channel_grey=[]

				nameOfFile ='trainYesPadd120/imagePad' + str(subject) + "_" + str(imageNum) + ".png"
				image = Image.open(nameOfFile)
				imarray = np.array(image)
				imlist=imarray.tolist()
				image.close()
				#print(imarray)
				#print (imarray.shape)
				
				channel_grey.append(imlist)
				X_all_data.append(channel_grey)
				Y_all_data.append(1)
				#print (len(imlist))
				#print (len(imlist[0]))
				#print (len(channel_grey))
				#print (len(X_all_data))

			except:#Some images are not marked with a mask
				pass

			#then add the images that are NOT neck cells. Add 0 to y
			try:
				channel_grey=[]

				nameOfFile ='trainNOPadd120/no-imagePad' + str(subject) + "_" + str(imageNum) + ".png"
				image = Image.open(nameOfFile)
				imarray = np.array(image)
				imlist=imarray.tolist()
				image.close()
				#print(imarray)
				#print (imarray.shape)
				
				channel_grey.append(imlist)
				X_all_data.append(channel_grey)
				Y_all_data.append(0)
				#print (len(imlist))
				#print (len(imlist[0]))
				#print (len(channel_grey))
				#print (len(X_all_data))

			except:
				pass

	#print(len(X_all_data))#2732
			
	num_trainSet= int(len(X_all_data)*0.6)#0.6
	num_valSet= int(len(X_all_data)*0.2)#0.2
	#num_testSet= len(X_all_data) -  num_trainSet - num_valSets
	#print(num_trainSet)#1639
	#print(num_valSet)#546
	#print(num_testSet)#547
	""" Using deepcopy takes a very long time (and unnecessary in this case)
	print("creating X_train, y_train, X_val, y_val, X_test, y_test...(takes a few minutes)")
	print("1")
	X_train=copy.deepcopy(X_all_data)
	X_train=X_train[:num_trainSet-1]#[:1638]
	print("2")
	X_val=copy.deepcopy(X_all_data)
	X_val=X_val[num_trainSet:num_trainSet+num_valSet-1]#[1639:2186]
	print("3")
	X_test=copy.deepcopy(X_all_data)
	X_test=X_test[num_trainSet+num_valSet:]
	print("4")
	y_train=copy.deepcopy(Y_all_data)
	y_train=y_train[:num_trainSet-1]
	print("5")
	y_val=copy.deepcopy(Y_all_data)
	y_val=y_val[num_trainSet:num_trainSet+num_valSet-1]
	print("6")
	y_test=copy.deepcopy(Y_all_data)
	y_test=y_test[num_trainSet+num_valSet:]
	"""
	#X_train=copy.deepcopy(X_all_data)
	X_train=X_all_data[:num_trainSet-1]#[:1638]
	#X_val=copy.deepcopy(X_all_data)
	X_val=X_all_data[num_trainSet:num_trainSet+num_valSet-1]#[1639:2186]
	#X_test=copy.deepcopy(X_all_data)
	X_test=X_all_data[num_trainSet+num_valSet:]
	#y_train=copy.deepcopy(Y_all_data)
	y_train=Y_all_data[:num_trainSet-1]
	#y_val=copy.deepcopy(Y_all_data)
	y_val=Y_all_data[num_trainSet:num_trainSet+num_valSet-1]
	#y_test=copy.deepcopy(Y_all_data)
	y_test=Y_all_data[num_trainSet+num_valSet:]

	X_train=np.array(X_train)
	y_train=np.array(y_train)
	X_val=np.array(X_val)
	y_val=np.array(y_val)
	X_test=np.array(X_test)
	y_test=np.array(y_test)


	return X_train, y_train, X_val, y_val, X_test, y_test

"""Build a Convolutional Neural Network"""
def buildCNN(X_train, y_train, X_val, y_val,nb_classes,filepath):

	# convert class vectors to binary class matrices
	y_train = np_utils.to_categorical(y_train, nb_classes)
	y_val = np_utils.to_categorical(y_val, nb_classes)
	#or else I get the error:
	#Exception: Error when checking model target: expected activation_4 to have shape (None, 2) 
	#but got array with shape (1638, 1)

	#build model
	grey_channel=1#greyscale = 1, rgb = 3
	model, batch_size, nb_epoch = create_model(grey_channel)

	model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

	model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_val, y_val))

	score = model.evaluate(X_val, y_val, verbose=0)
	print('crossVal score:', score[0])
	print('crossVal accuracy:', score[1])

	"""ETA: Estimated Time of Arrival"""

	#Save trained model
	#model.save_weights(filepath) #save a Keras model into a single HDF5 file
	#Can also save to JSON / YAML files(human-readable)
	#ex)
	model.save_weights(filepath)  # creates a HDF5 file 'my_model.h5'
	print("saved trained model")
	del model  # deletes the existing model

def create_model(channel):"""should test out with parameters here"""
	#channel=1: grey, channel=3: rgb
	batch_size = 30
	nb_classes = 2
	nb_epoch = 1

	# input image dimensions
	img_rows =X_train.shape[2]#120
	img_cols =X_train.shape[3]#120
	#print("img_rows: " + str(img_rows))#120
	#print("img_cols: " + str(img_cols))#120

	# number of convolutional filters to use(dimensionality of the output)
	#nb_filters1 = 100
	#nb_filters2 = 20
	
	# size of pooling area for max pooling
	nb_pool = 2

	# convolution kernel("window" in cnn) size
	#nb_conv1 = 15
	#nb_conv2 = 5
	

	#dropout: prevent overfitting
	# randomly setting a fraction p of input units to 0 at each update during training time
	dropout_p1 = 0.25
	dropout_p2 = 0.5

	model = Sequential()
	#2d convolution(20 X 20 kernel)
	model.add(Convolution2D(20, 40, 40, border_mode='valid', input_shape=(channel, img_rows, img_cols)))
	#model.add(Convolution2D(number of filters, window shape, window shape...)
	#border_mode='valid': compute convolution where input and the filter fully overlap. output is smaller than input(the more "typical" cnn)
	#border_mode='same': output and input is same size
	model.add(Activation('relu'))
	
	model.add(Convolution2D(20, 40, 40))#only have to specify input dimensions the first time
	model.add(Activation('relu'))#activation_2

	#model.add(Convolution2D(20, 40, 40))#only have to specify input dimensions the first time
	#model.add(Activation('relu'))#activation_2

	model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
	model.add(Dropout(dropout_p1))
	"""
	Watch out for the the dimensions used in CNN
	Depending on th input dimensions,downsampling, convolution, pooling, etc, 
	feature maps may have height zero and give an error
	"""
	#Last part of a cnn is like a REGULAR neural network
	model.add(Flatten())#flatten cnn to nn
	model.add(Dense(128))#REGULAR (not cnn) fully connected nn layer
	#128: output of array shape(can be any number)
	model.add(Activation('sigmoid'))#activation_3
	model.add(Dropout(dropout_p1))
	model.add(Dense(nb_classes))#2

	model.add(Activation('softmax'))#activation_4
	return model, batch_size, nb_epoch

#predict(use the model I trained)
#ex)
#print('prediction of [1, 1]: ', model.predict_classes(np.array([[1, 1]]), verbose=verbose))
#print("prediction: ", model )
def get_predictions(filepath,imagesToTest):
	print("loading model...")
	grey_channel=1
	model = create_model(grey_channel)[0]
	model.load_weights(filepath)

	#predict_label1 = model.predict(imagesToTest, verbose=0)
	predict_label2 = model.predict_classes(imagesToTest, batch_size=10, verbose=0)
	#predict_label3 = model.predict_classes(imagesToTest, batch_size=10, verbose=1)
	predict_label4 = model.predict_on_batch(imagesToTest)
	print("prediction: ")
	#print(predict_label1)
	print(predict_label2)
	#print(predict_label3)
	print(predict_label4)
	#Yes(1): it is a neck cell
	#No(0): Not a cell

def predictUnseenImages(imageSource):
	unseenImage=[]
	return unseenImage


"""Run Program"""
print("Loading data...")
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
print (X_train.shape)#(1638, 1, 120, 120)
print (y_train.shape)#(1638,)
print (X_val.shape)#(545, 1, 120, 120)
print (y_val.shape)#(545,)
print (X_test.shape)#(547, 1, 120, 120)
print (y_test.shape)#(547,)

filepath="model8.h5"
nb_classes=2
buildCNN(X_train, y_train, X_val, y_val, nb_classes,filepath)
#Did not use test set yet

imageFile="trainYes/"+str(1)+".tif"
UnseenX=predictUnseenImages(imageFile)

#print(y_val)
#X_val= X_val[1:50]
#print (X_val)
get_predictions(filepath,X_val)

