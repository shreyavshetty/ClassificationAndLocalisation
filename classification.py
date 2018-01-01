import os
import sys
import numpy as np
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical,np_utils
from keras.layers import Dense, Input, GlobalMaxPooling1D,Dropout, Activation, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Convolution2D, MaxPooling2D,Conv2D
from keras.models import Model
import random
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM , Bidirectional,Dropout
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU
#from .utils.generic_utils import get_from_module
import json
import scipy
import os
import numpy as np
import argparse
import time
import cv2
from skimage import io,data 
import tensorflow as tf
from sklearn.metrics import confusion_matrix
with tf.device('/device:GPU:0'):
	sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
	d = {}
	c = 0
	classes =['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
	#train
	for i in classes:
		filename ='/home/shibani/anaconda2/envs/ai2/AILAB2_v0/HACK_AI/VOCdevkit/VOC2010/ImageSets/Main/'+i+'_train.txt'
		with open (filename, 'rt') as in_file:  # Open file lorem.txt for reading of text data.
			for line in in_file: # Store each line in a string variable "line"
				if c%2 ==0:
					lsplit = line.split()
					a1,a2 = [h for h in lsplit]
					a2 = int(a2)
					if a2 == -1:
                                		a2 = 0
					if a1 in d:
						# append the new number to the existing array at this slot
                                	        d[a1].append(a2)
		    			else:
						# create a new array in this slot
						d[a1] = [a2]
				c=c+1
		c=0

	pre_x_train = list(d.keys())
	y_train = np.array(list(d.values()))
	x = []
	for i in pre_x_train:
		filename = '/home/shibani/anaconda2/envs/ai2/AILAB2_v0/HACK_AI/VOCdevkit/VOC2010/JPEGImages/'+i+'.jpg'
		#print "reading image:"+ i + ".jpg"
	    	img = filename
	    	img = cv2.imread(img)
	    	img = cv2.resize(img,(224,224))
	    	#img = img.transpose((2,0,1))
	    	x.append(img)
	x_train = np.array(x)
	x_train = x_train.astype('float32')
	x_train /= 255	
	#endtrain
	#val
	d1 = {}
	c1 = 0
	for i in classes:
		filename ='/home/shibani/anaconda2/envs/ai2/AILAB2_v0/HACK_AI/VOCdevkit/VOC2010/ImageSets/Main/'+i+'_val.txt'
		with open (filename, 'rt') as in_file:  # Open file lorem.txt for reading of text data.
			for line in in_file: # Store each line in a string variable "line"
				if c1%2 ==0:
					l = line.split()
					a1,a2 = [h for h in l]
					a2 = int(a2)
                                	if a2 == -1:
                                        	a2 = 0
					if a1 in d1:
                                        	d1[a1].append(a2)
		    			else:
						# create a new array in this slot
						d1[a1] = [a2]
				c1=c1+1
		c1=0
	pre_x_val = list(d1.keys())
	y_val = np.array(list(d1.values()))
	print(len(y_val))
	x = []
	for i in pre_x_val:
		filename = '/home/shibani/anaconda2/envs/ai2/AILAB2_v0/HACK_AI/VOCdevkit/VOC2010/JPEGImages/'+i+'.jpg'
		#print "reading image:"+ i + ".jpg"
	    	img = filename
	    	img = cv2.imread(img)
	    	img = cv2.resize(img,(224,224))
	    	#img = img.transpose((2,0,1))
	    	x.append(img)
	x_val = np.array(x)
	x_val = x_val.astype('float32')
	x_val /= 255
        batch_size = 64
	epochs = 5
	num_classes = 20
       	#y_val = keras.utils.to_categorical(y_val, num_classes=20)
	#y_train = keras.utils.to_categorical(y_train, num_classes=20)
	print(len(y_val))
	print(len(y_train))
        print(x_train.shape, y_train.shape)
	print(x_val.shape, y_val.shape)	
	model = Sequential() 
	model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(224,224,3)))
	model.add(Convolution2D(32, 3, 3, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.25))
 	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(20, activation='softmax'))
	#fashion_model.add(LeakyReLU(alpha=0.1))                  
	#fashion_model.add(Dense(num_classes, activation='softmax'))
	# let's train the model using SGD + momentum (how original).
	#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	#model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
	#model.fit(x_train, y_train, batch_size=32, nb_epoch=20,callbacks=[check],validation_data=(x_val,y_val))
	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
	model.fit(x_train,y_train,batch_size,epochs, verbose=1,validation_data=(x_val, y_val))
	#fashion_train = fashion_model.fit(x_train,y_train, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_val, y_val))
	#score , acc = model.evaluate(x_val, y_val, batch_size=batch_size)
	y_pred = model.predict(x_val)
	print(y_pred)
	tn = [],fn=[],tp=[],fp=[]
	for a in range(len(classes)):
	    	tp1 =0
	   	fp1 =0
	   	tn1 =0
	   	fn1 =0
	    	for b in range(len(y_pred)):
	    		if y_pred[a][b] == y_val[a][b]:
	    			if y_pred[a][b] == 1:
	    				tp1 = tp1+1
	    			else:
	    				tn1 = tn1+1
	    		else:
	    			if y_pred[a][b] == 1 and y_val[a][b] == 0:
	    				fp1 = fp1+1
	    			else:
	    				fn1 = fn1+1
	    	tp.append(tp1)
	    	fp.append(fp1)
	    	tn.append(tn1)
	    	fn.append(fn1)
	    	accuracy = []
	co_mat = confusion_matrix(y_val, y_pred,labels =['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor'])
	for a in range(len(classes)):
	    	acc[a] = (tp[a]+tn[a])/(tp[a] + tn[a] + fp[a] + fn[a])
	    	pre[a] = (tp[a]) / (tp[a] + fp[a])
	    	rec[a] = (tp[a] / (tp[a] + fn[a]))
	print(acc)
	print(pre)
	print(rec)
