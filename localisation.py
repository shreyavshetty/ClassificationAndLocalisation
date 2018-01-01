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
import xmltodict
pre_xtrain = []
y_train = []
with tf.device('/device:GPU:0'):
	sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))	
	xml_filepath="/home/shibani/anaconda2/envs/ai2/AILAB2_v0/HACK_AI/VOCdevkit/VOC2010/Annotations/"
	mylist = os.listdir(xml_filepath)
	bnd='bndbox'
	count=0
	coords={}
	for i in mylist:
		xml_file = xml_filepath +i
		arg1=i.split('.')[0]
		discard=0
		with open(xml_file, "rb") as f:    # notice the "rb" mode
			d = xmltodict.parse(f, xml_attribs=True)
			l=[]
			#print  type(d["annotation"]["object"])
			if type(d["annotation"]["object"]) == type(l):
				#print "loss"
				discard=1
			else:
				coords={arg1:[]}
				pre_xtrain.append(arg1)
			if(discard==0):
				x1=((int(d['annotation']['object']['bndbox']['xmin']))*224)/(int(d['annotation']['size']['width']))
				x2=((int(d['annotation']['object']['bndbox']['xmax']))*224)/(int(d['annotation']['size']['width']))
				x3=((int(d['annotation']['object']['bndbox']['ymax']))*224)/(int(d['annotation']['size']['height']))
				x4=((int(d['annotation']['object']['bndbox']['ymin']))*224)/(int(d['annotation']['size']['height']))
				coords[arg1].append(x1)
				coords[arg1].append(x2)
				coords[arg1].append(x3)
				coords[arg1].append(x4)
				y_train.append(coords[arg1])					
	y_train = np.array(list(y_train))
	#print (len(pre_xtrain))
	x = []
	for i in pre_xtrain:
		filename = '/home/shibani/anaconda2/envs/ai2/AILAB2_v0/HACK_AI/VOCdevkit/VOC2010/JPEGImages/'+i+'.jpg'
		#print "reading image:"+ i + ".jpg"
	    	img = filename
	    	img = cv2.imread(img)
	    	img = cv2.resize(img,(224,224))
	    	#img = img.transpose((2,0,1))
	    	x.append(img)
	
	x_train = np.array(x)
	x_train = x_train.astype('float32')
	#x_train /= 255	
	#endtrain
	#val
	d1 = {}
	c1 = 0
        '''
	for i in pre_xval:
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
	'''
        batch_size = 64
	epochs = 20
       	#y_val = keras.utils.to_categorical(y_val, num_classes=20)
        print(x_train.shape, y_train.shape)
	#print()
	'''
	print (type(x_train))
	print (type(y_train))
	print (type(y_val))
	print (type(x_val))
	'''
	model = Sequential() 
	model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(224,224,3)))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Convolution2D(64, (5, 5), activation='relu', input_shape=(224,224,3)))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.25))
 	model.add(Flatten())
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4,kernel_initializer='normal', activation='relu'))
	model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])
	model.summary()
	#print(x_val.shape, y_val.shape)
	model.fit(x_train,y_train,epochs=10,validation_split=0.1,verbose=1,batch_size=32)
	print("fit done")
	a= model.evaluate(x_train,y_train,verbose=0)
	print a
	#fashion_model.add(LeakyReLU(alpha=0.1))                  
	#fashion_model.add(Dense(num_classes, activation='softmax'))
	# let's train the model using SGD + momentum (how original).
	#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	#model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
	#model.fit(x_train, y_train, batch_size=32, nb_epoch=20,callbacks=[check],validation_data=(x_val,y_val))
	#model.compile(loss='_crossentropy',optimizer='adam',metrics=['accuracy'])
	#model.fit(x_train,y_train,batch_size,epochs, verbose=1,validation_data=(x_val, y_val))
'''
	y_pred = model.predict(x_train)
	for a in range(len(classes)):
	    	tp1 =0
	   	fp1 =0
	   	tn1 =0
	   	fn1 =0
	    	for b in range(len(y_val)):
	    		if y_pred[a][b] == y_train[a][b]:
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
	for a in range(len(classes)):
	    	acc[a] = (tp[a]+tn[a])/(tp[a] + tn[a] + fp[a] + fn[a])
	    	pre[a] = (tp[a]) / (tp[a] + fp[a])
	    	rec[a] = (tp[a] / (tp[a] + fn[a]))
	print(acc)
	print(pre)
	print(rec)
'''
