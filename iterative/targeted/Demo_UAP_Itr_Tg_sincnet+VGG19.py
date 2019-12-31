# Demo code for UAP generation based on iterative greedy algorithm. 
# Attacking scenario is targeted attack. The target model is SincNet+VGG19
# This code fits the model and generates the UAP
# This code also tests the generated UAP on test set
# Model is trained on UrbanSound8k (https://urbansounddataset.weebly.com/urbansound8k.html) dataset.

import numpy as np
import sys
import os.path
from os.path import dirname, realpath
import pickle
from sklearn.model_selection import KFold # import KFold

import tensorflow as tf
config = tf.ConfigProto( )
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
import keras.backend.tensorflow_backend as tf_bkend
tf_bkend.set_session(sess)

filepath = realpath("..")
dir_of_file = dirname(filepath)

sys.path.insert(0, os.path.join(dir_of_file,"Sincnet_supp_files"))
sys.path.insert(0, os.path.join(dir_of_file,"iterative"))

from misc import energy_np, create_wave_batch, proj_lp
from UAP_it_targ import un_pert_gen
from UAP_validation import validation
from model import model_generator
from keras.callbacks import ModelCheckpoint

from ddn import *

if __name__ == '__main__':


	#Define default parameters:

	# Number of available samples for UAP generation
	len_batch= 1000

	# Test fold  
	fold = 0

	# batch_shape : tuple (B x L x C) signal with the duration of ~3 seconds
	# B:Batch , L: Length of the signal C: Channels
	L = 50999
	batch_shape = (1, L, 1)

	# Target Class
	#target_names=['air_conditioner'=0,'car_horn'=1,'children_playing'=2,'dog_bark'=3,
	#              'drilling'=4,'engine_idling'=5,'gun_shot'=6,'jackhammer'=7,'siren'=8,'street_music'=9]
	Target_class=6

	#controls the l_inf magnitude of the perturbation (default = 0.12)
	xi=0.12
    
    # desired fooling rate on perturbed training samples
    delta=0.1

	#LOAD SAVED UrbanSound8k DATASET dumped in .npy files (8732 audio samples)
	X=np.load(os.path.join(dir_of_file,"data","data_X.npy"))
	Y=np.load(os.path.join(dir_of_file,"data","data_Y.npy"))


	# In order to split the data set into 10 folds:

	kf = KFold(n_splits=10, shuffle=True) # Define the split - into 10 folds 
	kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
	print(kf)
	mylist = list(kf.split(X))

	# Save the index to the samples of the dataset for 10 folds in disk 
	file_pi = open(os.path.join(dir_of_file,"data",'FOLDS_ATTACK.obj'), 'wb') 
	pickle.dump(mylist, file_pi)

	# Load the index to the samples of the dataset for 10 folds 
	mylist = pickle.load(open(os.path.join(dir_of_file,"data","FOLDS_ATTACK.obj"), 'rb' ))


	#Train the model on folds 1 to 9 and test on fold 0 (default)
	train_index, test_index = mylist[fold]

	#checkpoints 
	str1="weightsurban_ATTACK_SINCNET+VGG19_"
	str2=""+str(0)+".best.hdf5" 
	filepath_model_weights=str1+str2 
	print(filepath_model_weights)

	model_end = model_generator()

	#train & test set 
	X_train, X_test = X[train_index], X[test_index] 
	Y_train, Y_test = Y[train_index], Y[test_index]

	#Fit the model

	checkpoint = ModelCheckpoint(filepath_model_weights, monitor='val_acc', verbose=1, save_best_only=True, mode='max') 
	callbacks_list = [checkpoint]

	model_end.fit( X_train, Y_train, batch_size=20,nb_epoch=100, verbose=1,callbacks=callbacks_list,validation_split=0.10)


	#Load weights of the model
	model_end.load_weights(filepath_model_weights)

	#evaluating the model
	print("testing on test set of fold number:"+str(fold)) 
	print(model_end.evaluate([X_test], Y_test, verbose=1))
	print(model_end.metrics_names)


	print('>> Creating pre-processed data...')
	X,Y = create_wave_batch(len_batch,X_train,Y_train, L)


	# define parameters of DDN attack
	attacker = DDN_tf(model_end,  batch_shape, steps=50, targeted=True, quantize=True, init_norm=0.2)



	print ("attacking target:", Target_class)

	# Call UAP generation function
	v, fooling_rate = un_pert_gen(Target_class, model_end, X, attacker, sess, xi, delta)
	print ("fooling_rate_train", fooling_rate)


	# validation on test set
	SNR_avg, fooling_rate_test=validation(v, Target_class, model_end, X_test)
	print ("SNR", SNR_avg)
	print ("fooling rate",fooling_rate_test)


	# save file of UAP on disk
	str1="universal-itr-TARGETED-sincnet+VGG19_T"
	str2=""+str(Target_class)+".best.npy" 
	np.save(str1+str2, v)

