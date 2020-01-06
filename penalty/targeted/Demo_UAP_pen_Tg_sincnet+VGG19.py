# Demo code for UAP generation based on Penalty-based algorithm. 
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
sys.path.insert(0, os.path.join(dir_of_file,"penalty"))

from misc import energy_np, energy_tensor, create_wave_batch_t
from UAP_pen_targ import un_pert_gen
from UAP_validation import validation


from model import model_generator
from keras.callbacks import ModelCheckpoint


if __name__ == '__main__':


    #Define default parameters:

    # Number of available samples for UAP generation
    len_batch= 1000

    # Test fold
    fold = 0

    # Traget class
    #target_names=['air_conditioner'=0,'car_horn'=1,'children_playing'=2,'dog_bark'=3,
    #              'drilling'=4,'engine_idling'=5,'gun_shot'=6,'jackhammer'=7,'siren'=8,'street_music'=9]
    Target_class = 6

    # Batch size
    len_batch= 70
    # No. of availablew training samples
    len_train_set_att=1000

    # Penalty method parameters
    CONFIDENCE=10
    C=0.15
    delta_fooling_rate=0.1
    nb_classes = 10


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
    str2=""+str(fold)+".best.hdf5" 
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

    #loading the best weights for this fold 
    model_end.load_weights(filepath_model_weights)

    #evaluating the model
    print("testing on test set of fold number:"+str(fold)) 
    print(model_end.evaluate([X_test], Y_test, verbose=1))
    print(model_end.metrics_names)


    print('>> Creating pre-processed data...')

    X_train_att=X_train[0:len_train_set_att]
    Y_train_att=Y_train[0:len_train_set_att]

    # convert audio signals to tanh-space
    X_train_arced = np.arctanh((np.multiply(2.0 , X_train_att) - 1.0) * (1.0 - 0.0000001))


    print ("attacking target:", Target_class)
    v, fooling_rate = un_pert_gen(Target_class,nb_classes, len_train_set_att, len_batch, X_train_arced, Y_train_att, model_end,
                    CONFIDENCE, C, delta_fooling_rate)

    print ("fooling_rate_train", fooling_rate)


    SNR_avg, fooling_rate_test=validation(v, Target_class, model_end, X_test)


    print ("SNR", SNR_avg)
    print ("fooling rate",fooling_rate_test)
    
    
    # save file of UAP on disk
    str1="universal-pen-TARGETED-sincnet+VGG19_T"
    str2=""+str(Target_class)+".best.npy" 
    np.save(str1+str2, v)





