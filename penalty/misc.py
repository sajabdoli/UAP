# auxiliary functions for UAP generation

import numpy as np
import random
import tensorflow as tf

def energy_tensor(x):
    signal_shape=tf.constant(50999,dtype=tf.float32)
    e = tf.sqrt(tf.math.divide(tf.reduce_sum(tf.math.square(x)),signal_shape))
    return e

def energy_np(x):
    signal_shape=50999.0
    e = np.sqrt(np.divide(np.sum(np.square(x)),signal_shape))
    return e

def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


# Create batch of of audio samples for UAP crafting
def create_wave_batch_t(X_train_att, Y_train_att, len_batch, batch_ind):
       
    ind=range((len_batch*batch_ind),(len_batch*batch_ind)+len_batch)

    sz_wav = [50999]
    num_channels=1
    wav_array = np.zeros([len_batch] + sz_wav + [num_channels], dtype=np.float32)
      
    wav_array[:,:,:] = X_train_att[ind]
        
    return wav_array


def create_wave_batch_u(X_train_att, Y_train_att, len_batch, batch_ind):
       
    ind=range((len_batch*batch_ind),(len_batch*batch_ind)+len_batch)

    sz_wav = [50999]
    num_channels=1
    wav_array = np.zeros([len_batch] + sz_wav + [num_channels], dtype=np.float32)
      
    wav_array[:,:,:] = X_train_att[ind]
        
    return wav_array, Y_train_att[ind]



# Perturb the dataset with computed perturbation and compute ASR on training set for Untargeted Attack
def fool_rate_comp_untargeted(wav, X, num_wav, batch_size, labels_true, model_end):
    
    labels_true = np.zeros((num_wav))
    est_labels_pert = np.zeros((num_wav))

    num_batches = np.int(np.ceil(np.float(num_wav) / np.float(batch_size)))

    # Compute the estimated labels in batches    
    for ii in range(0, num_batches):

        m = (ii * batch_size)
        M = min((ii+1)*batch_size, num_wav)
        
        #print (m)
        #print (M)

        labels_true[m:M] = np.argmax(model_end.predict(wav[m:M, :, :]), axis=1).flatten()
        est_labels_pert[m:M] = np.argmax(model_end.predict(X[m:M, :, :]), axis=1).flatten()

        #print (labels_target)
        #print (est_labels_pert)

        # Compute the fooling rate

        fooling_rate = float(np.sum(est_labels_pert != labels_true) / float(num_wav))
        print('FOOLING RATE = ')
        print ("%.5f" % fooling_rate)

        
        return fooling_rate
    
# Perturb the dataset with computed perturbation and compute ASR on training set for Targeted Attack
def fool_rate_comp_targeted(X, num_wav, batch_size, labels_true, model_end):
    
    labels_target = np.zeros((num_wav))
    est_labels_pert = np.zeros((num_wav))

    num_batches = np.int(np.ceil(np.float(num_wav) / np.float(batch_size)))

    # Compute the estimated labels in batches    
    for ii in range(0, num_batches):

        m = (ii * batch_size)
        M = min((ii+1)*batch_size, num_wav)
        
        #print (m)
        #print (M)

        labels_target[m:M] = Target_class
        est_labels_pert[m:M] = np.argmax(model_end.predict(X[m:M, :, :]), axis=1).flatten()
     
        #print (labels_target)
        #print (est_labels_pert)

        # Compute the fooling rate

        fooling_rate = float(np.sum(est_labels_pert == labels_target) / float(num_wav))
        print('FOOLING RATE = ')
        print ("%.5f" % fooling_rate)
        
        return fooling_rate  

