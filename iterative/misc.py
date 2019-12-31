#auxiliary functions
 
import numpy as np
import random

def energy_np(x):

    # X: signal
    signal_shape=50999.0
    e = np.sqrt(np.divide(np.sum(np.square(x)),signal_shape))
    return e


def create_wave_batch(len_batch,X_train,Y_train, L):

    # len_batch = Number of available samples for UAP generation, 
    # X_train,Y_train = Samples and labels of the training set
    # L = length of the signal
    
    ovp=int(X_train.shape[0]/len_batch)

    batch_ind=random.randint(1,ovp)

    ind=range((len_batch*batch_ind)-len_batch,len_batch*batch_ind)

    sz_wav = [L]
    num_channels=1
    
    wav_array = np.zeros([len_batch] + sz_wav + [num_channels], dtype=np.float32)
      
    wav_array[:,:,:] = X_train[ind]
        
    return wav_array, np.argmax(Y_train[ind], axis=1)



# function for projecting projected on the lp ball of radius xi 
def proj_lp(v, xi, p):

    # Project on the lp ball centered at 0 and of radius xi
    # SUPPORTS only p = 2 and p = Inf
    
    if p == 2:
        v = v * min(1, xi/np.linalg.norm(v))
        
    elif p == np.inf:
        v = np.sign(v) * np.minimum(abs(v), xi)

    return v
