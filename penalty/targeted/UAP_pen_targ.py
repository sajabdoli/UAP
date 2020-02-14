# Main function for UAP generation
# Penalty-based algorithm is used for targeted UAP generation.
# inputs to the "un_pert_gen" function are:
     # Target_class: The index number of the target class
     # nb_classes: Total number of classes
     # len_train_set_att: Number of samples in training set for UAP generation
     # len_batch: Batch size
     # X_train_arced: Training set transformed to tanh space
     # Y_train_att: Labels of the samples in Training set
     # model_end: End-to-end audio classifier used as target model
     # CONFIDENCE (kappa): confidence level of sample misclassification (default= 10)
     # C: penalty coefficient (default= 0.15)
     # delta_fooling_rate: desired fooling rate on perturbed training samples (default = 0.1)
# outputs of "un_pert_gen" function are pertubation vector=v and Attack Success Rate (ASR) on train set.
#target_names=['air_conditioner'=0,'car_horn'=1,'children_playing'=2,'dog_bark'=3,
#              'drilling'=4,'engine_idling'=5,'gun_shot'=6,'jackhammer'=7,'siren'=8,'street_music'=9]

from misc import fool_rate_comp_targeted, log10, create_wave_batch_t, energy_tensor, energy_np

import numpy as np
from keras.layers.core import Lambda

import tensorflow as tf
config = tf.ConfigProto( )
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
import keras.backend.tensorflow_backend as tf_bkend
tf_bkend.set_session(sess)
from keras import backend as K

# Define loss function based on penalty method 
    # Loss1 is the hinge function,
    # Loss2 is the function which computes the Sound Pressure Level (SPL) of the perturbation vector.
def target_category_loss_targ(logits,new_wav,v, category_index, nb_classes, C, CONFIDENCE):
 
    # Loss1
    tlab = tf.one_hot([category_index], nb_classes)
    real = tf.reduce_sum((tlab)*logits,1)
    other = tf.reduce_max((1-tlab)*logits - (tlab*10000),1)
    loss1 =  tf.maximum(0.0, other-real+CONFIDENCE)
    
    #Loss2
    v=tf.dtypes.cast(v, dtype=tf.float32)
    db_dist=20.0*log10(energy_tensor(v))
    
    loss2 =tf.reduce_sum(db_dist)
    loss1 = tf.reduce_sum(C*loss1)

    return loss1+loss2

def un_pert_gen(Target_class,nb_classes, len_train_set_att, len_batch, X_train_arced, Y_train_att, model_end,
                CONFIDENCE=10, C=0.15, delta_fooling_rate=0.1):
    
    signal_shape = 50999
    boxmin=0 #Minimum signal value (default 0).
    boxmax=1 #Maximum signal value (default 1).
    boxmul = (boxmax - boxmin) / 2.
    boxplus = (boxmin + boxmax) / 2.
    
    ovp=int(len_train_set_att/len_batch)
    

    #Adam optimization algorithm parameters
    lr_G = 0.0035
    roh_1 = 0.9
    roh_2 =0.999
    eps = 1e-8
    r=0
    s=0

    max_epoch=100  # termination criterion
    epoch=0        # epoch Counter
    itr= 0        # Iteration counter

    # true perturbation: v_trainset in [0, 1]
    v_trainset= np.zeros((signal_shape,1), dtype=float)

    fooling_rate = 0.0

    num_wav_trainset =  np.shape(X_train_arced)[0] 

    while fooling_rate < 1-delta_fooling_rate and epoch < max_epoch:
        print ("#epoch:", epoch)

        for i in range(ovp):

            # batch_wav in [-inf, inf]
            batch_wav= create_wave_batch_t(X_train_arced, Y_train_att, len_batch, i)

            # Perturbation of signal in [-inf, inf]
            added_noise = np.add (batch_wav , v_trainset)
            # Transform back to [0, 1]
            new_wav_batch = np.tanh(added_noise) * boxmul + boxplus

            target_layer = lambda x: target_category_loss_targ(x, new_wav_batch,v_trainset, 
                                                  Target_class, nb_classes, C, CONFIDENCE)

            #Compute Gradinets
            input1 = model_end.layers[0].get_output_at(0)
            logits = model_end.layers[-1].get_output_at(0)
            loss=Lambda(target_layer)(logits)
            grads =  tf.gradients(loss, input1)
            gradient_function = K.function([input1], grads)
            grads_val = gradient_function([new_wav_batch])[0]
            grads_val = np.sum(grads_val, axis=0)

            itr= itr+1

            #Update biased ﬁrst moment estimate:
            s = roh_1*s + (1-roh_1)*grads_val

            #Update biased second moment estimate:
            r = roh_2*r + (1-roh_2) * np.multiply(grads_val,grads_val)

            # Correct bias in ﬁrst moment:
            s_hat =  s/(1-np.power(roh_1,itr))

            #Correct bias in second moment:
            r_hat = r/(1-np.power(roh_2,itr))

            # update perturbation
            deltav = np.multiply(-lr_G , (np.divide(s_hat,(np.sqrt(r_hat)+eps))))

            v_trainset = np.add(v_trainset,deltav)

        #compute fooling rate on all of the samples in train set
        #Transformation on signal  

        added_noise_train=X_train_arced+v_trainset
        new_wav_trainset=np.tanh(added_noise_train)* boxmul + boxplus

        # compute fooling rate    
        fooling_rate = fool_rate_comp_targeted(new_wav_trainset, num_wav_trainset, len_train_set_att, Target_class, model_end)
        epoch=epoch+1
   
    return v_trainset,fooling_rate



