# This model uses 227 Sinc filters to extract features from the raw audio signal as it is introduced in SincNet (https://arxiv.org/abs/1808.00158) the output of sincnet is stacked along time axis to form a 2D representation. This time-frequency representation is used as input to a VGG19 network followed by a FC layer and softmax layer for classification.

import tensorflow as tf
import sincnet
from keras.layers import Dense, Dropout, Activation
from keras.layers import MaxPooling1D, Conv1D, LeakyReLU, BatchNormalization, Dense, Flatten
from keras.layers import InputLayer, Input, Dense, MaxPool1D, Flatten, Activation, Conv2D, MaxPooling2D, Input, Activation, Convolution1D
from keras.models import Model
from keras import regularizers, optimizers
import keras.initializers as init
from keras import backend as K

from keras.applications.vgg19 import VGG19
from keras.callbacks import ModelCheckpoint
from keras.layers import Reshape

from functools import partial, update_wrapper

def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func

def model_generator():
    
    inp =  Input(shape=(50999,1))
    
    x = sincnet.SincConv1D(227, 251, 16000)(inp)
    
    x = MaxPooling1D(pool_size=218)(x)
    
    x = sincnet.LayerNorm()(x)
    
    reshape_layer=Reshape((232, 227,-1))(x)
    
    x=VGG19(include_top=False, weights=None, 
            input_tensor=reshape_layer, input_shape=(232, 227,-1), pooling=None)
    
    x = Flatten()(x.output)
    
    x = Dense(4096,activation="relu")(x)
    
    dense4 = Dense(10,activation=None)(x)
    
    model = Model(inputs=inp, outputs=dense4)
    
    model.compile(loss=wrapped_partial(K.categorical_crossentropy, from_logits=True),
          optimizer=optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
          ,metrics=['accuracy'])
    
    return model



