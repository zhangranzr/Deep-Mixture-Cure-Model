import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, LSTM, Reshape, Masking, Dropout

def RNN_layer(n_timesteps:int, n_features:int,  rnn_units:int,
n_layer_rnn = 2, type_layer = None, activation_fn_rnn = None, mask_value = 0, 
drop_out = True, r_drop = 0.3):
    '''
    input data: 
        input_shape  : shape of RNN ('batch_size', n_timestep, n_features) 
        n_layer_rnn  : number of rnn layers, by default 2
        rnn_units    : number of units in each hidden rnn layer
        activation_fn_rnn : activation function for rnn layer, by default 'relu'
        type_layer   : which rnn layer to construct network, by default LTSM       
        mask_value   : mask_value as signal of missing value, by default 0
        drop_out     : drop out layer by default True
        r_drop       : drop_out rate by default 0.3

    output:
        list of layers
    '''

    lst_layer = []
    #first layer masking for RNN 
    l_mask = Masking(mask_value = mask_value, input_shape=(n_timesteps, n_features))
    lst_layer.append(l_mask)
    
    if type_layer == None:
        FUNC = LSTM
    else:
        FUNC = type_layer

    if n_layer_rnn == 1:
        pass 
    else: 
        #construct rnn layers
        for i in range(n_layer_rnn-1):
            l_rnn = FUNC(
                rnn_units, 
                return_sequences=True   #return hidden state
            )
            lst_layer.append(l_rnn)

    #last layer of RNN
    if drop_out == True:
        l_dropout = Dropout(
            r_drop
            )
        lst_layer.append(l_dropout)
    l_last = FUNC(1, 
    return_sequences=True
    )

    lst_layer.append(l_last)

    return lst_layer

def head_RNN(n_timesteps):
    reshape  = Reshape(
    (n_timesteps, ), 
    )
    pi       = Dense(
    1, 
    activation="sigmoid", 
    input_shape=(n_timesteps,)
    )

    return [reshape, pi]