import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, LSTM, Reshape, Masking, Dropout, Layer
import tensorflow.keras.backend as K

def RNN_layer(n_timesteps:int, n_features:int,  rnn_units:int,
n_layer_rnn = 2, type_layer = None, activation_fn_rnn = None, mask_value = 0, 
r_drop = 0.3):
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

    tf.random.set_seed(
    1234)

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
                rnn_units, kernel_initializer='glorot_normal', 
                return_sequences=True, dropout = r_drop,   #return hidden state,
            )
            lst_layer.append(l_rnn)

    l_last = FUNC(rnn_units, return_sequences= True)

    lst_layer.append(l_last)

    return lst_layer

def FNN_layer(dense_units:int, n_layer_dense = 3, activation_fn_dense = None, 
drop = True, r_drop = 0.3):
    '''
    input data: 
        n_layer_dense  : number of dense layers, by default 3
        dense_units    : number of units in each hidden dense layer
        activation_fn_dense : activation function for rnn layer, by default 'relu'
        type_layer   : which rnn layer to construct network, by default LTSM       
        mask_value   : mask_value as signal of missing value, by default 0
        drop_out     : drop out layer by default True
        r_drop       : drop_out rate by default 0.3

    output:
        list of layers
    '''

    tf.random.set_seed(
    1234)

    lst_layer = []

    if n_layer_dense == 1:
        pass 
    else: 
        #construct rnn layers
        for i in range(n_layer_dense-1):
            l_fnn = Dense(
                dense_units, kernel_initializer='glorot_normal', 
                )
            lst_layer.append(l_fnn)
            if drop == True:
                l_drop = Dropout(r_drop)
                lst_layer.append(l_drop)

    return lst_layer


class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")        
        super(attention, self).build(input_shape)

    def call(self,x):
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)
        output=x*at
        return K.sum(output,axis=1)

    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        return super(attention,self).get_config()