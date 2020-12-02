#compute loss
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, LSTM, Reshape, Masking, Dropout, Layer

class get_loss():
    def __init__(self, batch_index, mask, OUT_pi, OUT_hi, censoring_status,
    payoff_status, RNN_net, alpha = 0.5, _EPSILON = 1e-08):
        self.batch_index = batch_index
        self.mask = mask
        self.OUT_h_i = OUT_hi
        self.OUT_p_i  = OUT_pi
        self.RNN_net = RNN_net 
        self.censoring_status = censoring_status
        self.payoff_status = payoff_status
        self.alpha = alpha
        self.p_il = None
        self.S_t  = None
        self.loss_likelihood_mix = None
        self.loss_incidence = None
        self._EPSILON = _EPSILON 

    #log_likelihood preparation
    def Prob_Surv(self):

        _EPSILON = self._EPSILON
        (mask_1, mask_2, mask_3) = self.mask 
        
        mask_1 = tf.constant(mask_1[self.batch_index], shape = self.OUT_h_i.shape, dtype = 'float32')
        mask_2 = tf.constant(mask_2[self.batch_index], shape = self.OUT_h_i.shape, dtype = 'float32')
        mask_3 = tf.constant(mask_3[self.batch_index], shape = self.OUT_h_i.shape, dtype = 'float32')

        if self.RNN_net == False:
            h_i = tf.nn.softmax(self.OUT_h_i, axis = 1)
            h_i = tf.reshape(tf.math.multiply(h_i, mask_1), mask_1.shape, name='h_i') 
            W_t = tf.clip_by_value(tf.reshape(tf.math.reduce_sum(h_i, axis=1, name='W_t'), 
            (len(self.batch_index), 1)), _EPSILON, 1-_EPSILON)
            S_t = 1 - W_t
            p_il = tf.clip_by_value(tf.reshape(tf.reduce_sum(tf.reshape(tf.math.multiply(h_i, mask_2), mask_2.shape), axis = 1), 
            (len(self.batch_index), 1)), _EPSILON, 1-_EPSILON)

            self.p_il = p_il
            self.S_t  = S_t

            return self.p_il, self.S_t

        else:
            h_i = tf.reshape(tf.math.multiply(self.OUT_h_i, mask_1), mask_1.shape, name='h_i')
            S_t = tf.clip_by_value(tf.reshape(tf.math.reduce_prod(1- h_i, axis=1, keepdims=False, name='S_i'),
            (len(self.batch_index), 1)), _EPSILON, 1-_EPSILON)

            h_last = tf.reduce_sum(tf.reshape(tf.math.multiply(self.OUT_h_i, mask_2), mask_2.shape), axis = 1) 
            # shape (batch_size, 1) unwanted points == 0
            mul_h = tf.math.reduce_prod(1- tf.math.multiply(self.OUT_h_i, mask_3, mask_3.shape,), axis=1)  # (50, 1)
            p_il = tf.clip_by_value(tf.reshape(tf.math.multiply(h_last, mul_h), 
            (len(self.batch_index), 1)), _EPSILON, 1-_EPSILON)

            self.p_il = p_il
            self.S_t  = S_t

            return self.p_il, self.S_t

    def loss_likelihood_mixture(self):   # censoring case == 0
        #weigths_censored   =  
        #weigths_uncensored = 
        
        Prob, S_t = self.Prob_Surv()
        
        p_i  = self.OUT_p_i
        #Prob = self.p_il
        #S_t  = self.S_t

        # censored data
        l_censored = 1 - p_i + tf.math.multiply(p_i, S_t)
        l_censored = l_censored[self.censoring_status == 0]
        l_censored = tf.reduce_sum(
            tf.math.log(l_censored)
            ) 
            
        # uncensored data, true event time z
        l_uncensored = tf.math.multiply(p_i, Prob)
        l_uncensored = l_uncensored[self.censoring_status == 1]
        l_uncensored = tf.reduce_sum(
            tf.math.log(l_uncensored)
            )
        # punishment of t>z
        l_uncensored_2 = tf.reduce_sum(
            tf.math.log((1-S_t)[self.censoring_status == 1])
            )

        self.loss_likelihood_mix = -(l_censored + l_uncensored + l_uncensored_2)

        return self.loss_likelihood_mix 

    def loss_icds(self):
        p_i  = self.OUT_p_i
        l_incidence  = tf.keras.losses.binary_crossentropy(
                self.censoring_status[self.censoring_status + self.payoff_status == 1], 
                p_i[self.censoring_status + self.payoff_status == 1])

        self.loss_incidence = l_incidence
        return self.loss_incidence
    
    def AUC(self):
        acc   = tf.keras.metrics.AUC(int(len(self.censoring_status)/2))
        acc.update_state(self.censoring_status[self.censoring_status + self.payoff_status == 1], 
        self.OUT_p_i[self.censoring_status + self.payoff_status == 1]
        )

        return acc.result()
    #def ranking_loss(self):


    
    #def loss_total(self):

        


