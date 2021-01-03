#compute loss
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, LSTM, Reshape, Masking, Dropout, Layer
from lifelines.utils import concordance_index

class get_loss():
    def __init__(self, batch_index, pat_info, MASK, OUT_pi, OUT_hi, censoring_status,
    payoff_status, RNN_net, sigma1, alpha = 0.5, _EPSILON = 1e-08, punish_z = True,
    ):
        self.batch_index = batch_index
        self.pat_info    = pat_info
        self.mask = MASK
        self.OUT_h_i = OUT_hi  # conditional prob/ instantaneous probability (n_customer, n_maxtime)
        self.OUT_p_i  = OUT_pi # instance probability (n_customer,1)
        self.RNN_net = RNN_net 
        self.censoring_status = censoring_status
        self.payoff_status = payoff_status
        self.alpha = alpha
        self.p_il = None
        self.S_t  = None
        self.loss_likelihood_mix = None#
        self.punish_z = punish_z
        self._EPSILON = _EPSILON 
        self.sigma1 = sigma1

    #log_likelihood preparation
    def Prob_Surv(self):

        _EPSILON = self._EPSILON
        (mask_1, mask_2, mask_3) = self.mask[0] 
        
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

            self.p_il = p_il # probability of event time
            self.S_t  = S_t

            return self.p_il, self.S_t

    def loss_likelihood_mixture(self):   # censoring case == 0
        
        _EPSILON = self._EPSILON
        Prob, S_t = self.Prob_Surv()
        
        p_i  = self.OUT_p_i
        #Prob = self.p_il
        #S_t  = self.S_t

        # censored data
        l_censored = 1 - p_i + tf.clip_by_value(tf.math.multiply(p_i, S_t), _EPSILON, 1-_EPSILON)
        l_censored = l_censored[self.censoring_status == 0]
        l_censored = tf.reduce_sum(
            tf.math.log(l_censored)
            ) 
            
        # uncensored data, true event time z
        l_uncensored = tf.clip_by_value(tf.math.multiply(p_i, Prob), _EPSILON, 1-_EPSILON)
        l_uncensored = l_uncensored[self.censoring_status == 1]
        l_uncensored = tf.reduce_sum(
            tf.math.log(l_uncensored)
            )
        
        # censored data, max the prob of cumulative prob until event time
        W_t = tf.clip_by_value(tf.math.multiply(p_i, 1-S_t), _EPSILON, 1-_EPSILON)
        l_uncensored_2 = tf.reduce_sum(
            tf.math.log(W_t[self.censoring_status == 1]
            ))
        
        if (self.punish_z):
            self.loss_likelihood_mix = -(l_censored*0.25 + l_uncensored*0.5 + l_uncensored_2*0.25)
        else:
            self.loss_likelihood_mix = -(l_censored*0.5 + l_uncensored*0.5)
        
        return self.loss_likelihood_mix, (l_censored, l_uncensored, l_uncensored_2)

    def loss_likelihood_non_mixture(self):   # censoring case == 0
        
        _EPSILON = self._EPSILON
        Prob, S_t = self.Prob_Surv()

        # censored data
        l_censored = tf.clip_by_value(S_t, _EPSILON, 1-_EPSILON)
        l_censored = l_censored[self.censoring_status == 0]
        l_censored = tf.reduce_sum(
            tf.math.log(l_censored)
            ) 
            
        # uncensored data, true event time z
        l_uncensored = tf.clip_by_value(Prob, _EPSILON, 1-_EPSILON)
        l_uncensored = l_uncensored[self.censoring_status == 1]
        l_uncensored = tf.reduce_sum(
            tf.math.log(l_uncensored)
            )
        
        # censored data, max the prob of cumulative prob until event time
        W_t = tf.clip_by_value(1-S_t, _EPSILON, 1-_EPSILON)
        l_uncensored_2 = tf.reduce_sum(
            tf.math.log(W_t[self.censoring_status == 1]
            ))
        
        if (self.punish_z):
            self.loss_likelihood_mix = -(l_censored*0.25 + l_uncensored*0.5 + l_uncensored_2*0.25)
        else:
            self.loss_likelihood_mix = -(l_censored*0.5 + l_uncensored*0.5)
        
        return self.loss_likelihood_mix, (l_censored, l_uncensored, l_uncensored_2)
    
    def AUC(self):
        acc   = tf.keras.metrics.AUC(int(len(self.censoring_status)/2))
        acc.update_state(self.censoring_status, 
        self.OUT_p_i
        )

        return acc.result()

    ### LOSS-FUNCTION 2 -- Ranking loss
    def loss_Ranking(self):
        '''
        compute ranking loss. 
        return: tf.tensor
        input(no competing risk):
            pi_l        : probability matrix of customers dim(customers, max_time)
            mask_ranking: 
        '''
        mask_ranking = self.mask[1][self.batch_index]
        sigma1 = self.sigma1
        label = self.censoring_status

        OUT_hi = self.OUT_h_i 

        if self.RNN_net:
            pil_matrix = np.zeros(OUT_hi.shape)
            pil_matrix[:, 0] = OUT_hi[:, 0]
            for i in range(OUT_hi.shape[-1]):
                if i == 0:
                    pass
                else:
                    temp_h = OUT_hi[:, i]
                    temp_h_1 = tf.math.subtract(1, OUT_hi)[:, :i-1]
                    h = tf.reduce_prod(temp_h_1, axis = 1)
                    pil_matrix[:, i] = tf.math.multiply(temp_h, h).numpy()
            OUT_hi = tf.nn.softmax(tf.constant(pil_matrix, dtype=tf.float32), axis = 1)
        else: 
            pass

        sigma1 = tf.constant(sigma1, dtype=tf.float32)
        label  = tf.constant(label,  dtype=tf.dtypes.float32)
        mask_ranking = tf.constant(mask_ranking,  dtype=tf.dtypes.float32)
        
        one_vector = tf.ones_like(label, dtype=tf.float32)

        tmp_e = tf.reshape(OUT_hi, shape = mask_ranking.shape)
        R = tf.matmul(tmp_e, tf.transpose(mask_ranking)) 
        # cummulative incidence prob at actual event time
        diag_R = tf.reshape(tf.linalg.diag_part(R), [-1, 1]) 
        R = tf.matmul(one_vector, tf.transpose(diag_R)) - R 
        R = tf.transpose(R)  
        T = tf.nn.relu(tf.sign(tf.matmul(one_vector, tf.transpose(label)) - tf.matmul(label, tf.transpose(one_vector))))
        eta = tf.reduce_sum(tf.reduce_mean(T * tf.exp(-R/sigma1), axis=1, keepdims=True))
        return eta
    
    def ANLP(self):

        Prob, S_t = self.Prob_Surv()
        _EPSILON = self._EPSILON
        Prob = tf.clip_by_value(Prob, _EPSILON, 1-_EPSILON)

        P = tf.reduce_sum(
            tf.math.log(Prob[self.censoring_status == 1]
            ))
        P = P/sum((self.censoring_status == 1))*(-1)

        return P

    def c_index(self):

        OUT_hi = self.OUT_h_i
        if self.RNN_net:
            pil_matrix = np.zeros(OUT_hi.shape)
            pil_matrix[:, 0] = OUT_hi[:, 0]
            for i in range(OUT_hi.shape[-1]):
                if i == 0:
                    pass
                else:
                    temp_h = OUT_hi[:, i]
                    temp_h_1 = tf.math.subtract(1, OUT_hi)[:, :i-1]
                    h = tf.reduce_prod(temp_h_1, axis = 1)
                    pil_matrix[:, i] = tf.math.multiply(temp_h, h).numpy()
            pil_matrix = tf.nn.softmax(pil_matrix, axis = 1)
        else:
            pil_matrix = OUT_hi

        events = self.pat_info[:, 3]
        pred_cindex = []
        preds_t = []

        for i in [12, 24, 36, 48, OUT_hi.shape[-1]]:
            preds = tf.math.argmax(pil_matrix[:, :i+1], axis = 1).numpy()
            res = concordance_index(events, preds)
            pred_cindex.append(res)
            preds_t.append(preds)

        return pred_cindex, preds_t, pil_matrix
        


