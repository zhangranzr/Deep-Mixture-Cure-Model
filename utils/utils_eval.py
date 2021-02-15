import tensorflow as tf
from utils_lossFun import *
 

def train_step(batch_index:list, input, pat_info, MASK, 
censoring_status, payoff_status, model, sigma1:float, MIX:bool, RNN_net:bool, alpha:float, _EPSILON: float):

    with tf.GradientTape() as tape:
        if MIX:
            OUT_hi, OUT_pi = model(inputs = input)  #output shape (batch_size, 1)
            if RNN_net:
                OUT_hi = tf.clip_by_value(OUT_hi[:, :, -1], _EPSILON, 1-_EPSILON)
        else:
            OUT_hi = model(inputs = input)
            if RNN_net:
                OUT_hi = tf.clip_by_value(OUT_hi[:, :, -1], _EPSILON, 1-_EPSILON)
            OUT_pi = []
        
        # make a prediction using the model and then calculate the
        # loss
        get_loss_fun  = get_loss(batch_index, pat_info, MASK, OUT_pi, OUT_hi, censoring_status,
        payoff_status, RNN_net = RNN_net, sigma1 = sigma1)

        if MIX:
            loss_m, loss_m_detail = get_loss_fun.loss_likelihood_mixture()
        else:
            loss_m, loss_m_detail = get_loss_fun.loss_likelihood_non_mixture()

        loss_ranking = get_loss_fun.loss_Ranking()

        if alpha == 1:
            loss = loss_m
        else:    
            loss = tf.multiply(tf.constant(alpha), loss_m) + tf.multiply(tf.constant(1-alpha), loss_ranking)
        pred_cindex, preds_t, pil_matrix = get_loss_fun.c_index()

    gradients = tape.gradient(loss, model.trainable_variables)
    
    moniter_batch = []
    if MIX:
        moniter_batch.append([loss_m_detail, get_loss_fun.loss_likelihood_non_mixture(), loss_m.numpy(),
    loss_ranking.numpy(), get_loss_fun.AUC().numpy(), get_loss_fun.ANLP().numpy(), pred_cindex[-1]])
    else:
        moniter_batch.append([loss_m_detail, get_loss_fun.loss_likelihood_non_mixture(), loss_m.numpy(),
    loss_ranking.numpy(), 0, get_loss_fun.ANLP().numpy(), pred_cindex[-1]])

    return loss, moniter_batch, gradients

def test_step(batch_index:list, input, pat_info, MASK, censoring_status, payoff_status, model,
    MIX:bool, RNN_net:bool, sigma1:float, alpha:float, _EPSILON: float):

    if MIX:
        OUT_hi, OUT_pi = model(inputs = input)  #output shape (batch_size, 1)     
        if RNN_net:
                OUT_hi = tf.clip_by_value(OUT_hi[:, :, -1], _EPSILON, 1-_EPSILON)   
    else:
        OUT_hi = model(inputs = input)
        if RNN_net:
            OUT_hi = tf.clip_by_value(OUT_hi[:, :, -1], _EPSILON, 1-_EPSILON)
        OUT_pi = []
    
    get_loss_fun  = get_loss(batch_index, pat_info, MASK, OUT_pi, OUT_hi, censoring_status,
    payoff_status, RNN_net = RNN_net, sigma1 = sigma1)
    pred_cindex, preds_t, pil_matrix = get_loss_fun.c_index()

    if MIX:
        loss_m, loss_m_detail = get_loss_fun.loss_likelihood_mixture()
    else:
        loss_m, loss_m_detail = get_loss_fun.loss_likelihood_non_mixture()    

    loss_ranking = get_loss_fun.loss_Ranking()

    if alpha == 1:
        loss = loss_m
    else:    
        loss = alpha*loss_m + (1-alpha)*loss_ranking
    
    # performance moniter:
    moniter_batch = []
    if MIX:
        moniter_batch.append([loss_m_detail, get_loss_fun.loss_likelihood_non_mixture(), loss_m.numpy(),
    loss_ranking, get_loss_fun.AUC().numpy(), get_loss_fun.ANLP().numpy(), pred_cindex[-1]])
    else:
        moniter_batch.append([loss_m_detail, get_loss_fun.loss_likelihood_non_mixture(), loss_m.numpy(),
    loss_ranking, 0, get_loss_fun.ANLP().numpy(), pred_cindex[-1]])
            

    return loss.numpy()/len(batch_index), moniter_batch

