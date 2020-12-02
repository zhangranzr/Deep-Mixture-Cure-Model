from lifelines.utils import concordance_index
def c_index(batch_index, OUT_h_i_, pat_info, RNN = False):
    pil_matrix = np.zeros(OUT_h_i_.shape)
    pil_matrix[:, 0] = OUT_h_i_[:, 0]
    for i in range(n_timesteps):
        if i == 0:
            pass
        else:
            temp_h = OUT_h_i_[:, i]
            temp_h_1 = tf.math.subtract(1, OUT_h_i_)[:, :i-1]
            h = tf.reduce_prod(temp_h_1, axis = 1)
            pil_matrix[:, i] = tf.math.multiply(temp_h, h).numpy()

    names = batch_index
    events = pat_info[:, 3][batch_index]
    pred_cindex = []
    preds_t = []
    for i in [12, 24, 36, 48]:
        preds = tf.math.argmax(pil_matrix[:, :i], axis = 1).numpy()
        res = concordance_index(events, preds)
        pred_cindex.append(res)
        preds_t.append(preds)
    return pred_cindex, preds_t, pil_matrix, 

#def compute_loss

#def AUC