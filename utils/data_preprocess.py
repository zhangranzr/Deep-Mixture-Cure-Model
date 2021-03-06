#Reference: C. Lee, J. Yoon, M. van der Schaar, 
#"Dynamic-DeepHit: A Deep Learning Approach for Dynamic Survival Analysis With Competing Risks Based on Longitudinal Data," 
#IEEE Transactions on Biomedical Engineering (TBME). 2020

'''
this step helps to generate the data and mask for further analysis.
This step will not involve standardization or normalization. 
Such data preprocess will take place after train-validation-test split.
'''
import pandas as pd
import numpy as np
import math

class preprocess_data():

    def __init__(self, data_original, feature_list, T_max):

        self.data_original = data_original
        self.feat_list     = feature_list
        self.T_max         = T_max 
        self.pat_info      = None

    def construct_dataset(self):
        
        '''
        input data: 
            id   : indicator of individual 
            tte  : time-to-event or time-to-censoring
                   usually the same as the last obervation time of each individual
            time : time at which observations are measured, start time is the initial recognition of loan 
                    (in loan data, corresponding to column t)
            label: censoring information
                - 0: censoring
                - 1: event type 1
            payoff_label: censoring case, y = 0
        output:
            data: array format of dimension (n_individuals, timestamps, n_features)
            pat_info: data specific measuring information
        '''
        df = self.data_original
        feat_list = self.feat_list
        
        df                 = df[['id', 't', 'label', 'payoff_label']+feat_list]

        grouped  = df.groupby(['id'])
        id_list  = pd.unique(df['id'])
        max_measurment = max(int(math.ceil(df.t.max()*1.2)), self.T_max) #max duration

        data     = np.zeros([len(id_list), max_measurment, len(feat_list)])
        # data dimention (n_customers, durations, n_features) 
        pat_info = np.zeros([len(id_list), 5]) 

        for i, tmp_id in enumerate(id_list):
            tmp = grouped.get_group(tmp_id).reset_index(drop=True)

            #if payoff or not
            pat_info[i,4] = np.max(tmp['payoff_label']) 
            #number of measurements
            pat_info[i,3] = tmp.shape[0]            
            #default_info(payoff is marked as censoring) 
            pat_info[i,2] = np.max(tmp['label'])         
            #time of measurement, snycronized at initial loan recognition 
            pat_info[i,1] = np.max(tmp['t'])     
            #customer id
            pat_info[i,0] = tmp['id'][0]                 

            for each_t in tmp['t']:
                data[i, each_t, :]  = tmp[tmp['t'] == each_t][feat_list]
            #data[i, :int(pat_info[i, 3]-1), 0] = np.diff(tmp['t'])
            # return the interval of diff measurement times
            
        self.data = data
        self.pat_info = pat_info

        return self.pat_info, self.data


    def preprocess(self):
        
        pat_info, data     = self.construct_dataset()

        data_mi                  = np.zeros(np.shape(data))
        data_mi[np.isnan(data)]  = 1
        data[np.isnan(data)]     = 0 

        payoff_label    = self.pat_info[:,[4]]
        duration        = self.pat_info[:,[3]]  #pat_info[:, 3] contains age/month at the last measurement
        label           = self.pat_info[:,[2]]  #status 0,1,2,....
        time            = self.pat_info[:,[1]]  #time when event occurred or censoring time
        
        num_Category    = data.shape[1]
        #or specifically define larger than the max tte 
        num_Event       = len(np.unique(label)) - 1

        if num_Event == 1:
            label[np.where(label!=0)] = 1 #make single risk

        mask_rnn       = self.get_mask_rnn(data, pat_info)
        mask_ranking    = self.get_mask_ranking(time, -1, num_Category)

        #Dimension       = (x_dim, x_dim_cont, x_dim_bin)
        DATA            = (data, time, label, payoff_label)
        MASK            = (mask_rnn, mask_ranking)

        return DATA, MASK, data_mi, pat_info, #Dimension

    def get_mask_rnn(self, data, pat_info):
        
        '''
        mask1 is required to get the likelihood_loss
        formula:

        '''
        mask_1 = np.zeros(data.shape[:2])   # comput multiply until last time point
        for i in range(len(pat_info[:,1])):
            mask_1[i, :int(pat_info[i,1])]      = 1

        mask_2 = np.zeros(data.shape[:2])   # mask at the last time point
        for i in range(len(pat_info[:,1])):
            mask_2[i, int(pat_info[i,1]-1)]     = 1

        mask_3 = np.zeros(data.shape[:2])   # mask before the last time point
        for i in range(len(pat_info[:,1])):
            mask_3[i, :(int(pat_info[i,1]-1))]  = 1

        return (mask_1, mask_2, mask_3)

    def get_mask_ranking(self, time, meas_time, num_Category):
        '''
            mask5 is required calculate the ranking loss (for pair-wise comparision)
            mask5 size is [N, num_Category]. 
            - For longitudinal measurements:
                1's from the last measurement to the event time (exclusive and inclusive, respectively)
            - For single measurement:
                1's from start to the event time(inclusive)
        '''
        mask = np.zeros([np.shape(time)[0], num_Category]) # for the first loss function
        if np.shape(meas_time):  #lonogitudinal measurements 
            for i in range(np.shape(time)[0]):
                t1 = int(meas_time[i, 0]) # last measurement time
                t2 = int(time[i, 0]) # censoring/event time
                mask[i,(t1+1):(t2+1)] = 1  #this excludes the last measurement time and includes the event time
        else:                    #single measurement
            for i in range(np.shape(time)[0]):
                t = int(time[i, 0]) # censoring/event time
                mask[i,:(t+1)] = 1  #this excludes the last measurement time and includes the event time
        return mask


import random
from sklearn import preprocessing
import pickle
from sklearn.model_selection import StratifiedKFold

def CV_preprocess(path, var_selection, T_max, fold, n_splits = 3,):
    '''
    split the dataset into 3 train test folds 
    return split dataset
    '''
    random.seed(116)
    try:
        df = pd.read_pickle(path)
    except:
        df = pd.read_csv(path)

    cv = StratifiedKFold(n_splits=n_splits)
    X = df.groupby('id').count()
    y = df.groupby('id').label.max().values

    #shuffle the data
    train_cv = []
    test_cv  = []

    DATA_train = []
    MASK_train = []
    pat_info_train = []

    DATA_test = []
    MASK_test = []
    pat_info_test = [] 
    
    t = 0
    for train_index, test_index in cv.split(y, y):
        if t == fold:
            print('fold:' + str(fold))
            id_s = X.iloc[train_index].index
            id_mask_train = [True if id in id_s else False for id in df.id]
            id_mask_test  = [False if id in id_s else True for id in df.id]
            train = df[id_mask_train]
            test  = df[id_mask_test]
            
            #preprocess data
            scaler = preprocessing.StandardScaler().fit(train[var_selection])
            train[var_selection] = scaler.transform(train[var_selection])
            test[var_selection]  = scaler.transform(test[var_selection])
            train_cv.append(train)
            test_cv.append(test)

            data_process_train = preprocess_data(data_original = train, feature_list = var_selection, T_max = T_max)
            DATA, MASK, data_mi, pat_info = data_process_train.preprocess()
            DATA_train = DATA
            MASK_train = MASK
            pat_info_train = pat_info 

            data_process_test = preprocess_data(data_original = test, feature_list = var_selection, T_max = T_max)
            DATA_, MASK_, data_mi_, pat_info_ = data_process_test.preprocess()
            DATA_test = DATA_
            MASK_test = MASK_
            pat_info_test = pat_info_
        else:
            pass
            
        t = t+1 

    return (DATA_train, MASK_train, pat_info_train), (DATA_test, MASK_test, pat_info_test), (train_cv, test_cv) 

def DATA_tvt(TRAIN, TEST, info_chunk = 3):
    train_list = []
    test_list  = []
    for i in range(info_chunk):
        train_list.append(TRAIN[i])
        test_list.append(TEST[i])   
    return train_list, test_list