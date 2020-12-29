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

    def __init__(self, data_original, feature_list, 
    binary_f = [], 
    continuous_f = []):

        self.data_original = data_original
        self.pat_info      = None
        self.mask1         = None
        self.mask2         = None
        self.normalize     = None
        self.feat_list     = feature_list
        self.binary_f      = binary_f 
        self.continuous_f  = continuous_f

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
        max_measurment = int(math.ceil(df.t.max()*1.2)) #max duration


        data     = np.zeros([len(id_list), max_measurment, len(feat_list)+1])
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
                data[i, each_t, 1:]  = tmp[tmp['t'] == each_t][feat_list]
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

        mask1_rnn       = self.get_mask_rnn(data, pat_info)
        mask_ranking    = self.get_mask_ranking(time, -1, num_Category)

        #Dimension       = (x_dim, x_dim_cont, x_dim_bin)
        DATA            = (data, time, label, payoff_label)
        MASK            = (mask1_rnn, mask_ranking)

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