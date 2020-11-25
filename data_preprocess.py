#Reference: C. Lee, J. Yoon, M. van der Schaar, 
#"Dynamic-DeepHit: A Deep Learning Approach for Dynamic Survival Analysis With Competing Risks Based on Longitudinal Data," 
#IEEE Transactions on Biomedical Engineering (TBME). 2020

import pandas as pd
import numpy as np


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
            time : time at which observations are measured, starts time must be sycronized
            label: censoring information
                - 0: censoring
                - 1: event type 1
                - 2: event type 2
        output:
            data: array format of dimension (n_individuals, timestamps, n_features)
            pat_info: data specific measuring information
        '''
        df = self.data_original
        feat_list = self.feat_list

        grouped  = df.groupby(['id'])
        id_list  = pd.unique(df['id'])
        max_measurment = np.max(grouped.count())[0]

        data     = np.zeros([len(id_list), max_measurment, len(feat_list)+1])
        pat_info = np.zeros([len(id_list), 5])

        for i, tmp_id in enumerate(id_list):
            tmp = grouped.get_group(tmp_id).reset_index(drop=True)

            pat_info[i,4] = tmp.shape[0]            # number of measurements
            pat_info[i,3] = np.max(tmp['time'])     # last measurement time
            pat_info[i,2] = tmp['label'][0]         # cause
            pat_info[i,1] = np.max(tmp['time'])     # time to event
            pat_info[i,0] = tmp['id'][0]      

            data[i, :int(pat_info[i, 4]), 1:]  = tmp[feat_list]
            data[i, :int(pat_info[i, 4]-1), 0] = np.diff(tmp['time'])
            # return the interval of diff measurement times
            
            self.data = data
            self.pat_info = pat_info

        return self.pat_info, self.data


    def preprocess(self, 
    norm_mode = 'standard'):

        df_                = self.data_original

        bin_list           = self.binary_f# ['drug', 'sex', 'ascites', 'hepatomegaly', 'spiders']
        cont_list          = self.continuous_f #['age', 'edema', 'serBilir', 'serChol', 'albumin', 'alkaline', 'SGOT', 'platelets', 'prothrombin', 'histologic']
        
        if len(bin_list) + len(cont_list):
            feat_list = self.feat_list
        else:    
            feat_list      = cont_list + bin_list
        df_                = df_[['id', 'time', 'label']+feat_list]
        df_org_            = df_.copy(deep=True)

        #df_[cont_list]     = f_get_Normalization(np.asarray(df_[cont_list]).astype(float), norm_mode)

        pat_info, data     = self.construct_dataset()

        data_mi                  = np.zeros(np.shape(data))
        data_mi[np.isnan(data)]  = 1
        data[np.isnan(data)]     = 0 

        x_dim           = np.shape(data)[2] 
        # 1 + x_dim_cont + x_dim_bin (including delta)
        x_dim_cont      = len(cont_list)
        x_dim_bin       = len(bin_list) 

        last_measurement= self.pat_info[:,[3]]  #pat_info[:, 3] contains age/month at the last measurement
        label           = self.pat_info[:,[2]]  #status 0,1,2,....
        time            = self.pat_info[:,[1]]  #time when event occurred or censoring time
        
        num_Category    = int(np.max(self.pat_info[:, 1]) * 1.2) 
        #or specifically define larger than the max tte 
        num_Event       = len(np.unique(label)) - 1

        if num_Event == 1:
            label[np.where(label!=0)] = 1 #make single risk

        mask1           = self.get_mask1(last_measurement, num_Event, num_Category)
        mask2           = self.get_mask2(time, label, num_Event, num_Category)
        mask3           = self.get_mask3(time, -1, num_Category)

        Dimension       = (x_dim, x_dim_cont, x_dim_bin)
        DATA            = (data, time, label)
        MASK            = (mask1, mask2, mask3)

        return Dimension, DATA, MASK, data_mi, pat_info

    def get_mask1(self, meas_time, num_Event, num_Category):
        
        '''
        mask1 is required to get the contional probability (to calculate the denominator part)
        mask1 size is [N, num_Event, num_Category]. 1's until the last measurement time
        formula:

        '''

        mask = np.zeros([np.shape(meas_time)[0], num_Event, num_Category]) # for denominator
        for i in range(np.shape(meas_time)[0]):
            mask[i, :, :int(meas_time[i, 0]+1)] = 1 # last measurement time

        return mask

    def get_mask2(self, time, label, num_Event, num_Category):
        '''
        mask2 is required to get the log-likelihood loss 
        mask2 size is [N, num_Event, num_Category]
        if not censored : one element = 1 (0 elsewhere)
        if censored     : fill elements with 1 after the censoring time (for all events)
        '''
        mask = np.zeros([np.shape(time)[0], num_Event, num_Category]) # for the first loss function
        for i in range(np.shape(time)[0]):
            if label[i,0] != 0:  #not censored
                mask[i,int(label[i,0]-1),int(time[i,0])] = 1
            else: #label[i,2]==0: censored
                mask[i,:,int(time[i,0]+1):] =  1 #fill 1 until from the censoring time (to get 1 - \sum F)
        return mask

    def get_mask3(self, time, meas_time, num_Category):
        '''
        mask5 is required calculate the ranking loss (for pair-wise comparision)
        mask5 size is [N, num_Category]. 
            - For longitudinal measurements:
                1's from the last measurement to the event time (exclusive and inclusive, respectively)
                denom is not needed since comparing is done over the same denom
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