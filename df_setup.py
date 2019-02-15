# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 12:07:44 2018

@author: HIWI

"""
import pandas as pd
from glob import glob
import os
import numpy as np
import scipy.io
    
def fill_df(n_ch_tot, n_ch, seg, task_names, folder, fs, fsv):
    
    columns = ['task','trial', 'array1','array2','array3', 't_steps', 'time_duration']#,
               #'info', 'side','gain','filt_parameters','notes'] #to add further info
    
    for i in range(n_ch_tot): #add channels as predictors
        columns.append('ch'+str(i+1),)
    
    df = pd.DataFrame(columns=columns) #create dataframe
    
    ch_col = df.columns[ df.columns.get_loc('ch1') : df.columns.get_loc('ch1') + n_ch_tot] #select channel columns
    
    file_list = []
    task_list_name = []
    task_list_num = []
    
    for task_num, task_name in enumerate(task_names):
        
        file_list_temp = glob(os.path.join(folder+'recordings/', '*'+task_name+'*.mat'))
        file_list.extend(file_list_temp)
        task_list_name.extend((task_name,)*np.size(file_list_temp))
        task_list_num.extend(range(np.size(file_list_temp)))
                          
    for file_num, file_name in enumerate(file_list):
        print (file_name) #print loaded files
        mat = scipy.io.loadmat(file_name)
        
        t_i = int((seg[file_num][0]-1)/fsv *fs)
        t_f = int((seg[file_num][1]-1)/fsv *fs)
        
        array1_temp = np.array(mat["EMG_array"][0][0]['data'][0][0]) #first or second index? check with trial 1
        array2_temp = np.array(mat["EMG_array"][0][1]['data'][0][0])
        array3_temp = np.array(mat["EMG_array"][0][2]['data'][0][0])
        
        array1 = [] #batches to reshape data in 64 x time_steps matrices
        array2 = []
        array3 = []
            
        for i in range(n_ch):
            for j in range(n_ch):
                array1.append(np.squeeze(array1_temp[i][j][0][t_i:t_f]))
                array2.append(np.squeeze(array2_temp[i][j][0][t_i:t_f]))
                array3.append(np.squeeze(array3_temp[i][j][0][t_i:t_f]))
            
        df = df.append({'array1': array1, 'array2': array2, 'array3': array3,
                        't_steps': t_f-t_i},
                        #'t_steps': np.size(mat["EMG_array"][0][0]['data'][0][0][0][0])},
                        ignore_index=True)
    
    df['task'] = task_list_name
    df['trial'] = task_list_num
    
    df['time_duration'] = [round(df['t_steps'][trialN]/fs, 2) for trialN in df.index] #Add time duration in seconds
    
    df['arrays'] = df[['array1','array2','array3']].apply(lambda x: np.concatenate(x),axis=1) #Adding single channels as columns
    df[ch_col.tolist()] = [df['arrays'][trialN].tolist() for trialN in df.index]
    
    return df, ch_col