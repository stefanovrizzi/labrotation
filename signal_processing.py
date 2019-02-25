# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 15:22:16 2018

@author: STEFANO VRIZZI
"""

import pandas as pd
from scipy import signal

#Drop faulty channels

def drop_channels(df, folder):
    
    #with open(folder+'ch_drop_updated.txt') as f:
    #    chN_to_drop = [int(x) for x in f]
    
    with open(folder+"ch_drop_updated.txt", "r", encoding="utf-8") as g:
        chN_to_drop = list(map(int, g.readlines()))
    
    ch_to_drop = [] # list of strings for channel names
    
    for i in range(len(chN_to_drop)): #list of channels to be dropped
        ch_to_drop.append('ch'+str(chN_to_drop[i]),)
    
    df = df.drop(ch_to_drop, axis=1)
    
    ch_col = df.columns[ df.columns.get_loc('time_duration')+1 : df.columns.get_loc('arrays')]
    
    n_ch_tot = len(ch_col)
    
    return df, ch_col, n_ch_tot, chN_to_drop

##########################

# Rectify (full-wave rectifier)

def rectify(df, n_ch_tot, ch_col):
    
    import numpy as np
    
    col_rct = []
    for i in range(n_ch_tot): #create new columns to store processed signal
        col_rct.append(ch_col[i]+'_rct',)
        
    df = pd.concat([df,pd.DataFrame(columns=col_rct)], sort=False)
    
    df[col_rct] = [[abs(np.array(df[ch][trialN])) for ch in ch_col] for trialN in df.index]
    
    return df, col_rct

##########################

# Smooth (Gaussian filter)
def gauss_filtering(df, n_ch_tot, ch_col, fs, col_rct, window_length):
    
    col_filt = []
    for i in range(n_ch_tot): #create new columns to store processed signal
        col_filt.append(ch_col[i]+'_filt',)
        
    df = pd.concat([df,pd.DataFrame(columns=col_filt)], sort=False)
    
    window = signal.gaussian(window_length, std=50) #one second window
    
    df[col_filt] = [[signal.convolve(df[ch][trialN], window, mode='same', method='direct') / sum(window)
                     for ch in col_rct] for trialN in df.index]
    
    return df, col_filt