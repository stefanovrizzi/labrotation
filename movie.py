# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 10:44:05 2019

@author: STEFANO VRIZZI
"""

# Movie from processed signal. This function is useful to create short videos to see how EMG activity evolved
# in time. Signal power is shown as heat map for each channel, as array configuration. The suitable frame rate per second and video duration can be set in the lines below.

# The frames generated can then be assembled using ImageJ (openly accessible software)
# (also downloaded in main folder). The frames are saved in the 'img/video' folder of the given subject.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def movie_mak(df, n_ch, fs, chN_to_drop, trialN, folder):
    
    fps = int(fs/48) # frames per second, fs is the EMG sampling rate
    
    plt.ioff()
    
    vmax = 500 # max value to plot heat map of activity

    X = np.zeros((df['t_steps'].loc[trialN] , (n_ch**2)*3))
    
    # plot only signal in channel kept; white colour for channels removed before
    ch_keep = np.arange(1, (n_ch**2)*3+1)
    ch_keep[np.array(chN_to_drop)-1] = 0
    ch_keep = ch_keep[ch_keep!=0]
    
    for chN in ch_keep:
        X[:, chN-1] = df['ch'+str(chN)+'_filt'].loc[trialN]

    for time in range(0, df['t_steps'].loc[trialN], fps):
    
        fig, axn = plt.subplots(1, 3, sharey=True, figsize=(16,5))
        cbar_ax = fig.add_axes([.91, .3, .03, .4])
        
        for i, ax in enumerate(axn.flat):
            
            x = X[time, (n_ch**2)*i:(n_ch**2)*(i+1)] # temporary
            Z = np.array(x).reshape((n_ch,n_ch)) # temporary reshape for array layout
            ch_idx = np.linspace(((n_ch**2)*i)+1,((n_ch**2)*(i+1)),(n_ch**2)).astype(int).reshape((n_ch,n_ch))
            
            sns.heatmap(Z, ax=ax,
                        annot=ch_idx, annot_kws={'color': 'k'}, fmt='d',
                        xticklabels=False, yticklabels=False,
                        cmap='hot_r', cbar=i == 0,
                        vmin=.0, vmax=vmax,
                        cbar_ax=None if i else cbar_ax)
        
        fig.tight_layout(rect=[0, 0, .9, 1])
        
        fig.savefig(folder+'img/movie/movie_'+str(time)+'.png', bbox_inches='tight')
        
        plt.close()