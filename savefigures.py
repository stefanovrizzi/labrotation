# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 20:45:11 2018

@author: HIWI
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Save figure files

# Histograms to assess channel viability. In red, channels with spikes blasting over the signal trend; in blue, noisy channels.
# In black, viable channels. Please check time course with next function ('traces').
# This function also saves the list of viable and faulty channels.

def histogram_ch(df, n_ch_tot, ch_col, file_ID, folder):
    
    ch_list = []
    chN_to_drop = []
    
    for ch in ch_col:
        temp = []
        for i in df[ch].tolist():
            temp += i
            
        #temp=[abs(i) for i in temp]
            
        ch_list.append(temp)
    
    plt.ioff()
    
    for i in range(int(n_ch_tot/25)+1):
    
        plt.figure(figsize=(12,12))
    
        for chN, ch in enumerate(ch_col[25*i:25*(i+1)]): #each channel, rows
    
            #ax = plt.subplot(n_ch, n_ch, chN+1)
            ax = plt.subplot(5, 5, chN+1)
    
            counts = ax.hist(abs(np.array(ch_list[25*i+chN])), bins=30, density=False, color='k', log=True)
            
            # re-colour faulty channels
            if counts[1][-1] <= 250 : # noisy channels
                counts = ax.hist(abs(np.array(ch_list[25*i+chN])), bins=30, density=False, color='b', log=True)
                chN_to_drop.append(25*i+chN+1)
                
            if counts[1][-1] >= 8000 : # channels with spikes blasting over the signal trend
                counts = ax.hist(abs(np.array(ch_list[25*i+chN])), bins=30, density=False, color='r', log=True)
                chN_to_drop.append(25*i+chN+1)
            
            #counts = ax.hist([1,2,3], bins=30, density=False, color='k', log=True)
            #ax.set_xlabel('Voltage (mV)')
            #ax.set_ylabel('Frequency')
            ax.set_title(ch)
    
        plt.tight_layout()
    
        plt.savefig(folder+'img/hist_'+str(i+1)+file_ID+'.png', bbox_inches='tight')
        plt.close()
        
    # Save list of viable and faulty channels
    with open(folder+'ch_drop.txt', 'w') as f:
        for i in chN_to_drop:
            f.write('%d \n' % i)
            
    with open(folder+'ch_drop_updated.txt', 'w') as f:
        for i in chN_to_drop:
            f.write('%d \n' % i)



# Display rectified signals for each channel, see Lab report Figure 2.2
def traces(df, task_names, ch_col, chN_to_drop, file_ID, folder):
    
    df_sort = df.sort_values(['trial','task'], ascending=[1, task_names])
    col = ['k','b','r','g','c','m','y']
    
    plt.ioff()
    
    for chN, ch in enumerate(ch_col):
    
        plt.figure(figsize=(20, 20))
    
        for trialN in range(len(df.index)): #each trial, columns
    
            idx = [i for i, s in enumerate(task_names) if df_sort['task'].iloc[trialN] in s]
    
            ax = plt.subplot(5, len(task_names), trialN+1)
            
            x = df_sort[ch].iloc[trialN]
            x = abs(np.array(x))
            
            ax.plot(x, color=col[idx[0]], linewidth=.5)
            ax.set_ylim([0, 3500])
    
            if df_sort['trial'].iloc[trialN] == 0:
                ax.set_title(task_names[idx[0]])
    
            ax.set_axis_off()
        
        if chN+1 in chN_to_drop:
            plt.savefig(folder+'img/drop_'+ch+file_ID+'.png', bbox_inches='tight')
        else:
            plt.savefig(folder+'img/'+ch+file_ID+'.png', bbox_inches='tight')
        
        plt.close()


# Display rectified traces overlaid by smoothed traces. Each channel is one figure
def filt_traces(df, task_names, ch_col, file_ID, folder):
    
    df_sort = df.sort_values(['trial','task'], ascending=[1, task_names])
    
    plt.ioff()
    
    for chN, ch in enumerate(ch_col):
    
        plt.figure(figsize=(20, 20))
    
        for trialN in range(len(df.index)): #each trial, columns
    
            idx = [i for i, s in enumerate(task_names) if df_sort['task'].iloc[trialN] in s]
    
            ax = plt.subplot(5, len(task_names), trialN+1)
            
            x = df_sort[ch+'_rct'].iloc[trialN]
            filt = df_sort[ch+'_filt'].iloc[trialN]
            
            ax.plot(x, color='k')
            ax.plot(filt, color='r')
            ax.set_ylim([0, 3500])
    
            if df_sort['trial'].iloc[trialN] == 0:
                ax.set_title(task_names[idx[0]])
    
            ax.set_axis_off()
        
        plt.savefig(folder+'img/'+ch+'_gfilt'+file_ID+'.png', bbox_inches='tight')
        plt.close()
        
# Spectrograms of single channels, all trials and tasks - Figure 3.2 in lab report
        
def spectrograms(df, fs, nperseg, task_names, col_filt, file_ID, folder):
    
    df_sort = df.sort_values(['trial','task'], ascending=[1, task_names])
    
    plt.ioff()
    
    for chN, ch in enumerate(col_filt):
    
        plt.figure(figsize=(20, 20))
    
        for trialN in range(len(df.index)): #each trial, columns
    
            idx = [i for i, s in enumerate(task_names) if df_sort['task'].iloc[trialN] in s]
    
            ax = plt.subplot(5, len(task_names), trialN+1)
    
            x = df_sort[ch].iloc[trialN]
            
            f, t, Sxx = signal.spectrogram(x, fs, window=('tukey', 1), nperseg=nperseg, noverlap=fs*.01)
    
            ax.pcolormesh(Sxx, cmap='hot', vmax=8000, vmin=0)
            ax.set_ylim([0,20])
    
            #divider = make_axes_locatable(ax)
            #cax = divider.append_axes("right", size="2%", pad=0.05)
            #plt.colorbar(im, cax=cax)
    
            if df_sort['trial'].iloc[trialN] == 0:
                ax.set_title(task_names[idx[0]])
    
            ax.set_axis_off()
            
        plt.savefig(folder+'img/'+ch+'_spectrogram'+file_ID+'.png', bbox_inches='tight')
        plt.close()