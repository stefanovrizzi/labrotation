# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 15:33:47 2018

@author: HIWI
"""

import matplotlib.pyplot as plt
from scipy import signal

df_sort = df.sort_values(['trial','task'], ascending=[1, task_names])
    
for chN, ch in enumerate(col_filt):

    plt.figure(figsize=(20, 20))

    for trialN in range(len(df.index)): #each trial, columns

        idx = [i for i, s in enumerate(task_names) if df_sort['task'].iloc[trialN] in s]

        ax = plt.subplot(5, len(task_names), trialN+1)

        x = df_sort[ch].iloc[trialN]
        
        f, t, Sxx = signal.spectrogram(x, fs, window=('tukey', 1), nperseg=fs, noverlap=fs*.01)

        im = ax.pcolormesh(t, f, Sxx, cmap='hot', vmax=8000, vmin=0)
        ax.set_ylabel('Frequency [Hz]')
        ax.set_ylim([0,20])

        #divider = make_axes_locatable(ax)
        #cax = divider.append_axes("right", size="2%", pad=0.05)
        #plt.colorbar(im, cax=cax)

        if df_sort['trial'].iloc[trialN] == 0:
            ax.set_title(task_names[idx[0]])

        ax.set_axis_off()
        
    plt.savefig(folder+'spectrogram_'+ch+'.png', bbox_inches='tight')