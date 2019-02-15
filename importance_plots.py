# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 13:33:55 2018

@author: HIWI
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

##########################
# Plot number of features VS. cross-validation scores

def n_features_vs_CV_scores(rfecv, file_ID_wp, folder):
    
    plt.ioff()
    
    plt.figure(figsize=(12,6))
    ax = plt.subplot(111)
    ax.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, color='k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("Number of features selected")
    ax.set_ylabel("Cross validation score (%)")
    
    plt.savefig(folder+'img/feature_sel'+file_ID_wp+'.png', bbox_inches='tight')
    
    plt.close()
    #plt.show()

##########################
# Plot heatmap of pixel importance as channel number (row) and pixel number (column): useful to see range of frequencies

def pixel_importance(pixel_imp_reshape, file_ID_wp, folder, ch_col): 
    
    plt.ioff()
    
    plt.figure(figsize=(8,6))
    plt.pcolor(pixel_imp_reshape, cmap='jet')
    clb = plt.colorbar()
    clb.set_label('Pixel importance')
    plt.xlabel('pixel #')
    plt.ylabel('Channel #')
    #plt.yticks(np.arange(len(ch_col))+.5, ch_col, size='small')
    
    plt.savefig(folder+'img/pixel_importance'+file_ID_wp+'.png', bbox_inches='tight')
    
    plt.close()
    #plt.show()

##########################
# Plot channel number (sorted) VS. cross-validation scores: useful to see range of channel importance

def channel_importance_sorted(ch_imp, file_ID_wp, folder, n_ch_tot):
    
    plt.ioff()
    
    plt.figure(figsize=(12,6))
    ax = plt.subplot(111)
    ax.scatter(range(1,len(ch_imp)+1),-np.sort(-ch_imp), color='k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Channels (sorted)')
    ax.set_ylabel('Channel importance')
    ax.set_xticks([1, 50, 100, n_ch_tot])
    
    plt.savefig(folder+'img/ch_importance'+file_ID_wp+'.png', bbox_inches='tight')
    
    plt.close()
    #plt.show()

##########################
# Plot channel importance in array configuration: useful to see range of channel importance

def array_importance(ch_imp_reshape, n_ch, file_ID_wp, folder):
    
    plt.ioff()
        
    fig, axn = plt.subplots(1, 3, sharey=True, figsize=(16,5))
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    
    for i, ax in enumerate(axn.flat):
        
        t = ch_imp_reshape[(n_ch**2)*i:(n_ch**2)*(i+1)] # temporary
        Z = np.array(t).reshape((n_ch,n_ch)) # temporary reshape for array layout
        ch_idx = np.linspace(((n_ch**2)*i)+1,((n_ch**2)*(i+1)),(n_ch**2)).astype(int).reshape((n_ch,n_ch))
        
        sns.heatmap(Z, ax=ax,
                    annot=ch_idx, annot_kws={'color': 'k', "size": 16},
                    cbar_kws={'label': 'Channel importance'},
                    xticklabels=False, yticklabels=False, fmt='d',
                    cmap='hot_r', cbar=i == 0,
                    vmin=.0, vmax=1,
                    cbar_ax=None if i else cbar_ax)
        ax.set_title('Array '+str(i+1))
    
    #fig.tight_layout(rect=[0, 0, .9, 1])
    
    fig.savefig(folder+'img/arrays_importance'+file_ID_wp+'.png', bbox_inches='tight')
    
    plt.close()
    #plt.show()

