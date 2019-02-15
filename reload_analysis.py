# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 15:44:46 2019

@author: HIWI
"""

import numpy as np
import pickle

sbj = 1
c = 0.025
t_W = 5
f_W = 10

t_seg = 't_seg'

file_ID_wp = "_S"+str(sbj)+"_C"+str(c)+"_F"+str(f_W)+"_T"+str(t_W)+"_"+t_seg
folder = '20181024_KONSENS_recordings/S'+str(sbj)+'/'

with open(folder+"rfecv"+file_ID_wp, "rb") as fp:   # Unpickling
    rfecv = pickle.load(fp)
    
with open(folder+"data"+file_ID_wp, "rb") as fp:   # Unpickling
    data = pickle.load(fp)
    
with open(folder+"df_S"+str(sbj)+'_'+t_seg, "rb") as fp:   # Unpickling
    df = pickle.load(fp)
    
with open('comparison.txt', "wb") as fp:   #Pickling
    comparison = pickle.load(fp)

#sbj = data[0]
ch_col = data[1]
n_ch = data[2]
fs = data[3]
chN_to_drop = data[4]
n_ch_tot = data[5]
col_filt = data[6]
nperseg = data[7]
task_names = data[8]
#folder = data[9]
clf = data[10]
score = data[11]
features = data[12]
X = data[13]
y = data[14]
K = data[15]
ch_imp_idx_sorted = data[16]
ch_imp_idx_displacement = data[17]
  
##########################
# Importance of features and channels

import importance_measures

pixel_imp, pixel_imp_reshape, ch_imp, ch_imp_reshape = importance_measures.imp(rfecv, t_W, f_W, n_ch_tot, chN_to_drop)

##########################

# Plot importance of features and channels

import importance_plots

importance_plots.n_features_vs_CV_scores(rfecv, file_ID_wp, folder)
importance_plots.pixel_importance(pixel_imp_reshape, file_ID_wp, folder, ch_col)
importance_plots.channel_importance_sorted(ch_imp, file_ID_wp, folder, n_ch_tot)
importance_plots.array_importance(ch_imp_reshape, n_ch, file_ID_wp, folder)

############################

import min_ch_necessary

ch_imp_idx_sorted = min_ch_necessary.ch_imp_idx(ch_imp_reshape)
mean_score, std_score, Score = min_ch_necessary.Kfold(df, clf, y, n_ch_tot, ch_imp_reshape, features, t_W, f_W, ch_imp_idx_sorted, K)

# Minimum channel necessary - useful variables
min_ch = np.argmax(mean_score)+1
max_mean_accuracy = np.max(mean_score)
max_mean_accuracy_sd = std_score[min_ch-1]
min_ch_list = ch_imp_idx_sorted[0:min_ch]
min_ch_fraction = min_ch/n_ch_tot

ch_imp_idx_displacement = min_ch_necessary.displacement(ch_imp_reshape, min_ch, min_ch_list, n_ch)
mean_score_displacement, std_score_displacement, Score_displacement = min_ch_necessary.Kfold(df, clf, y, min_ch, ch_imp_reshape, features, t_W, f_W, ch_imp_idx_displacement, K)

# Plots
min_ch_necessary.plot(n_ch_tot, mean_score, std_score, mean_score_displacement, std_score_displacement, file_ID_wp, folder)
min_ch_necessary.conf_m(df, clf, y, n_ch_tot, ch_imp_reshape, features, t_W, f_W, ch_imp_idx_sorted, min_ch, task_names, file_ID_wp, folder)

#df = pd.DataFrame(np.array(comparison).reshape(3,3), columns = list("abc"))