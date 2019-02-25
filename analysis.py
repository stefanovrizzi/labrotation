# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 12:00:36 2018

@author: STEFANO VRIZZI
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
import save_files

# CHOOSE INITIAL PARAMETERS

sbj = 1 #subject number
t_seg = 't_seg' #time segmentation file
t_W = 1 #number of time bins of interest in the spectrogram analysis
f_W = 5 #number of frequency bands of interest in the spectrogram analysis
c = 0.025 #penalty term in classifier

##########################

# Useful parameters - do not change

fs = 2048 #Hz, sampling frequency
fsv = 120 #Hz, frame per second in VICON
n_ch = 8 #channles in one row (or column) of one array
n_ch_tot = (n_ch**2)*3 #total channels

##########################

# Useful names for folder and naming files and plots generated, according to initial parameters set

folder = '20181024_KONSENS_recordings/S'+str(sbj)+'/' # choose subject
file_ID_wp = "_S"+str(sbj)+"_C"+str(c)+"_F"+str(f_W)+"_T"+str(t_W)+"_"+t_seg # File name With Parameters
file_ID = "_S"+str(sbj)+'_'+t_seg # File name

##########################

# Choose tasks

task_names = [#'iso',
              'bottle',
              'screwdriver',
              'knife',
              'hammer',
              'peg',
              #'free_eating',
              'jar', 
              'typing']
              #'swing']
task_names.sort()

##########################

# TIME SEGMENTATION
# Load segmentation times for EMG traces from kinematics to keep signal of interest

with open(folder+t_seg+'.txt') as f:
    seg = [[int(x) for x in line.split()] for line in f if not line.startswith("#")]

##########################

# DATAFRAME (df)
# Create df and its columns, fill it with EMG, trial number, task label and time duration
    
import df_setup

df, ch_col = df_setup.fill_df(n_ch_tot, n_ch, seg, task_names, folder, fs, fsv)                

##########################

# HISTOGRAMS OF EACH CHANNEL
# Useful to investigate faulty channels by voltage range (0-6000 mV as acceptable).

import savefigures

#The following line saves the histograms with all the absolute values recorded by each channel.
#It also highlights which channels are considered faulty (red) and noise only (blue)

# UNCOMMENT IF NEEDED
#savefigures.histogram_ch(df, n_ch_tot, ch_col, folder)

##########################

# PRE-PROCESSING OF EMG TRACES: REMOVE BAD CHANNELS, RECTIFY, SMOOTH

import signal_processing

df, ch_col, n_ch_tot, chN_to_drop = signal_processing.drop_channels(df, folder) # DROP CHANNELS WITH ARTIFACTS

df, col_rct = signal_processing.rectify(df, n_ch_tot, ch_col) # RECTIFY SIGNALS

window_length = int(fs*1)
        
df, col_filt = signal_processing.gauss_filtering(df, n_ch_tot, ch_col, fs, col_rct, window_length) # SMOOTHING TRACES CONVOLVING GAUSSIAN WINDOW

##########################

# Save dataframe before adding features - UNCOMMENT IF NEEDED
# df_file_ID = folder+"saved_files/"+'df'+file_ID
# save_files.save_wp(df, df_file_ID)

##########################

# RECTIFIED AND SMOOTHED TRACES FOR EACH CHANNEL
# Useful to investigate reproducibility of traces.

nperseg = int(fs*1) #number of sample points in window to decompose signal into spectrogram

#The following lines save plots for each channel kept, plotting all their trials and tasks.
#The first line plots the voltage traces, the second one the spectrograms.

# UNCOMMENT IF NEEDED

#savefigures.traces(df, fs, task_names, ch_col, folder)

#savefigures.spectrograms(df, fs, nperseg, task_names, col_filt, folder)

##########################

# CLASSIFICATION

import classification

# FILL DATAFRAME WITH PIXELS FROM TIME-FREQUENCY DECOMPOSITION FROM EACH CHANNEL
df_class, features = classification.feature_extraction(df, t_W, f_W, n_ch_tot, col_filt, fs, nperseg)     

# Save dataframe with extracted features - UNCOMMENT IF NEEDED
# df_file_ID = folder+'saved_files/'+'df_class'+file_ID+"_F"+str(f_W)+"_T"+str(t_W)
# save_files.save_wp(df_class, df_file_ID)

# Choose classifier
clf = SVC(kernel="linear", C=c)

# Set features and labels

X = np.array(df_class[features])
y = df_class['task']

# Evaluate K fold cross validation

K = 5 # K-fold

score = classification.K_fold_CV(X, y, df_class, clf, K, task_names, file_ID_wp, folder)

##########################

# Recursive feature selection

rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(K, shuffle=True), scoring='accuracy')
rfecv.fit(X, y)
print("Optimal number of features : %d" % rfecv.n_features_)


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

# MINIMUM CHANNEL ANALYSIS

# Extracts quantities related to minimum set of channels needed for maximal mean accuracy

import min_ch_necessary
                    
ch_imp_idx_sorted = min_ch_necessary.ch_imp_idx(ch_imp_reshape) #index of channels order by decreasing importance
# compute accuracy scores for each set of channels
mean_score, std_score, Score = min_ch_necessary.Kfold(df_class, clf, y, n_ch_tot, ch_imp_reshape, features, t_W, f_W, ch_imp_idx_sorted, K)

import perm_channels
# It computes when the trend increments significantly across sets of channels, comparing them by pairs
p_value, significance, alpha, index_changes = perm_channels.trend_diff(Score, n_ch_tot-1)

# Minimum channel necessary - useful variables: channel number when max mean accuracy starts and ends
min_ch, stop_plateau = min_ch_necessary.min_ch_function(index_changes, Score, n_ch_tot)

# Creating the list to pool all values along the plateau, to compute mean and SD
score_flat = [item for sublist in Score[(min_ch-1):stop_plateau] for item in sublist]

# Extended set of minimum number of channels: either 3 channels more or 20% more channels, whichever is larger
if int(min_ch*1/5) < 3:
    min_ch_comp = min_ch + 3    
else:
    min_ch_comp = int(min_ch*6/5)

# Compute max mean accuracy, SD and channel list of the extended set of channels
max_mean_accuracy = np.mean(score_flat)
max_mean_accuracy_sd = np.std(score_flat)
min_ch_list = ch_imp_idx_sorted[0:min_ch_comp]
min_ch_fraction = min_ch/n_ch_tot

# Variables from minimum channel analysis
significant_changes = [p_value, significance, alpha, index_changes,
                       min_ch, stop_plateau,
                       max_mean_accuracy,
                       max_mean_accuracy_sd,
                       min_ch_list,
                       min_ch_fraction]

############################
                
# DISPLACEMENT ANALYSIS

# Computing two example sets of displaced channels
ch_imp_idx_displacement = min_ch_necessary.displacement(ch_imp_reshape, min_ch_comp, min_ch_list, n_ch)
mean_score_displacement, std_score_displacement, Score_displacement = min_ch_necessary.Kfold(df_class, clf, y, min_ch_comp, ch_imp_reshape, features, t_W, f_W, ch_imp_idx_displacement, K)

ch_imp_idx_displacement2 = min_ch_necessary.displacement(ch_imp_reshape, min_ch_comp, min_ch_list, n_ch)
mean_score_displacement2, std_score_displacement2, Score_displacement2 = min_ch_necessary.Kfold(df_class, clf, y, min_ch_comp, ch_imp_reshape, features, t_W, f_W, ch_imp_idx_displacement2, K)

#####

# Permutation tests against control group (no displacement)
p_displacement, significance_diplacement, alpha_displacement = perm_channels.displacement_diff(Score_displacement, Score, min_ch_comp)
p_displacement2, significance_diplacement2, alpha_displacement2 = perm_channels.displacement_diff(Score_displacement2, Score, min_ch_comp)


# Plots
# One example of displacement
min_ch_necessary.plot_displacement(mean_score, std_score, mean_score_displacement, std_score_displacement, min_ch_comp, K, file_ID_wp, folder)

# Two examples of displacement
min_ch_necessary.plot_displacement2(mean_score, std_score,
                                    mean_score_displacement, std_score_displacement,
                                    mean_score_displacement2, std_score_displacement2,
                                    min_ch_comp, K, file_ID_wp, folder)

############################

# SAVE FILES

# Save data
data = [sbj, ch_col, n_ch, fs, chN_to_drop, n_ch_tot, col_filt, nperseg, task_names, folder, clf, Score, features, X, y, K, ch_imp_idx_sorted]
data_file_ID = folder+"saved_files/"+"data"+file_ID_wp
save_files.save_wp(data, data_file_ID)
                
#Save data recursive feature selection
rfecv_file_ID = folder+"saved_files/"+"rfecv"+file_ID_wp
save_files.save_wp(rfecv, rfecv_file_ID)

# Variables from minimum channel analysis
significant_changes = [p_value, significance, alpha, index_changes,
                       min_ch, stop_plateau,
                       max_mean_accuracy,
                       max_mean_accuracy_sd,
                       min_ch_list,
                       min_ch_fraction]

# Save variables
significant_changes_file_ID = folder+"saved_files/"+"significant_changes"+file_ID_wp
save_files.save_wp(significant_changes, significant_changes_file_ID)

# Save variables for displacement analysis
displacement = [Score, Score_displacement, Score_displacement2,
                mean_score, std_score,
                mean_score_displacement, std_score_displacement,
                mean_score_displacement2, std_score_displacement2,
                ch_imp_idx_sorted, min_ch_list,
                ch_imp_idx_displacement, ch_imp_idx_displacement2,
                p_displacement, significance_diplacement, alpha_displacement,
                p_displacement2, significance_diplacement2, alpha_displacement2]

displacement_file_ID = folder+"saved_files/"+"displacement"+file_ID_wp
save_files.save_wp(displacement, displacement_file_ID)

############################

# Movie from processed signal of a given trial. This function is useful to create short videos to see how EMG activity evolved
# in time. The suitable frame rate per second and video duration can be set in the file 'movie.py'.

# The frames generated can then be assembled using ImageJ (openly accessible software)
# (also downloaded in main folder). The frames are saved in the 'img/video' folder of the given subject.

import movie

trialN = 1

movie.movie_mak(df, n_ch, fs, chN_to_drop, trialN, folder)
