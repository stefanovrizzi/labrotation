# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 15:44:46 2019

@author: STEFANO VRIZZI
"""

# Figures do not show up. They are saved automatically in the img folder of the given subject.
# Figures may not be saved if there are already figures produced with exactly the same parameters.

######################################################

#Load libraries

import numpy as np
import classification
import pickle
import save_files
import min_ch_necessary
import perm_channels

######################################################

#Choose subject number and other parameters, to load the corresponding files

sbj = 1 #Subject number
c = 0.025 #Penalty term of Support Vector Machine
t_W = 1 #Time bin range 0-t_W in spectrogram, to extract window of pixels of interest
f_W = 5 #Frequency bin range 0-f_W in spectrogram, to extract window of pixels of interest

t_seg = 't_seg' #Name of time segmentation file

######################################################

#Useful variables to load and save files according to parameters chosen

file_ID_wp = "_S"+str(sbj)+"_C"+str(c)+"_F"+str(f_W)+"_T"+str(t_W)+"_"+t_seg
folder = '20181024_KONSENS_recordings/S'+str(sbj)+'/' 

######################################################
    
#Load basic data about channels and classification

with open(folder+"saved_files/"+"data"+file_ID_wp+'.txt', "rb") as fp:   # Unpickling
    data = pickle.load(fp)

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
Score = data[11]
features = data[12]
X = data[13]
y = data[14]
K = data[15]
ch_imp_idx_sorted = data[16]
#ch_imp_idx_displacement = data[17]

######################################################

# It loads the dataframe *before* extracting the features from the spectrograms.
# Arbitrary t_W and f_W can be set, to run new analysis

with open(folder+"saved_files/"+"df_S"+str(sbj)+'_'+t_seg+'.txt', "rb") as fp:   # Unpickling
    df = pickle.load(fp)

# Extracts the features (pixels of spectrograms) and updates the dataframe
clf, score, df_class, features, X, y, K = classification.setup(df, t_W, f_W, c, n_ch_tot, col_filt, fs, nperseg, task_names, file_ID_wp, folder)

# Saves the updated dataframe with extracted features
file_ID = "_S"+str(sbj)+'_'+t_seg
df_file_ID = folder+'saved_files/'+'df_class'+file_ID+"_F"+str(f_W)+"_T"+str(t_W)
save_files.save_wp(df_class, df_file_ID)

######################################################

# Loads dataframe *after* extracting the features from the spectrograms, as indicated by t_W and f_W set at the beginning

with open(folder+'saved_files/'+'df_class_S'+str(sbj)+'_'+t_seg+"_F"+str(f_W)+"_T"+str(t_W)+'.txt', "rb") as fp:   # Unpickling
    df_class = pickle.load(fp)

######################################################

# Load recursive feature selection object

with open(folder+"saved_files/"+"rfecv"+file_ID_wp+'.txt', "rb") as fp:   # Unpickling
    rfecv = pickle.load(fp)   

######################################################

# Importance of features and channels

import importance_measures

# It computes pixel and channel importance measures
pixel_imp, pixel_imp_reshape, ch_imp, ch_imp_reshape = importance_measures.imp(rfecv, t_W, f_W, n_ch_tot, chN_to_drop)

# It plots the importance of pixels and channels
import importance_plots

importance_plots.n_features_vs_CV_scores(rfecv, file_ID_wp, folder)
importance_plots.pixel_importance(pixel_imp_reshape, file_ID_wp, folder, ch_col)
importance_plots.channel_importance_sorted(ch_imp, file_ID_wp, folder, n_ch_tot)
importance_plots.array_importance(ch_imp_reshape, n_ch, file_ID_wp, folder)

########################################################

# MINIMUM CHANNEL ANALYSIS
# It computes mean accuracy for incresing sets of channels, adding them by decreasing importance

ch_imp_idx_sorted = min_ch_necessary.ch_imp_idx(ch_imp_reshape)
mean_score, std_score, Score = min_ch_necessary.Kfold(df_class, clf, y, n_ch_tot, ch_imp_reshape, features, t_W, f_W, ch_imp_idx_sorted, K)

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

# Save variables
significant_changes_file_ID = folder+"saved_files/"+"significant_changes"+file_ID_wp
save_files.save_wp(significant_changes, significant_changes_file_ID)

########################################################

# MINIMUM CHANNEL ANALYSIS
# Load variables
with open(folder+"saved_files/"+"significant_changes"+file_ID_wp+'.txt', "rb") as fp:   # Unpickling
    significant_changes = pickle.load(fp)   

p_value = significant_changes[0]
significance = significant_changes[1]
alpha = significant_changes[2]
index_changes = significant_changes[3]
min_ch = significant_changes[4]
stop_plateau = significant_changes[5]
max_mean_accuracy = significant_changes[6]
max_mean_accuracy_sd = significant_changes[7]
min_ch_list = significant_changes[8]
min_ch_fraction = significant_changes[9]

########################################################

# MINIMUM CHANNEL ANALYSIS
# Plots

# Show trend of mean accuracy across sets of channels
min_ch_necessary.plot(n_ch_tot, mean_score, std_score, K, file_ID_wp, folder)

# Confusion matrix fromm extended set of channels
min_ch_necessary.conf_m(df_class, clf, y, n_ch_tot, ch_imp_reshape, features, K, t_W, f_W, ch_imp_idx_sorted, min_ch_comp, task_names, file_ID_wp, folder)

########################################################

# DISPLACEMENT ANALYSIS

# Load displacement variables, if already saved from serial analysis
with open(folder+"saved_files/"+"displacement"+file_ID_wp+'.txt', "rb") as fp:   # Unpickling
    displacement = pickle.load(fp)

Score = displacement[0]
Score_displacement = displacement[1]
Score_displacement2 = displacement[2]
mean_score = displacement[3]
std_score = displacement[4]
mean_score_displacement = displacement[5]
std_score_displacement = displacement[6]
mean_score_displacement2 = displacement[7]
std_score_displacement2 = displacement[8]
ch_imp_idx_sorted = displacement[9]
min_ch_list = displacement[10]
ch_imp_idx_displacement = displacement[11]
ch_imp_idx_displacement2 = displacement[12]

#####

# WARNING: THE FOLLLOWING LINES WOULD PARTIALLY REPLACE THE VARIABLES LOADED IN THE PREVIOUS LINES

# Compute displaced channels and accuracy scores: WARNING
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

# COMPARISON ANALYSIS

# Although the file contains information to compare all the analyses ran serially, without any need to load them one by one
# the code to compare the values by plots has not been developed.

date = '' #insert date of the comparison file you wish to load; comparison files and their dates are
# in the folder

with open(folder+"saved_files/"+'comparison'+date+'.txt', "wb") as fp:   #Pickling
    comparison = pickle.load(fp)
    
#max_mean_accuracy
#max_mean_accuracy_sd
#min_ch
#min_ch_list
#min_ch_fraction
#Score
#Score_displacement
#sbj
#t_seg
#t_W
#f_W
#c
#n_ch_tot