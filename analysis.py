# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 12:00:36 2018

@author: HIWI
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
import pickle

##########################

# Initial parameters

fs = 2048 #Hz, sampling frequency
fsv = 120 #Hz, frame per second in VICON
n_ch = 8 #channles in one row (or column) of one array
n_ch_tot = (n_ch**2)*3 #total channels

##########################

folder = '20181024_KONSENS_recordings/S2/' # choose subject

##########################

# Choose tasks

task_names = [#'iso',
              'bottle',
              'screw',
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

with open(folder+'t_seg.txt') as f:
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

savefigures.histogram_ch(df, n_ch_tot, ch_col, folder)

##########################

# PRE-PROCESSING OF EMG TRACES: REMOVE BAD CHANNELS, RECTIFY, SMOOTH

import pre_processing

df, ch_col, n_ch_tot, chN_to_drop = pre_processing.drop_channels(df, folder)         # DROP CHANNELS WITH ARTIFACTS

df, col_rct = pre_processing.rectify(df, n_ch_tot, ch_col)      # RECTIFY SIGNALS
        
df, col_filt = pre_processing.gauss_filtering(df, n_ch_tot, ch_col, fs, col_rct)       # SMOOTHING TRACES CONVOLVING GAUSSIAN WINDOW

##########################

# RECTIFIED AND SMOOTHED TRACES FOR EACH CHANNEL
# Useful to investigate reproducibility of traces.

savefigures.traces(df, fs, task_names, ch_col, folder)

savefigures.spectrograms(df, fs, task_names, col_filt, folder)

##########################

# CLASSIFICATION

import classification

t_W = 5 # number of time pixels
f_W = 10 # number of frequency pixels

# FILL DATAFRAME WITH PIXELS FROM TIME-FREQUENCY DECOMPOSITION FROM EACH CHANNEL
df, features = classification.feature_extraction(df, t_W, f_W, n_ch_tot, col_filt, fs)     

# Choose classifier
clf = SVC(kernel="linear", C=0.025)

# Set features and labels

X = np.array(df[features])
y = df['task']

# Evaluate K fold cross validation

K = 5 # K-fold
rs = 42 # random seed for splitting training and test sets

score = classification.K_fold_CV(X, y, df, clf, K, rs)

##########################

# Recursive feature selection

rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(5), scoring='accuracy')
rfecv.fit(X, y)
print("Optimal number of features : %d" % rfecv.n_features_)


##########################

# Importance of features and channels

import importance_measures

pixel_imp, pixel_imp_reshape, ch_imp, ch_imp_reshape = importance_measures.imp(rfecv, t_W, f_W, n_ch_tot, chN_to_drop)

##########################

# Plot importance of features and channels

import importance_plots

importance_plots.n_features_vs_CV_scores(rfecv, folder)
importance_plots.pixel_importance(pixel_imp_reshape, folder, ch_col)
importance_plots.channel_importance_sorted(ch_imp, folder, n_ch_tot)
importance_plots.array_importance(ch_imp_reshape, n_ch, folder)

############################

import min_ch_necessary

mean_score, std_score = min_ch_necessary.Kfold(df, clf, y, n_ch_tot, ch_imp_reshape, features, t_W, f_W)

min_ch_necessary.plot(n_ch_tot, mean_score, std_score, folder)

############################

# Save recursive feature selection object

l = rfecv
with open("rfecv2.txt", "wb") as fp:   #Pickling
    pickle.dump(l, fp)

with open("rfecv2.txt", "rb") as fp:   # Unpickling
    RF = pickle.load(fp)

############################

# Save dataframe

#df.to_csv(folder+'df')

#df_new = pd.read_csv(folder+'df', index_col=0)
