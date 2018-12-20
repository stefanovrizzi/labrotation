# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 15:43:47 2018

@author: HIWI
"""

import numpy as np
from scipy import signal
from sklearn import cross_validation

# Extract features from time-frequency decomposition

def feature_extraction(df, t_W, f_W, n_ch_tot, col_filt, fs):
 
    features_tot = f_W*t_W*n_ch_tot
    
    df['features'] = ''
    
    for trialN in df.index: #each trial, columns
        
        F = []
        
        for chN, ch in enumerate(col_filt): #each channel, rows
    
            x = np.array(df[ch].loc[trialN])
    
            f, t, Sxx = signal.spectrogram(x, fs, window=('tukey', 1), nperseg=fs, noverlap=fs*.01)
                
            W = np.reshape(Sxx[0:f_W, 0:t_W], (t_W*f_W)) #time-frequency window of interest
            F.append(W)
    
        df.at[trialN, 'features'] = np.reshape(F, features_tot)
        
    # Create one column for each t-f pixel
    
    for p in range(features_tot):
        df['p'+str(p)] = ''
    features = df.columns[ df.columns.get_loc('p0') : df.columns.get_loc('p0') + features_tot]
    
    df[features] = df['features'].tolist()
    
    return df, features

##########################

def K_fold_CV(X, y, df, clf, K, rs):
    
    # Create the RFE object and compute a cross-validated score.
    
    kf = cross_validation.KFold(len(df.index), n_folds=K, shuffle=True, random_state=rs)
    
    score = []
    
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        clf.fit(X_train, y_train)
        score.append(clf.score(X_test, y_test))
    
    print ('Mean score: ', np.mean(score))
    print ('SD score: ', np.std(score))
    
    return score