# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 20:44:15 2018

@author: HIWI
"""

import numpy as np
from sklearn import cross_validation
import matplotlib.pyplot as plt

def Kfold(df, clf, y, n_ch_tot, ch_imp_reshape, features, t_W, f_W):

    ch_imp_idx_sorted = list(reversed(np.argsort(ch_imp_reshape)+1))
    
    P = []
    mean_score = []
    std_score = []
    
    for i in range(n_ch_tot):
        init = (df.columns.get_loc('ch'+str(ch_imp_idx_sorted[i])) - (df.columns.get_loc('time_duration')+1) )*(t_W*f_W)
        end = (df.columns.get_loc('ch'+str(ch_imp_idx_sorted[i])) - (df.columns.get_loc('time_duration')+1) +1 )*(t_W*f_W)
    
        P.extend(features[init:end])
        
        X = df[P]
        X = np.array(X)
        
        score = []
        
        kf = cross_validation.KFold(len(df.index), n_folds=5, shuffle=True, random_state=42)
    
        for train_index, test_index in kf:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
        
            clf.fit(X_train, y_train)
            score.append(clf.score(X_test, y_test))
    
        mean_score.append(np.mean(score))
        std_score.append(np.std(score))
        
    return mean_score, std_score

def plot(n_ch_tot, mean_score, std_score, folder):

    plt.figure(figsize=(12,8))
    plt.errorbar(range(1,n_ch_tot+1), mean_score, std_score, linestyle='None', marker='.', color = 'k')
    plt.xlabel('Number of channels (decreasing importance)')
    plt.ylabel('Accuracy (%)')
    
    plt.savefig(folder+'img/min_ch_necessary.png', bbox_inches='tight')
    
    plt.show()
    
#np.where(mean_score == max(np.array(mean_score)))