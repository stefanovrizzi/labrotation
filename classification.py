# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 15:43:47 2018

@author: HIWI
"""

import numpy as np
from scipy import signal
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

# Extract features from time-frequency decomposition

def feature_extraction(df, t_W, f_W, n_ch_tot, col_filt, fs, nperseg):
 
    features_tot = f_W*t_W*n_ch_tot
    
    df['features'] = ''
    
    for trialN in df.index: #each trial, columns
        
        F = []
        
        for chN, ch in enumerate(col_filt): #each channel, rows
    
            x = np.array(df[ch].loc[trialN])
    
            f, t, Sxx = signal.spectrogram(x, fs, window=('tukey', 1), nperseg=nperseg, noverlap=fs*.01)
                
            W = np.reshape(Sxx[0:f_W, 0:t_W], (t_W*f_W)) #time-frequency window of interest
            F.append(W)
    
        df.at[trialN, 'features'] = np.reshape(F, features_tot)
        
    # Create one column for each t-f pixel
    
    for p in range(features_tot):
        df['p'+str(p)] = ''
    features = df.columns[ df.columns.get_loc('p0') : df.columns.get_loc('p0') + features_tot]
    
    df[features] = df['features'].tolist()
    
    return df, features

# Plot confusion matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap='jet')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

##########################

def K_fold_CV(X, y, df, clf, K, rs, class_names, file_ID_wp, folder):
    
    # Create the RFE object and compute a cross-validated score.
    
    kf = cross_validation.KFold(len(df.index), n_folds=K, shuffle=True, random_state=rs)
    
    score = []
    y_pred = []
    Y_test = []
    
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        clf.fit(X_train, y_train)
        y_pred.extend(clf.fit(X_train, y_train).predict(X_test))
        Y_test.extend(y_test)
        score.append(clf.score(X_test, y_test))
    
    print ('Mean score: ', np.mean(score))
    print ('SD score: ', np.std(score))
    
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(Y_test, y_pred)
    np.set_printoptions(precision=2)
    
    # Plot non-normalized confusion matrix
    
    plt.ioff()
    
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')
    plt.close()
    
    # Plot normalized confusion matrix
    
    plt.ioff()
    
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    
    plt.savefig(folder+'img/cm_all_ch'+file_ID_wp+'.png', bbox_inches='tight')
    
    plt.close()
    
    return score

##########################

def setup(df, t_W, f_W, c, n_ch_tot, col_filt, fs, nperseg, task_names, file_ID_wp, folder):
    
    clf = SVC(kernel="linear", C=c)
    
    # FILL DATAFRAME WITH PIXELS FROM TIME-FREQUENCY DECOMPOSITION FROM EACH CHANNEL
    df, features = feature_extraction(df, t_W, f_W, n_ch_tot, col_filt, fs, nperseg)     
    
    # Set features and labels
    
    X = np.array(df[features])
    y = df['task']
    
    # Evaluate K fold cross validation
    
    K = 5 # K-fold
    rs = 42 # random seed for splitting training and test sets
    
    score = K_fold_CV(X, y, df, clf, K, rs, task_names, file_ID_wp, folder)
    
    return clf, score, df, features, X, y, K

##########################

# Recursive feature selection

def recursive_feature_selection(clf, K, X, y):

    rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(K, random_state=42, shuffle=True), scoring='accuracy')
    rfecv.fit(X, y)
    print("Optimal number of features : %d" % rfecv.n_features_)
    
    return rfecv