# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 20:44:15 2018

@author: HIWI
"""

import numpy as np
from sklearn import cross_validation
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

def ch_imp_idx(ch_imp_reshape):
    
    ch_imp_idx = list(reversed(np.argsort(ch_imp_reshape)+1))
    
    return ch_imp_idx

def Kfold(df, clf, y, n_ch_tot, ch_imp_reshape, features, t_W, f_W, ch_imp_idx, K):
    
    P = []
    mean_score = []
    std_score = []
    Score = []
    
    for i in range(n_ch_tot):
        init = (df.columns.get_loc('ch'+str(ch_imp_idx[i])) - (df.columns.get_loc('time_duration')+1) )*(t_W*f_W)
        end = (df.columns.get_loc('ch'+str(ch_imp_idx[i])) - (df.columns.get_loc('time_duration')+1) +1 )*(t_W*f_W)

        P.extend(features[init:end])
        
        X = df[P]
        X = np.array(X)
        
        score = []
        
        kf = cross_validation.KFold(len(df.index), n_folds=K, shuffle=True, random_state=42)
    
        for train_index, test_index in kf:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
        
            clf.fit(X_train, y_train)
            score.append(clf.score(X_test, y_test))
            
        Score.append(score)
        mean_score.append(np.mean(score))
        std_score.append(np.std(score))
        
    return mean_score, std_score, Score

def plot(n_ch_tot, mean_score, std_score, file_ID_wp, folder):

    plt.ioff()
    
    plt.figure(figsize=(12,6))
    ax = plt.subplot(111)
    ax.errorbar(range(1, n_ch_tot+1), mean_score, std_score, linestyle='None', marker='.', color = 'k')
    ax.set_xlabel('Number of channels (adding decreasing importance)', fontsize=18)
    ax.set_ylabel('Accuracy (normalised)', fontsize=18)
    
    #plt.rcParams.update({'font.size': 19})
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    y_low = min(np.array(mean_score)-np.array(std_score))
    ax.set_ylim([y_low*0.8, 1])
    ax.set_xlim([0, n_ch_tot])
    
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)

    plt.savefig(folder+'img/min_ch_necessary'+file_ID_wp+'.png', bbox_inches='tight')
    
    plt.close()
    
    
def plot_displacement(n_ch_tot, mean_score, std_score, mean_score_displacement, std_score_displacement, min_ch, file_ID_wp, folder):

    plt.ioff()
    
    plt.figure(figsize=(12,6))
    ax = plt.subplot(111)
    ax.errorbar(range(1, n_ch_tot+1), mean_score, std_score, linestyle='None', marker='.', color = 'k', label='ND')
    ax.errorbar(range(1, min_ch+1), mean_score_displacement, std_score_displacement,
                linestyle='None', marker='.', color = 'r', alpha=.5, label='D')
    ax.set_xlabel('Number of channels (adding decreasing importance)', fontsize=18)
    ax.set_ylabel('Accuracy (normalised)', fontsize=18)
    ax.legend(fontsize=18)
    
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    
    #plt.rcParams.update({'font.size': 19})
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    y_low = min(min(np.array(mean_score_displacement)-np.array(std_score_displacement)),
                min(np.array(mean_score)-np.array(std_score)))
    ax.set_ylim([y_low*0.8, 1])
    ax.set_xlim([0, min_ch+.5])

    plt.savefig(folder+'img/min_ch_necessary_displacement'+file_ID_wp+'.png', bbox_inches='tight')
    
    plt.close()
    #plt.show()
    
#np.where(mean_score == max(np.array(mean_score)))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    clb = plt.colorbar()
    clb.set_label('Accuracy (normalised)')
    
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

def conf_m(df, clf, y, n_ch_tot, ch_imp_reshape, features, t_W, f_W, ch_imp_idx, min_ch, class_names, file_ID_wp, folder):
    
    P = []
    mean_score = []
    std_score = []
    
    for i in range(min_ch):
        init = (df.columns.get_loc('ch'+str(ch_imp_idx[i])) - (df.columns.get_loc('time_duration')+1) )*(t_W*f_W)
        end = (df.columns.get_loc('ch'+str(ch_imp_idx[i])) - (df.columns.get_loc('time_duration')+1) +1 )*(t_W*f_W)
    
        P.extend(features[init:end])
        
    X = df[P]
    X = np.array(X)
        
    score = []
    y_pred = []
    Y_test = []
        
    kf = cross_validation.KFold(len(df.index), n_folds=5, shuffle=True, random_state=42)
    
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        clf.fit(X_train, y_train)
        y_pred.extend(clf.fit(X_train, y_train).predict(X_test))
        Y_test.extend(y_test)
        score.append(clf.score(X_test, y_test))
    
    mean_score.append(np.mean(score))
    std_score.append(np.std(score))
    
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(Y_test, y_pred)
    np.set_printoptions(precision=2)
    
    # Plot non-normalized confusion matrix
    
    plt.ioff()
    
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='')
    
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='')
    
    #plt.rcParams.update({'font.size': 12})
    
    plt.savefig(folder+'img/cm_min_ch'+file_ID_wp+'.png', bbox_inches='tight')
    
    plt.close()
    #plt.show()
    
def displacement(ch_imp_reshape, min_ch, min_ch_list, n_ch):
    
    ch_imp_idx_displacement = []
    
    array_ch_imp = np.zeros((3, n_ch, n_ch))
    array_ch_idx = np.zeros((3, n_ch, n_ch))
    
    for i in range(3):
        t = ch_imp_reshape[(n_ch**2)*i:(n_ch**2)*(i+1)] # temporary
        array_ch_imp[i,:,:] = np.array(t).reshape((n_ch,n_ch)) # temporary reshape for array layout
        array_ch_idx[i,:,:] = np.linspace(((n_ch**2)*i)+1,((n_ch**2)*(i+1)),(n_ch**2)).astype(int).reshape((n_ch,n_ch))
        
    for ch in range(min_ch):
        a = np.where([array_ch_idx==min_ch_list[ch]])
        #a[1][0] # array number
        X = a[2][0] # row or x coordinate
        Y = a[3][0] # column or y coordinate
        
        displacement_candidates = []
        
        if X==0:
            
            if Y==0:
                displacement_candidates = [array_ch_idx[a[1][0],X,Y+1],
                                           array_ch_idx[a[1][0],X+1,Y],
                                           array_ch_idx[a[1][0],X+1,Y+1]]
            
            elif Y==n_ch-1:
                displacement_candidates = [array_ch_idx[a[1][0],X,Y-1],
                                           array_ch_idx[a[1][0],X+1,Y],
                                           array_ch_idx[a[1][0],X+1,Y-1]]
                
            else:
                displacement_candidates = [array_ch_idx[a[1][0],X,Y-1],
                                           array_ch_idx[a[1][0],X+1,Y-1],
                                           array_ch_idx[a[1][0],X+1,Y+1],
                                           array_ch_idx[a[1][0],X+1,Y],
                                           array_ch_idx[a[1][0],X,Y+1]]
                
        elif X==n_ch-1:
            
            if Y==0:
                displacement_candidates = [array_ch_idx[a[1][0],X,Y+1],
                                           array_ch_idx[a[1][0],X-1,Y],
                                           array_ch_idx[a[1][0],X-1,Y+1]]
                
            elif Y==n_ch-1:
                displacement_candidates = [array_ch_idx[a[1][0],X,Y-1],
                                           array_ch_idx[a[1][0],X-1,Y],
                                           array_ch_idx[a[1][0],X-1,Y-1]]
                
            else:
                displacement_candidates = [array_ch_idx[a[1][0],X,Y-1],
                                           array_ch_idx[a[1][0],X-1,Y-1],
                                           array_ch_idx[a[1][0],X-1,Y+1],
                                           array_ch_idx[a[1][0],X-1,Y],
                                           array_ch_idx[a[1][0],X,Y+1]]
            
        else:
            
            if Y==0:
                displacement_candidates = [array_ch_idx[a[1][0],X,Y+1],
                                           array_ch_idx[a[1][0],X-1,Y],
                                           array_ch_idx[a[1][0],X-1,Y+1],
                                           array_ch_idx[a[1][0],X+1,Y],
                                           array_ch_idx[a[1][0],X+1,Y+1]]
                
            elif Y==n_ch-1:
                displacement_candidates = [array_ch_idx[a[1][0],X,Y-1],
                                           array_ch_idx[a[1][0],X-1,Y],
                                           array_ch_idx[a[1][0],X-1,Y-1],
                                           array_ch_idx[a[1][0],X+1,Y],
                                           array_ch_idx[a[1][0],X+1,Y-1]]
            
            else:
                displacement_candidates = [array_ch_idx[a[1][0],X,Y-1],
                                           array_ch_idx[a[1][0],X-1,Y],
                                           array_ch_idx[a[1][0],X-1,Y-1],
                                           array_ch_idx[a[1][0],X+1,Y-1],
                                           array_ch_idx[a[1][0],X-1,Y+1],
                                           array_ch_idx[a[1][0],X+1,Y+1],
                                           array_ch_idx[a[1][0],X+1,Y],
                                           array_ch_idx[a[1][0],X,Y+1]]
        
        displacement_candidates = [int(item) for item in displacement_candidates if ch_imp_reshape[int(item)-1] > 0]
        
        if not displacement_candidates:
            ch_imp_idx_displacement.extend(min_ch_list[ch])
        else:
            ch_imp_idx_displacement.extend(np.random.choice(displacement_candidates, size=1))
        
    return ch_imp_idx_displacement