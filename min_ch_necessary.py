# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 20:44:15 2018

@author: STEFANO VRIZZI
"""

import numpy as np
from sklearn import cross_validation
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

# index of channels order by decreasing importance

def ch_imp_idx(ch_imp_reshape):
    
    ch_imp_idx = list(reversed(np.argsort(ch_imp_reshape)+1))
    
    return ch_imp_idx


#K-fold cross-validation adding features (pixels) of the next channel by decreasing importance

def Kfold(df, clf, y, n_ch_tot, ch_imp_reshape, features, t_W, f_W, ch_imp_idx, K):
    
    P = []
    mean_score = []
    std_score = []
    Score = []
    
    for i in range(n_ch_tot): #add features (pixels) of the next channel by decreasing importance
        init = (df.columns.get_loc('ch'+str(ch_imp_idx[i])) - (df.columns.get_loc('time_duration')+1) )*(t_W*f_W)
        end = (df.columns.get_loc('ch'+str(ch_imp_idx[i])) - (df.columns.get_loc('time_duration')+1) +1 )*(t_W*f_W)

        P.extend(features[init:end])
        
        X = df[P]
        X = np.array(X)
        
        score = []
        
        for cv in range(20): #run 20 repetitions of CV

            kf = cross_validation.KFold(len(df.index), n_folds=K, shuffle=True)
        
            for train_index, test_index in kf:
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
            
                clf.fit(X_train, y_train)
                score.append(clf.score(X_test, y_test))
                
        Score.append(score)
        mean_score.append(np.mean(score))
        std_score.append(np.std(score))
        
    return mean_score, std_score, Score


#Plot mean accuracy scores for all channel sets, from most important channel, adding the next channel by decreasing importance

def plot(n_ch_tot, mean_score, std_score, K, file_ID_wp, folder):

    plt.ioff()
    
    plt.figure(figsize=(12,6))
    ax = plt.subplot(111)
    ax.errorbar(range(1, n_ch_tot+1), mean_score, std_score/np.sqrt(20*K), linestyle='None', marker='.', color = 'k')
    
    # Labels, axis and ticks improvements
    ax.set_xlabel('Number of channels (adding decreasing importance)', fontsize=18)
    ax.set_ylabel('Accuracy (normalised)', fontsize=18)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    y_low = min(np.array(mean_score)-np.array(std_score/np.sqrt(20*K)))
    ax.set_ylim([y_low*0.9, 1])
    ax.set_xlim([0, n_ch_tot])
    
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)

    plt.savefig(folder+'img/min_ch_necessary'+file_ID_wp+'.png', bbox_inches='tight')
    
    plt.close()
    #plt.show()
    
    
    
#Plot mean accuracy scores for one displacement example and the original channel sets

def plot_displacement(mean_score, std_score, mean_score_displacement, std_score_displacement, min_ch, K, file_ID_wp, folder):

    plt.ioff()
    
    plt.figure(figsize=(12,6))
    ax = plt.subplot(111)
    
    #Plot mean accuracy scores for original sets of channels
    ax.errorbar(range(1, min_ch+1), mean_score[0:min_ch], std_score[0:min_ch]/np.sqrt(20*K),
                linestyle='None', marker='.', color = 'k', alpha=1, label='ND',
                markersize='22', ecolor='k', capsize=8, elinewidth=4)

    #Plot mean accuracy scores for displacement example
    ax.errorbar(np.arange(1, min_ch+1)+0.1, mean_score_displacement[0:min_ch], std_score_displacement[0:min_ch]/np.sqrt(20*K),
                linestyle='None', marker='.', color = 'r', alpha=.9, label='D',
                markersize='20', ecolor='r', capsize=8, elinewidth=4)
    
    # Labels, axis and ticks improvements
    ax.set_xlabel('Number of channels (adding decreasing importance)', fontsize=18)
    ax.set_ylabel('Accuracy (normalised)', fontsize=18)
    ax.legend(fontsize=18)
    
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Compute lower limit of plot on y axis, for plotting purposes   
    y_low = min(min(np.array(mean_score_displacement)-np.array(std_score_displacement/np.sqrt(20*K))),
                min(np.array(mean_score)-np.array(std_score/np.sqrt(20*K))))
    ax.set_ylim([y_low*0.8, 1])
    ax.set_xlim([0, min_ch+.5])

    plt.savefig(folder+'img/min_ch_necessary_displacement'+file_ID_wp+'.png', bbox_inches='tight')
    
    plt.close()
    #plt.show()


#Plot mean accuracy scores for two displacement examples and the original channel sets

def plot_displacement2(mean_score, std_score,
                      mean_score_displacement, std_score_displacement,
                      mean_score_displacement2, std_score_displacement2,
                      min_ch_comp, K, file_ID_wp, folder):

    plt.ioff()
    
    plt.figure(figsize=(12,6))
    ax = plt.subplot(111)
    
    #Plot mean accuracy scores for original sets of channels
    ax.errorbar(range(1, min_ch_comp+1), mean_score[0:min_ch_comp], std_score[0:min_ch_comp]/np.sqrt(20*K),
                linestyle='None', marker='.', color = 'k', alpha=1, label='ND',
                markersize='22', ecolor='k', capsize=8, elinewidth=4)

    #Plot mean accuracy scores for first displacement example
    ax.errorbar(np.arange(1,min_ch_comp+1)+0.20, mean_score_displacement[0:min_ch_comp],
                std_score_displacement[0:min_ch_comp]/np.sqrt(20*K),
                linestyle='None', marker='.', color = 'r', alpha=.9, label='D1',
                markersize='20', ecolor='r', capsize=8, elinewidth=4)
    
    #Plot mean accuracy scores for second displacement example
    ax.errorbar(np.arange(1,min_ch_comp+1)-0.20, mean_score_displacement2[0:min_ch_comp],
                std_score_displacement2[0:min_ch_comp]/np.sqrt(20*K),
                linestyle='None', marker='.', color = 'b', alpha=.9, label='D2',
                markersize='20', ecolor='b', capsize=8, elinewidth=4)
    
    # Labels, axis and ticks improvements
    ax.set_xlabel('Number of channels (adding decreasing importance)', fontsize=18)
    ax.set_ylabel('Accuracy (normalised)', fontsize=18)
    ax.legend(fontsize=18)
    
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Compute lower limit of plot on y axis, for plotting purposes
    y_low = min(min(np.array(mean_score_displacement[0:min_ch_comp])-np.array(std_score_displacement[0:min_ch_comp]/np.sqrt(20*K))),
                min(np.array(mean_score_displacement2[0:min_ch_comp])-np.array(std_score_displacement2[0:min_ch_comp]/np.sqrt(20*K))),
                min(np.array(mean_score[0:min_ch_comp])-np.array(std_score[0:min_ch_comp]/np.sqrt(20*K))))
    ax.set_ylim([y_low*0.9, 1])
    ax.set_xlim([0, min_ch_comp+.5])
    ax.set_xticks(np.arange(0, min_ch_comp+1, 1))

    plt.savefig(folder+'img/min_ch_necessary_displacement'+file_ID_wp+'_2.png', bbox_inches='tight')
    
    plt.close()
    #plt.show()

#Code to plot confusion matrix (edited from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html)

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
    clb.set_label('Frequency (normalised)')
    
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


# Compute and plot confusion matrix for extended set of minimum set of channels

def conf_m(df, clf, y, n_ch_tot, ch_imp_reshape, features, K, t_W, f_W, ch_imp_idx, min_ch, class_names, file_ID_wp, folder):
    
    P = []
    
    for i in range(min_ch):
        init = (df.columns.get_loc('ch'+str(ch_imp_idx[i])) - (df.columns.get_loc('time_duration')+1) )*(t_W*f_W)
        end = (df.columns.get_loc('ch'+str(ch_imp_idx[i])) - (df.columns.get_loc('time_duration')+1) +1 )*(t_W*f_W)
    
        P.extend(features[init:end])
        
    X = df[P]
    X = np.array(X)
        
    y_pred = []
    Y_test = []
    
    for cv in range(20):
    
        kf = cross_validation.KFold(len(df.index), n_folds=K, shuffle=True)
    
        for train_index, test_index in kf:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            clf.fit(X_train, y_train)
            y_pred.extend(clf.fit(X_train, y_train).predict(X_test))
            Y_test.extend(y_test)
    
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(Y_test, y_pred)
    np.set_printoptions(precision=2)
    
    # Plot non-normalized confusion matrix
    
    plt.ioff()
    
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names)
    plt.close()
    
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True)
    
    #plt.rcParams.update({'font.size': 12})
    
    plt.savefig(folder+'img/cm_min_ch'+file_ID_wp+'.png', bbox_inches='tight')
    
    plt.close()
    #plt.show()
    
    
# Displacement of channels.
# WARNING: currently, if the channel is at the border or corner of the array, displacement can only happen in the same array
# Displaced channel cannot be the neighbouring channel in the next array. Please improve the code

def displacement(ch_imp_reshape, min_ch_comp, min_ch_list, n_ch):
    
    ch_imp_idx_displacement = []
    
    array_ch_imp = np.zeros((3, n_ch, n_ch))
    array_ch_idx = np.zeros((3, n_ch, n_ch))
    
    for i in range(3): # useful variables for each array
        t = ch_imp_reshape[(n_ch**2)*i:(n_ch**2)*(i+1)] # temporary - channel importance of one array
        array_ch_imp[i,:,:] = np.array(t).reshape((n_ch,n_ch)) # temporary reshape for array layout
        array_ch_idx[i,:,:] = np.linspace(((n_ch**2)*i)+1,((n_ch**2)*(i+1)),(n_ch**2)).astype(int).reshape((n_ch,n_ch)) #index channels in array
        
    for ch in range(min_ch_comp): # for each channel in the extended set of minimum channels
        a = np.where([array_ch_idx==min_ch_list[ch]])
        #a[1][0] # array number
        X = a[2][0] # row or x coordinate
        Y = a[3][0] # column or y coordinate
        
        displacement_candidates = []
        
        # Case by case, depending on border or corner position of the channel
        
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
        
        # Canidadate list, making sure that candidates are not the discarded channels (channel importance is 0 for them)
        displacement_candidates = [int(item) for item in displacement_candidates if ch_imp_reshape[int(item)-1] > 0]
        
        # 
        if not displacement_candidates: # if candidate list is empty, add original important channel
            ch_imp_idx_displacement.extend(min_ch_list[ch])
        else:
            displaced_ch = np.random.choice(displacement_candidates, size=1) # displaced channel
            
            # If chosen displacement was already chosen before, for other neighbouring channels, another candidate is chosen
            # This should ideally be repeated even after the replacement of the displacement, up to confirmation
            # that the list does not have any duplicate again. However, this may result in a never-ending search,
            # in case that two important neighbouring channels are e.g. in a corner and surrounded by removed channels.
            if displaced_ch in ch_imp_idx_displacement:
                displacement_candidates.remove(displaced_ch)
                displaced_ch = np.random.choice(displacement_candidates, size=1)
            
            # Update list of displaced channels
            ch_imp_idx_displacement.extend(displaced_ch)
                
    return ch_imp_idx_displacement

# Find minimum number of channels from the indeces of change found by permutation tests
def min_ch_function(index_changes, Score, n_ch_tot):
    
    if index_changes:
        
        increase = []
            
        for i in range(len(index_changes)):
            
            increase.append(np.mean(Score[index_changes[i]+1]) > np.mean(Score[index_changes[i]])) # detect significant increase
                
        idx_increase = [i for i,x in enumerate(increase) if x == True]
        
        if not idx_increase == False: # check that significant changes are for increasing values
                
            min_ch = index_changes[idx_increase[-1]]+2 # get largest index for which mean accuracy increases
                    
            if len(increase) > (idx_increase[-1]+1):
                stop_plateau = index_changes[idx_increase[-1]+1]+2
                
            else:
                stop_plateau = -1
        
        #print ('Plateau index: ', stop_plateau)
        
    else: # select all channels as minimum set of channel, in case that no significant increase was detected
        min_ch = 1
        stop_plateau = n_ch_tot
        
    print ('Minimum number of channels: ', min_ch)
    print ('Plateau: ', stop_plateau)
    
    return min_ch, stop_plateau