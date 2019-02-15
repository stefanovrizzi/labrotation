# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 18:32:20 2019

@author: HIWI
"""

import numpy as np
import data_processing
import classification
import save_files
import importance_measures
import importance_plots

#S = [1]
#T = ['t_seg']
#W = [[5,10]]
#C = [0.025]

S = [1, 2]
T = ['t_seg', 't_seg_dZ']
W = [[1,5],[5,10]]
C = [0.01, 0.025, 0.1, 1, 10]

save_fig = 0

##########################

comparison = []

for s in range(len(S)): # for each subject
    
    sbj = S[s]
    print ('Subject: '+str(sbj))
    
    for t in range(len(T)): # for each time segmentation criteria
        
        t_seg = T[t]
        
        # File name
        file_ID = "_S"+str(sbj)+'_'+t_seg
        
        df, ch_col, n_ch, fs, chN_to_drop, n_ch_tot, col_filt, nperseg, task_names, folder = data_processing.process(t_seg, sbj, save_fig, file_ID)
        
        # Save dataframe
        df_file_ID = folder+'df'+file_ID
        #save_files.save_wp(df, df_file_ID)
        
        for w in range(len(W)):
            
            f_W = W[w][1]
            t_W = W[w][0]
            
            for k in range(len(C)): # for each penalty term
                
                c = C[k]
                
                # File name with parameters
                file_ID_wp = "_S"+str(sbj)+"_C"+str(c)+"_F"+str(f_W)+"_T"+str(t_W)+"_"+t_seg
                
                clf, score, df, features, X, y, K = classification.setup(df, t_W, f_W, c, n_ch_tot, col_filt, fs, nperseg, task_names, file_ID_wp, folder)
                
                print('Recursive Feature Selection')
                rfecv = classification.recursive_feature_selection(clf, K, X, y)
                
                ##########################
                
                print('Importance')
                # Importance of features and channels
                    
                pixel_imp, pixel_imp_reshape, ch_imp, ch_imp_reshape = importance_measures.imp(rfecv, t_W, f_W, n_ch_tot, chN_to_drop)
                    
                ##########################
                    
                # Plot importance of features and channels
                    
                importance_plots.n_features_vs_CV_scores(rfecv, file_ID_wp, folder)
                
                if max(rfecv.ranking_) > 1:
                    
                    importance_plots.pixel_importance(pixel_imp_reshape, file_ID_wp, folder, ch_col)
                    importance_plots.channel_importance_sorted(ch_imp, file_ID_wp, folder, n_ch_tot)
                    importance_plots.array_importance(ch_imp_reshape, n_ch, file_ID_wp, folder)
                    
                    ############################
                    
                    print('Minimum channels necessary')
                    
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
                    min_ch_necessary.plot(n_ch_tot, mean_score, std_score, file_ID_wp, folder)
                    min_ch_necessary.plot_displacement(n_ch_tot, mean_score, std_score, mean_score_displacement, std_score_displacement, min_ch, file_ID_wp, folder)
                    min_ch_necessary.conf_m(df, clf, y, n_ch_tot, ch_imp_reshape, features, t_W, f_W, ch_imp_idx_sorted, min_ch, task_names, file_ID_wp, folder)
                    
                    # Update comparison of analyses                
                    comparison.append([max_mean_accuracy, max_mean_accuracy_sd, min_ch, min_ch_list, min_ch_fraction, Score, Score_displacement,
                                      sbj, t_seg, t_W, f_W, c, n_ch_tot])
                    
                    displacement = [Score, Score_displacement, mean_score, std_score, mean_score_displacement, std_score_displacement, min_ch_list, ch_imp_idx_displacement]
                    displacement_file_ID = folder+"displacement"+file_ID_wp
                    save_files.save_wp(displacement, displacement_file_ID)
                    
                else:
                    ch_imp_idx_sorted = 0

                ############################
                
                print('Save data')
                # Save data
                data = [sbj, ch_col, n_ch, fs, chN_to_drop, n_ch_tot, col_filt, nperseg, task_names, folder, clf, score, features, X, y, K, ch_imp_idx_sorted]
                data_file_ID = folder+"data"+file_ID_wp
                save_files.save_wp(data, data_file_ID)
                
                print('Save rfecv')
                # Save data recursive feature selection
                rfecv_file_ID = folder+"rfecv"+file_ID_wp
                save_files.save_wp(rfecv, rfecv_file_ID)
    
                ############################

                print ('Completed: '+file_ID_wp)
                print ()
                
            df = df.drop(columns=features)

save_files.save_general(comparison)