# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 18:32:20 2019

@author: STEFANO VRIZZI
"""

import numpy as np
import data_processing
import classification
import save_files
import importance_measures
import importance_plots
import min_ch_necessary
import perm_channels

#S = [1,2] #subject numbers
#T = ['t_seg', 't_seg_dZ'] #time segmentation files
#W = [[1,5],[5,10]] #time-frequency batches
#C = [0.025, 0.1, 1, 10] #penalty parameters

S = [1] #subject numbers
T = ['t_seg'] #time segmentation files
W = [[1,5]] #time-frequency batches
C = [0.025] #penalty parameters

save_fig = 0 #0 = do NOT save figures from channel removal analysis only; 1 = SAVE, this is computationally demanding

##########################

comparison = []

for sbj in S: # for each subject
    
    print ('Subject: '+str(sbj))
    
    for t in T: # for each time segmentation criteria
        
        t_seg = t
        
        # File name
        file_ID = "_S"+str(sbj)+'_'+t_seg
        
        df, ch_col, n_ch, fs, chN_to_drop, n_ch_tot, col_filt, nperseg, task_names, folder = data_processing.process(t_seg, sbj, save_fig, file_ID)
        
        # Save dataframe
        df_file_ID = folder+"saved_files/"+'df'+file_ID
        save_files.save_wp(df, df_file_ID)
        
        for w in W: # for each time-frequency batch
            
            f_W = w[1]
            t_W = w[0]
            
            for c in C: # for each penalty term
                
                # File name with parameters
                file_ID_wp = "_S"+str(sbj)+"_C"+str(c)+"_F"+str(f_W)+"_T"+str(t_W)+"_"+t_seg
                
                print ('Starting: '+file_ID_wp)
                
                # Setup features and classification objects
                
                clf, score, df_class, features, X, y, K = classification.setup(df, t_W, f_W, c, n_ch_tot, col_filt, fs, nperseg, task_names, file_ID_wp, folder)
                    
                # Save dataframe with extracted features
                df_file_ID = folder+'saved_files/'+'df_class'+file_ID+"_F"+str(f_W)+"_T"+str(t_W)
                save_files.save_wp(df_class, df_file_ID)
                
                print('Recursive Feature Selection')
                rfecv = classification.recursive_feature_selection(clf, K, X, y)
                
                ##########################
                
                print('Importance')
                # Importance of features and channels
                    
                pixel_imp, pixel_imp_reshape, ch_imp, ch_imp_reshape = importance_measures.imp(rfecv, t_W, f_W, n_ch_tot, chN_to_drop)
                    
                ##########################
                    
                # Plot importance of features and channels
                    
                importance_plots.n_features_vs_CV_scores(rfecv, file_ID_wp, folder)
                
                if max(rfecv.ranking_) > 1: # If all features are ranked as first, plots would be blank
                    
                    importance_plots.pixel_importance(pixel_imp_reshape, file_ID_wp, folder, ch_col)
                    importance_plots.channel_importance_sorted(ch_imp, file_ID_wp, folder, n_ch_tot)
                    importance_plots.array_importance(ch_imp_reshape, n_ch, file_ID_wp, folder)
                    
                ############################
                    
                print('Minimum channels necessary')
                    
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
                
                # Show trend of mean accuracy across sets of channels
                min_ch_necessary.plot(n_ch_tot, mean_score, std_score, K, file_ID_wp, folder)
                
                # Confusion matrix fromm extended set of channels
                min_ch_necessary.conf_m(df_class, clf, y, n_ch_tot, ch_imp_reshape, features, K, t_W, f_W, ch_imp_idx_sorted, min_ch_comp, task_names, file_ID_wp, folder)
                
                ############################
                
                print('Displacement analysis')

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
                
                # Update comparison of analyses                
                comparison.append([max_mean_accuracy, max_mean_accuracy_sd, min_ch, min_ch_list, min_ch_fraction, Score, Score_displacement,
                                      sbj, t_seg, t_W, f_W, c, n_ch_tot])

                ############################
                
                print('Save data')
                # Save data
                data = [sbj, ch_col, n_ch, fs, chN_to_drop, n_ch_tot, col_filt, nperseg, task_names, folder, clf, Score, features, X, y, K, ch_imp_idx_sorted]
                data_file_ID = folder+"saved_files/"+"data"+file_ID_wp
                save_files.save_wp(data, data_file_ID)
                
                print('Save rfecv')
                # Save data recursive feature selection
                rfecv_file_ID = folder+"saved_files/"+"rfecv"+file_ID_wp
                save_files.save_wp(rfecv, rfecv_file_ID)
                
                print('Save significant changes in mean accuracy trend')
                # Save variables from mean accuracy trend across channel sets
                significant_changes_file_ID = folder+"saved_files/"+"significant_changes"+file_ID_wp
                save_files.save_wp(significant_changes, significant_changes_file_ID)
                
                print('Save displacement')
                # Save variables from displacement analysis
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

                print ('Completed: '+file_ID_wp)
                print ()

save_files.save_general(comparison, folder)