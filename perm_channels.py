# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 16:32:05 2019

@author: STEFANO VRIZZI
"""

# Permutation tests to test increase or decrese in trend
# Permutation tests to test differences between displacement and control (no displacement) group

import numpy as np
from mlxtend.evaluate import permutation_test

def displacement_diff(Score1, Score2, n_ch_comp):
    
    p_value = []
    alpha = 0.01 #/n_ch_comp #set alpha value to define significant change; this can be scaled on the number of comparisons
    
    for i in range(n_ch_comp):
        
        treatment = Score1[i]
        control = Score2[i]
        
        p_value.append(permutation_test(treatment, control, method='approximate', num_rounds=int(alpha**(-1)*100)))
        
    significance = [i < alpha for i in p_value]
        
    return p_value, significance, alpha


def trend_diff(Score, n_ch_comp):
    
    p_value = []
    alpha = 0.01 #/n_ch_comp #set alpha value to define significant change; this can be scaled on the number of comparisons
        
    for i in range(n_ch_comp):
        
        treatment = Score[i+1]
        control = Score[i]
        
        p_value.append(permutation_test(treatment, control, method='approximate', num_rounds=int(alpha**(-1)*100)))
            
    significance = [i < alpha for i in p_value]
        
    print ('Number of changes: ', sum(significance))
    idx = np.nonzero(significance)[0]
    idx = idx.tolist() # indeces of comparison number where change was detected
        
    print ('Changes after n number of channels: ', np.array(idx)+1)
        
    return p_value, significance, alpha, idx
