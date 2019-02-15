# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 12:43:06 2019

@author: HIWI
"""

import numpy as np

def displacement(ch_imp_reshape, min_ch, min_ch_list):
    
    ch_imp_idx_displacement = []
    
    n_ch = 8
    array_ch_imp = np.zeros((3, n_ch, n_ch))
    array_ch_idx = np.zeros((3, n_ch, n_ch))
    
    for i in range(3):
        t = ch_imp_reshape[(n_ch**2)*i:(n_ch**2)*(i+1)] # temporary
        array_ch_imp[i,:,:] = np.array(t).reshape((n_ch,n_ch)) # temporary reshape for array layout
        array_ch_idx[i,:,:] = np.linspace(((n_ch**2)*i)+1,((n_ch**2)*(i+1)),(n_ch**2)).astype(int).reshape((n_ch,n_ch))
        
    for ch in range(min_ch-1):
        a = np.where([array_ch_idx==min_ch_list[ch]])
        a[1][0] # array number
        Y = a[2][0] # row or y coordinate
        X = a[3][0] # column or x coordinate
        
        displacement_candidates = []
        
        if X==0 and Y==0:
            displacement_candidates = [array_ch_idx[a[1][0],X,Y+1],
                                       array_ch_idx[a[1][0],X+1,Y],
                                       array_ch_idx[a[1][0],X+1,Y+1]]
        elif X==n_ch-1 and Y==0:
            displacement_candidates = [array_ch_idx[a[1][0],X,Y+1],
                                       array_ch_idx[a[1][0],X-1,Y],
                                       array_ch_idx[a[1][0],X-1,Y+1]]
            
        elif X==0 and Y==n_ch-1:
            displacement_candidates = [array_ch_idx[a[1][0],X,Y-1],
                                       array_ch_idx[a[1][0],X+1,Y],
                                       array_ch_idx[a[1][0],X+1,Y-1]]
            
        elif X==n_ch-1 and Y==n_ch-1:
            displacement_candidates = [array_ch_idx[a[1][0],X,Y-1],
                                       array_ch_idx[a[1][0],X-1,Y],
                                       array_ch_idx[a[1][0],X-1,Y-1]]
            
        else:
            displacement_candidates = [array_ch_idx[a[1][0],X,Y-1],
                                       array_ch_idx[a[1][0],X-1,Y],
                                       array_ch_idx[a[1][0],X-1,Y-1],
                                       array_ch_idx[a[1][0],X+1,Y-1],
                                       array_ch_idx[a[1][0],X-1,Y+1],
                                       array_ch_idx[a[1][0],X+1,Y+1],
                                       array_ch_idx[a[1][0],X+1,Y],
                                       array_ch_idx[a[1][0],X,Y+1]]
        
        ch_imp_idx_displacement.extend(np.random.choice(displacement_candidates, size=1))
        
    return ch_imp_idx_displacement
    