# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 14:01:21 2018

@author: HIWI
"""

import numpy as np

def imp(rfecv, t_W, f_W, n_ch_tot, chN_to_drop):
    
    pixel_imp = abs(rfecv.ranking_ - rfecv.ranking_.max()) # Normalised ranking: ranking 1 is importance 1, last ranking is importance 0
    
    pixel_imp_reshape = np.zeros((n_ch_tot, f_W*t_W)) # Reshape list as channels (rows) and pixels (columns)
    
    for chN in range(n_ch_tot):
        pixel_imp_reshape[chN,:] = pixel_imp[f_W*t_W*chN:f_W*t_W*(chN+1)]/(rfecv.ranking_.max()-1)
        
    ch_imp = pixel_imp_reshape.sum(axis=1)/(f_W*t_W) # Channel importance: average pixel importance over pixels belonging to each channel
    
    ch_imp_reshape = ch_imp.tolist()
    
    for chN in range(len(chN_to_drop)): # Adding 0s for missing channels
        ch_imp_reshape.insert(chN_to_drop[chN]-1, 0)
        
    return pixel_imp, pixel_imp_reshape, ch_imp, ch_imp_reshape