# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 17:04:31 2019

@author: HIWI
"""

import pickle

def save_wp(file, name_file_ID):    
    with open(name_file_ID+'.txt', "wb") as fp:   # Unpickling
        pickle.dump(file, fp)
        
def save_general(file):
    with open('comparison.txt', "wb") as fp:   #Pickling
        pickle.dump(file, fp)