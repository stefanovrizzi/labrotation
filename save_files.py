# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 17:04:31 2019

@author: STEFANO VRIZZI
"""

# Code to save files

import pickle
from time import gmtime, strftime

# Save With Parameters included in file name

def save_wp(file, name_file_ID):    
    with open(name_file_ID+'.txt', "wb") as fp:   # Unpickling
        pickle.dump(file, fp)

# Save comparison from serial analysis ('start.py') with time and day when the comparison was generated

def save_general(file, folder):
    with open(folder+"saved_files/"+'comparison'+strftime("%d-%m-%Y %H:%M", gmtime())+'.txt', "wb") as fp:   #Pickling
        pickle.dump(file, fp)