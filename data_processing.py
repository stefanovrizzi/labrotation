# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 15:38:06 2019

@author: STEFANO VRIZZI
"""

# File called in serial analysis (start) that replaces data processing code in 'analysis file'

def process(t_seg, sbj, save_fig, file_ID):
    
    folder = '20181024_KONSENS_recordings/S'+str(sbj)+'/'
    
    ##########################
    
    # Initial parameters
    
    fs = 2048 #Hz, sampling frequency
    fsv = 120 #Hz, frame per second in VICON
    n_ch = 8 #channles in one row (or column) of one array
    n_ch_tot = (n_ch**2)*3 #total channels

    ##########################

    # Choose tasks
    
    task_names = [#'iso',
                  'bottle',
                  'screwdriver',
                  'knife',
                  'hammer',
                  'peg',
                  #'free_eating',
                  'jar', 
                  'typing']
                  #'swing']
    task_names.sort()
    
    ##########################
    
    # TIME SEGMENTATION
    # Load segmentation times for EMG traces from kinematics to keep signal of interest
    
    with open(folder+t_seg+'.txt') as f:
        seg = [[int(x) for x in line.split()] for line in f if not line.startswith("#")]
        
    ##########################
        
    # DATAFRAME (df)
    # Create df and its columns, fill it with EMG, trial number, task label and time duration
            
    import df_setup
        
    df, ch_col = df_setup.fill_df(n_ch_tot, n_ch, seg, task_names, folder, fs, fsv)                
    
    ##########################
    
    # HISTOGRAMS OF EACH CHANNEL
    # Useful to investigate faulty channels by voltage range (0-6000 mV as acceptable).
    
    if save_fig == 1:
    
        import savefigures
        savefigures.histogram_ch(df, n_ch_tot, ch_col, file_ID, folder)
    
    with open(folder+'ch_drop.txt') as f:
        chN_to_drop = [int(x) for x in f]
    
    if save_fig == 1:
        savefigures.traces(df, task_names, ch_col, chN_to_drop, file_ID, folder)
    
    ##########################
    
    # CHECK CHANNELS AND UPDATE CH_DROP_UPDATED.TXT file, then LOAD FILE
    # Inspect histrograms and check plots 'ch*' and 'drop_ch*' to assess channel viability.
    # Manually update the ch_drop_update.txt file
        
    ##########################
    
    # PRE-PROCESSING OF EMG TRACES: REMOVE BAD CHANNELS, RECTIFY, SMOOTH
    
    import signal_processing
    
    df, ch_col, n_ch_tot, chN_to_drop = signal_processing.drop_channels(df, folder)         # DROP CHANNELS WITH ARTIFACTS
    
    df, col_rct = signal_processing.rectify(df, n_ch_tot, ch_col)      # RECTIFY SIGNALS
    
    window_length = int(fs*1)
            
    df, col_filt = signal_processing.gauss_filtering(df, n_ch_tot, ch_col, fs, col_rct, window_length)       # SMOOTHING TRACES CONVOLVING GAUSSIAN WINDOW
    
    ##########################
    
    # RECTIFIED AND SMOOTHED TRACES FOR EACH CHANNEL
    # Useful to investigate reproducibility of traces.
    
    if save_fig == 1:
        savefigures.filt_traces(df, task_names, ch_col, file_ID, folder)
    
    nperseg = int(fs*1)
    
    if save_fig == 1:
        savefigures.spectrograms(df, fs, nperseg, task_names, col_filt, file_ID, folder)
    
    return df, ch_col, n_ch, fs, chN_to_drop, n_ch_tot, col_filt, nperseg, task_names, folder