import soundfile

from scipy import signal

import sys
import numpy as np
import os
import glob

import pandas as pd
import csv

path_osmose_dataset = "/home/datawork-osmose/dataset/"


def process_file(audio_file):

    
    data, sample_rate = soundfile.read( os.path.join(path_audio_files,audio_file)  )
    
    bpcoef=signal.butter(20, np.array([fmin_HighPassFilter, sample_rate/2-1]), fs=sample_rate, output='sos', btype='bandpass')
    data = signal.sosfilt(bpcoef, data)   
    
    
    return np.mean(data), np.std(data)
    

if __name__ == "__main__":

    dataset_ID = sys.argv[1]
    analysis_fs = float(sys.argv[2])
    ind_min = int(sys.argv[3]) 
    ind_max = int(sys.argv[4]) 
         
    path_analysisFolder = os.path.join(path_osmose_dataset, dataset_ID, 'analysis')
    path_analysis_fiche = os.path.join(path_analysisFolder,'analysis_fiche.csv')
    analysis_fiche = pd.read_csv(path_analysis_fiche,header=0)
    
    maxtime_display_spectro = analysis_fiche['maxtime_display_spectro'][0]       

    if 'fmin_HighPassFilter' not in analysis_fiche.columns:
        fmin_HighPassFilter = 20
    else:
        fmin_HighPassFilter = analysis_fiche['fmin_HighPassFilter'][0] 
    
    
    folderName_audioFiles = str(int(maxtime_display_spectro))+'_'+str(int(analysis_fs))
    
    path_audio_files = os.path.join(path_osmose_dataset, dataset_ID, 'raw/audio', folderName_audioFiles )
    
    path_summstats = os.path.join(path_osmose_dataset,dataset_ID,'analysis','normaParams',folderName_audioFiles)      
    
#     tt = pd.read_csv(os.path.join(path_audio_files,'timestamp.csv'),header=None)    
        
    list_wav_withEvent_comp = glob.glob(os.path.join( path_audio_files , '*wav'))   
    list_wav_withEvent = list_wav_withEvent_comp[ind_min:ind_max]
    list_wav_withEvent = [os.path.basename(x) for x in list_wav_withEvent]
    
#     list_timestamp = list(tt[1].values)
#     list_timestamp = list_timestamp[ind_min:ind_max]

    list_summaryStats = []
#     for file,timm in zip(list_wav_withEvent,list_timestamp):
    for file in list_wav_withEvent:
        mn,ss = process_file(file)
        list_summaryStats.append([file,mn,ss])


    with open(os.path.join(path_summstats,'summaryStats_'+str(ind_min)+'.csv'), 'w') as f:

        write = csv.writer(f)
        write.writerow(['filename','mean','std'])
        write.writerows(list_summaryStats)        
        
        
    os.remove( os.path.join(path_analysisFolder, 'ongoing_pbsFiles', 'pbs_audioNormalization_' + str(ind_min) + '.pbs') )
