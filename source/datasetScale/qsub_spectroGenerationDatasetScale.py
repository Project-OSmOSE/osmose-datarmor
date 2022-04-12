import pandas as pd
import sys
from multiprocessing import Pool
import time
from module_soundscape import *
from module_activityFuncs import *
from module_saveNewVariables import *
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import glob
from datetime import datetime
from aux_functions import custom_date_range

import json

path_osmose_dataset = "/home/datawork-osmose/dataset/"
path_osmose_home = "/home/datawork-osmose/"

def process_file(current_time_periods):

    # explanation : you have two time variables
    # - time_periods : long time periods which are distributed over parallel batches and multiple threads. It is the time period
    # over which LTAS or soundscape metrics can be generated ;
    # - timesteps : short time periods within each current_time_periods used to re-aggregate features or compute detection ...

    ind_beg = max(0,np.argmin(np.abs(json_timestamps_start  - pd.to_datetime(current_time_periods[0]).timestamp()))-1) # take index -1 to be sure that the concatenated json file will contain the first timesteps
    ind_end = min(len(json_filenames)-1,np.argmin(np.abs(json_timestamps_end  - pd.to_datetime(current_time_periods[1]).timestamp()))+1)

    print(ind_beg)
    print(ind_end)
#     print(json_filenames[ind_beg])
    
    df_all = pd.read_json(json_filenames[ind_beg], lines=True)
    for cur_ind in range(ind_beg+1,ind_end+1):
        df_all = pd.concat([df_all, pd.read_json(json_filenames[cur_ind], lines=True) ])

    utc_json_timestamps=[]
    for curdt in df_all['timestamp']:
        utc_json_timestamps.append(pd.to_datetime(curdt,utc=True))
    utc_json_timestamps = np.array(utc_json_timestamps)

    mask = (utc_json_timestamps >= current_time_periods[0]) & (
            utc_json_timestamps < current_time_periods[1])

    welch_in_timePeriod = df_all.loc[mask]
    ind_time_timePer = welch_in_timePeriod['timestamp']

    json_timestep=[]
    for curdt in welch_in_timePeriod['timestamp']:
        json_timestep.append(pd.to_datetime(curdt,utc=True))
    json_timestep = np.array(json_timestep)
    
    if len(json_timestep)>0:
        print(json_timestep[0],'must be smaller than',current_time_periods[0])
        print(json_timestep[-1],'must be higher than',current_time_periods[1])
        print('if not so, you do not take enough json files..')

    nber_timesteps = 1080
    timesteps = pd.date_range(start=pd.to_datetime(current_time_periods[0]), end=pd.to_datetime(current_time_periods[1]), periods=nber_timesteps)
    timesteps = timesteps - (timesteps[0] - pd.to_datetime(current_time_periods[0], utc=True))
    
    # currently does not work, problem with (df_all['timestamp'][1]-df_all['timestamp'][0]) , has a type <class 'pandas.core.series.Series'> which does not have total_seconds
#     print(type(timesteps))
#     print(type((df_all['timestamp'][1]-df_all['timestamp'][0])))
#     print((timesteps[1]-timesteps[0]).total_seconds())
#     print((df_all['timestamp'][1]-df_all['timestamp'][0]))
#     print((df_all['timestamp'][1]-df_all['timestamp'][0]).total_seconds())
#     if (timesteps[1]-timesteps[0]).total_seconds() < (df_all['timestamp'][1]-df_all['timestamp'][0]).total_seconds():
#         print('$$$$$$$$$ explanation of VERTICAL WHITE BANDS !!!!!')
#         print('$$$$$$$$$ timestep of spectograms must be larger than time resolutions of your features in json !!!!')
    

    current_spectro = np.nan * np.ones(( int(analysis_fiche['datasetScale_nfft'][0]/2+1) , nber_timesteps-1 ))
    current_auxData = np.nan * np.ones(( nber_timesteps-1,1 ))
    aux_data=[]

    if len(welch_in_timePeriod)>0:

        for cpt_tt in range(nber_timesteps-1):

            mask = (json_timestep >= timesteps[cpt_tt]) & (
                    json_timestep < timesteps[cpt_tt+1])
            welch_in_timestep = welch_in_timePeriod.loc[mask]
            
            if len(welch_in_timestep) > 0:
                arr_welch_in_timestep = np.array(
                    10 * np.log10(np.stack(welch_in_timestep['welch'].apply(lambda feat: feat[0])).astype(None)))

                current_spectro[:, cpt_tt] = np.nanmedian( arr_welch_in_timestep , 0)

                if aux_variable in list(welch_in_timestep): 
                    aux_data = welch_in_timestep[aux_variable].values
                    current_auxData[cpt_tt] = np.nanmedian(aux_data)

    current_spectro = np.transpose(current_spectro)
    
    print( current_time_periods[0] , current_time_periods[1])
    
    with open(path_output_spectrogramsDatasetScale+'ficheInfo.txt', 'w') as f:
        f.write('Shape of spectrogram : '+ str(current_spectro.shape) +'\n')       
    
    print(current_spectro.shape)
    print(list(timesteps[:-1]))
    
    plot_LTAS_withAuxVar(current_spectro,list(timesteps[:-1]),path_analysisFolder,path_output_spectrogramsDatasetScale,datasetScale_maxDisplaySpectro,analysis_fs,current_auxData)


if __name__ == "__main__":

    dataset_ID = sys.argv[1]
    analysis_fs = float(sys.argv[2])
    ind_min = int(sys.argv[3])
    ind_max = int(sys.argv[4])

    path_analysisFolder = os.path.join(path_osmose_dataset, dataset_ID, 'analysis')
    path_datasetScaleFeatures =  os.path.join(path_osmose_dataset, dataset_ID, 'analysis', 'datasetScale_features', str(int(analysis_fs)) )

    # load needed variables from analysis fiche
    analysis_fiche = pd.read_csv(os.path.join(path_analysisFolder, 'analysis_fiche.csv'), header=0)
    datasetScale_maxDisplaySpectro = analysis_fiche['datasetScale_maxDisplaySpectro'][0]
    aux_variable = analysis_fiche['aux_variable'][0]

#     path_output_spectrogramsDatasetScale =  os.path.join(path_osmose_dataset, dataset_ID, 'analysis', 'spectrograms', str(int(analysis_fs)) , 'datasetScale' , datasetScale_maxDisplaySpectro )
    path_output_spectrogramsDatasetScale = os.path.join(path_analysisFolder, 'spectrograms',analysis_fiche['datasetScale_maxDisplaySpectro'][0]+'_'+str(int(analysis_fs)) )

    # just need to estimate number of time periods here
    metadata = pd.read_csv( os.path.join(path_osmose_dataset , dataset_ID , 'raw/metadata.csv') )    
    orig_fileDuration = metadata['orig_fileDuration'][0]    
    folderName_audioFiles = str(int(orig_fileDuration))+'_'+str(int(analysis_fs)) 
    output_path_audio_files = os.path.join(path_osmose_dataset, dataset_ID, 'raw','audio', folderName_audioFiles )
    
    df = pd.read_csv(os.path.join(output_path_audio_files, 'timestamp.csv'), header=None)  
    
    start_timestamp = df[1][0]
    
    if 'M' in datasetScale_maxDisplaySpectro:
        datasetScale_maxDisplaySpectro=datasetScale_maxDisplaySpectro+'S'
        start_timestamp = pd.to_datetime(df[1][0], utc=True) - pd.DateOffset(months=1) # this is done because of the behavior of date_range with '1MS' which begins at the month after the ongoing one
    
    if str(analysis_fiche['datasetScale_maxDisplaySpectro'][0]) != 'all':
        time_periods = custom_date_range(start_timestamp, df[1][len(df[1]) - 1], datasetScale_maxDisplaySpectro)
    else:
        time_periods = [ pd.to_datetime(start_timestamp, utc=True) , pd.to_datetime(df[1][len(df[1]) - 1] , utc=True)]
        
    print(ind_min)
    print(ind_max)
    time_periods = time_periods[ind_min:ind_max]
    time_periods = [[st, ed] for st, ed in zip(time_periods[0:-1] , time_periods[1:])]        
        
    # load json timestamps
    [json_timestamps_start, json_timestamps_end, json_filenames] = pickle.load( open(os.path.join(path_datasetScaleFeatures, 'timestamps_json.pkl'),'rb'))

    # default params
    ncpus = 8
    
    if str(analysis_fiche['datasetScale_maxDisplaySpectro'][0]) == 'all':
    
        for tt in time_periods:
            print('%%%%%%%%%%%%START :',tt)
            process_file(tt)
        
    else:
        
        t0=time.time()
        with Pool(processes=ncpus) as pool:
            pool.map(process_file, time_periods)
            pool.close()
        print(time.time()-t0)


    os.remove( os.path.join(path_analysisFolder, 'ongoing_pbsFiles', 'pbs_spectroGenerationDatasetScale_' + str(ind_min) + '.pbs') )

 