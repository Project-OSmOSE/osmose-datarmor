

import pandas as pd
import sys
from multiprocessing import Pool
import time
from module_saveNewVariables import *
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import glob
from datetime import datetime
from aux_functions import custom_date_range



# import pickle
# import pandas as pd
#
# [welch_values, tol,ind_time,frequencies] = pickle.load(open('complete.pkl', 'rb'))

# listt=ind_time[1:]
# time_diff = []
# for ii in range(len(listt)-1):
#     if int((pd.to_datetime(listt[ii+1][0]) - pd.to_datetime(listt[ii][0])).total_seconds()) != 3600:
#         print(listt[ii][0])
#         print(listt[ii+1][0])
#
#         print(int((pd.to_datetime(listt[ii+1][0]) - pd.to_datetime(listt[ii][0])).total_seconds()))
#
# plt.plot(time_diff)
# plt.show()

path_osmose_dataset = "/home/datawork-osmose/dataset/"
path_osmose_home = "/home/datawork-osmose/"

def process_file(current_time_periods):

    ind_beg = max(0,np.argmin(np.abs(json_timestamps_start  - pd.to_datetime(current_time_periods[0]).timestamp()))-1) # take index -1 to be sure that the concatenated json file will contain the first timesteps
    ind_end = min(len(json_filenames)-1,np.argmin(np.abs(json_timestamps_end  - pd.to_datetime(current_time_periods[1]).timestamp()))+1)

    print(ind_beg)
    print(ind_end)
    print(json_filenames[ind_beg])
    
    df_all = pd.read_json(json_filenames[ind_beg], lines=True)
    for cur_ind in range(ind_beg+1,ind_end+1):
        print(json_filenames[cur_ind])
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

    
    mm = pd.to_datetime(current_time_periods[1]) # min([pd.to_datetime(current_time_periods[1]),real_last_timestamp])
    
    timesteps = custom_date_range(pd.to_datetime(current_time_periods[0]),mm,newFeatures_timescale)
    nber_timesteps = len(timesteps)

    current_spectro = np.nan * np.ones(( int(analysis_fiche['datasetScale_nfft'][0]/2+1) , nber_timesteps-1 ))

    current_TOL = np.nan * np.ones(( lengthTOL , nber_timesteps-1 ))

    current_AUX = np.nan * np.ones(( len(aux_variable) , nber_timesteps-1 ))


    if len(welch_in_timePeriod)>0:

        for cpt_tt in range(nber_timesteps-1):

            # print('from: ',timesteps[cpt_tt])
            # print('to: ',timesteps[cpt_tt+1])

            mask = (json_timestep >= timesteps[cpt_tt]) & (
                    json_timestep < timesteps[cpt_tt+1])
            welch_in_timestep = welch_in_timePeriod.loc[mask]

            if len(welch_in_timestep) > 0:
#                 arr_welch_in_timestep = np.array(
#                     10 * np.log10(np.stack(welch_in_timestep['welch'].apply(lambda feat: feat[0])).astype(None)))
                arr_welch_in_timestep = np.array(
                    np.stack(welch_in_timestep['welch'].apply(lambda feat: feat[0])).astype(None))    
                current_spectro[:, cpt_tt] = np.nanmedian( arr_welch_in_timestep , 0)

                arr_TOL_in_timestep = np.array(
                    np.stack(welch_in_timestep['tol'].apply(lambda feat: feat[0])).astype(None))
                current_TOL[:, cpt_tt] = np.nanmedian(arr_TOL_in_timestep, 0)

                if aux_variable[0] in list(welch_in_timestep.columns):
                    
                    arr_AUX_in_timestep = welch_in_timestep[aux_variable].values
                    current_AUX[:, cpt_tt] = np.nanmedian(arr_AUX_in_timestep, 0)                                                
    
    current_AUX = np.transpose(current_AUX)
    current_TOL = np.transpose(current_TOL)
    current_spectro = np.transpose(current_spectro)

    save_newPSD(current_spectro,current_TOL,current_AUX,list(timesteps[:-1]),path_analysisFolder,newFeatures_timescale,analysis_fs)

if __name__ == "__main__":

    dataset_ID = sys.argv[1]
    analysis_fs = float(sys.argv[2])
    ind_min = int(sys.argv[3])
    ind_max = int(sys.argv[4])

    path_analysisFolder = os.path.join(path_osmose_dataset, dataset_ID, 'analysis')
    path_datasetScaleFeatures =  os.path.join(path_osmose_dataset, dataset_ID, 'analysis', 'datasetScale_features', str(int(analysis_fs)) )

    # load needed variables from analysis fiche
    analysis_fiche = pd.read_csv(os.path.join(path_analysisFolder, 'analysis_fiche.csv'), header=0)
    newFeatures_timescale = analysis_fiche['newFeatures_timescale'][0]
    
    if str(analysis_fiche['aux_variable'][0])=='nan':
        aux_variable = ['']    
    else:
        aux_variable = analysis_fiche['aux_variable'][0].split('-')

    # just need to estimate number of time periods here
    df = pd.read_csv(os.path.join(path_osmose_dataset, dataset_ID, 'raw/audio/timestamp.csv'), header=None)

    # here you need to have a frequency rr at least longer than newFeatures_timescale , so we simply double it
    rr = str(2*int(newFeatures_timescale[:len(newFeatures_timescale)-1]))+newFeatures_timescale[len(newFeatures_timescale)-1]
    time_periods = custom_date_range(df[1][0], df[1][len(df[1]) - 1], rr)
    # time_periods = time_periods[ind_min:ind_max]
    time_periods = [[st, ed] for st, ed in zip(time_periods[0:-1] , time_periods[1:])]
    # time_periods = pd.date_range(start=df[1][0], end=df[1][len(df[1]) - 1], periods=20)
    # time_periods = [[st, ed] for st, ed in zip(time_periods[0:-1] , time_periods[1:])]

    real_last_timestamp = pd.to_datetime(df[1][len(df[1]) - 1])

    # load json timestamps
    [json_timestamps_start, json_timestamps_end, json_filenames] = pickle.load( open(os.path.join(path_datasetScaleFeatures, 'timestamps_json.pkl'),'rb'))

    # just to get length of TOL vector
    tet = pd.read_json(json_filenames[0], lines=True)
    lengthTOL = len( tet['tol'][0][0] )

    # default params
    ncpus = 8

    t0=time.time()
    with Pool(processes=ncpus) as pool:
        pool.map(process_file, time_periods)
        pool.close()
    print(time.time()-t0)
    
#     for tt in time_periods:
#         print('loop:',tt)
#         process_file(tt)
        
    
    os.remove( os.path.join(path_analysisFolder, 'ongoing_pbsFiles', 'pbs_buildNewFeatures_0.pbs') )
 
    
