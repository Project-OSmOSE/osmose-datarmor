# trial for improvement : i tried to replace by json.loads(jj) but https://stackoverflow.com/questions/48140858/json-decoder-jsondecodeerror-extra-data-line-2-column-1-char-190

import matplotlib.pyplot as plt

import pandas as pd
import glob
import os
import numpy as np
import os.path
import sys

from datetime import datetime

import time

import pickle

import json

path_osmose_dataset = "/home/datawork-osmose/dataset/"


def _tob_bounds_from_toc(center_freq):
    return center_freq * np.power(10 , np.array([-0.05 , 0.05]) )



if __name__ == "__main__":

    dataset_ID = sys.argv[1]
    analysis_fs = float(sys.argv[2])

    # build a few paths
    path_analysisFolder = os.path.join(path_osmose_dataset, dataset_ID, 'analysis')
    path_output_datasetScaleFeatures =  os.path.join(path_analysisFolder, 'datasetScale_features', str(int(analysis_fs)) )

    
    df_all = pd.read_json(os.path.join(path_osmose_dataset, dataset_ID, 'analysis','ongoing_pbsFiles/compute.json'), lines=True)
                    
    max_third_octave_index = np.floor(10 * np.log10(df_all['high_freq_tol'][0]))
    tob_center_freqs = np.power(
        10, np.arange(0, max_third_octave_index + 1) / 10
    )
    all_tob = np.array([
        _tob_bounds_from_toc(toc_freq) for toc_freq in tob_center_freqs
    ])

    with open(os.path.join(path_output_datasetScaleFeatures,'TOL_frequencyVector.pkl'), 'wb') as handle:
        pickle.dump(all_tob, handle)

    list_timestamps=[]
    nber_welc_per_json=5
    
    # create the timestamps of json files
    json_timestamps_start = []
    json_timestamps_end = []
    json_filenames = []

    for subdir, dirs, files in os.walk(path_output_datasetScaleFeatures):
        files.sort()
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".json"):
                df_all = pd.read_json(filepath, lines=True)

                if 'timestamp' in df_all:# some json can be empty
                    print(filepath)
                    print(df_all['timestamp'].values[0])
                    print(df_all['timestamp'].values[-1])

                    json_timestamps_start.append(df_all['timestamp'].values[0])
                    json_timestamps_end.append(df_all['timestamp'].values[-1])
                    json_filenames.append(filepath)
                    
                    
                    # for LS aggregation
#                     arr_welch_in_timestep = np.stack(df_all['welch'].apply(lambda feat: feat[0]))

#                     if 'welch_concat' not in locals():
#                         welch_concat = np.zeros((1,arr_welch_in_timestep.shape[1])) 

#                     vec_ind = np.arange(0 , arr_welch_in_timestep.shape[0] , round(arr_welch_in_timestep.shape[0] / nber_welc_per_json))
#                     for ind in range(len(vec_ind)-1):

#                         arr_welch_in_timestepctro = np.transpose(np.nanmedian( arr_welch_in_timestep[vec_ind[ind] : vec_ind[ind+1] , :] , 0))

#                         welch_concat = np.concatenate([welch_concat, arr_welch_in_timestepctro[np.newaxis,:] ])

#                         list_timestamps.append(df_all['timestamp'][round( (vec_ind[ind] + vec_ind[ind+1])/2 )])                    
                    

                else:
                    print("empty json:",filepath)

                    
#     welch_concat = welch_concat[1:,:]

#     with open(os.path.join(path_json ,'LSfeatures_.pkl'), 'wb') as handle:
#         pickle.dump([welch_concat,list_timestamps], handle)
        
                    
    json_timestamps_start_array = []
    for cc in json_timestamps_start:
        json_timestamps_start_array.append(pd.to_datetime(cc).timestamp())
    json_timestamps_start_array = np.array(json_timestamps_start_array)

    json_timestamps_end_array = []
    for cc in json_timestamps_end:
        json_timestamps_end_array.append(pd.to_datetime(cc).timestamp())
    json_timestamps_end_array = np.array(json_timestamps_end_array)

    with open(os.path.join(path_output_datasetScaleFeatures, 'timestamps_json.pkl'), 'wb') as handle:
        pickle.dump([json_timestamps_start_array, json_timestamps_end_array, json_filenames], handle)

    os.remove( os.path.join(path_analysisFolder, 'ongoing_pbsFiles', 'pbs_timestampJsonCreation.pbs'))
    