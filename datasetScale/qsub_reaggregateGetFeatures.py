import matplotlib.pyplot as plt

import pandas as pd
import glob
import os
import numpy as np
import os.path
import sys

import glob
import numpy as np

import pickle



path_osmose_dataset = "/home/datawork-osmose/dataset/"

if __name__ == "__main__":

    dataset_ID = sys.argv[1]
    analysis_fs = float(sys.argv[2])

    path_analysisFolder = os.path.join(path_osmose_dataset, dataset_ID, 'analysis')

    # load needed variables from analysis fiche
    analysis_fiche = pd.read_csv( os.path.join(path_analysisFolder,'analysis_fiche.csv'),header=0)
    newFeatures_timescale = analysis_fiche['newFeatures_timescale'][0]
    normalizeByMax_welch = analysis_fiche['normalizeByMax_welch'][0]


    if str(analysis_fiche['aux_variable'][0])=='nan':
        aux_variable = ['']    
    else:
        aux_variable = analysis_fiche['aux_variable'][0].split('-')
        
    path_output_newFeatures = os.path.join(path_analysisFolder, 'getFeatures', str(int(analysis_fs)), newFeatures_timescale)

    nfft=int(analysis_fiche['datasetScale_nfft'][0] / 2 + 1)

    # just to get length of TOL vector
    path_datasetScaleFeatures =  os.path.join(path_osmose_dataset, dataset_ID, 'analysis', 'datasetScale_features', str(int(analysis_fs)) )
    [json_timestamps_start, json_timestamps_end, json_filenames] = pickle.load( open(os.path.join(path_datasetScaleFeatures, 'timestamps_json.pkl'),'rb'))
    tet = pd.read_json(json_filenames[0], lines=True)
    lengthTOL = len( tet['tol'][0][0] )

    total_time = np.empty((1, 1))
    total_welch = np.empty((1, nfft))
    total_tol = np.empty((1, lengthTOL))
    total_AUX = np.empty((1, len(aux_variable) ))

    for gg in sorted(glob.glob(os.path.join(path_output_newFeatures,'*.pkl'))):

        [welch_values, tol,current_AUX,ind_time,frequencies] = pickle.load(open(gg, 'rb'))

        total_AUX = np.vstack((total_AUX, current_AUX))
        total_welch = np.vstack((total_welch, welch_values))
        total_tol = np.vstack((total_tol, tol))
        total_time = np.vstack((total_time, np.array(ind_time).reshape(-1, 1)))

    # remove first elements because of np.empty , which creates a weird floating number
    total_time = total_time[1:]
    total_welch = total_welch[1:,:]
    total_tol = total_tol[1:,:]
    total_AUX = total_AUX[1:,:]
    
    if normalizeByMax_welch:
        total_welch = total_welch / np.nanmax(total_welch)

    with open( os.path.join(path_output_newFeatures,'complete.pkl'), 'wb') as handle:
        pickle.dump([total_welch,total_tol,total_AUX,total_time,frequencies], handle)
        
    os.remove( os.path.join(path_analysisFolder, 'ongoing_pbsFiles', 'pbs_reaggregateGetFeatures_0.pbs') )
        