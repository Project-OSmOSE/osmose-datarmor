
from scipy import ndimage
import numpy as np
import pickle
import os
import pandas as pd

# def write_AED_results(diffsp,start_date ,end_date,out_path):
#
#     file1 = open(os.path.join(out_path , str(start_date)+'.csv'), "w")
#     print(out_path+str(start_date)+'.csv')
#     L = [str(start_date),',', str(end_date),',',str(diffsp),'\n']
#     file1.writelines(L)
#     file1.close()  # to change file access modes

def save_newPSD(welch_values,current_TOL,current_AUX , ind_time,path_analysisFolder,timescale,analysis_fs):

    analysis_fiche = pd.read_csv(os.path.join(path_analysisFolder, 'analysis_fiche.csv'), header=0)

    path_output_newFeatures = os.path.join(path_analysisFolder, 'getFeatures', str(int(analysis_fs)), timescale)

    frequencies = np.fft.rfftfreq(analysis_fiche['datasetScale_nfft'][0], 1 / analysis_fs)

    print('saving: ',path_output_newFeatures+'/'+str(ind_time[0])+'.pkl')
    with open(path_output_newFeatures+'/'+str(ind_time[0])+'.pkl', 'wb') as handle:
        pickle.dump([welch_values,current_TOL,current_AUX,ind_time,frequencies], handle)
