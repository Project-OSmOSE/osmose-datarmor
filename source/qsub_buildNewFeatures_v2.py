
import sys
import numpy as np
import pickle
import os
import pandas as pd

def _tob_bounds_from_toc(center_freq):
    return center_freq * np.power(10, np.array([-0.05, 0.05]))

path_osmose_home = "/home/datawork-osmose/"
path_osmose_dataset = "/home/datawork-osmose/dataset/"

dataset_ID = sys.argv[1]
analysis_fs = float(sys.argv[2])
ind_min = int(sys.argv[3])
ind_max = int(sys.argv[4])
       
path_analysisFolder = os.path.join(path_osmose_dataset, dataset_ID, 'analysis')
path_output_datasetScaleFeatures =  os.path.join(path_analysisFolder, 'datasetScale_features', str(int(analysis_fs)) )

# load needed variables from analysis fiche
analysis_fiche = pd.read_csv(os.path.join(path_analysisFolder, 'analysis_fiche.csv'), header=0)
exec('newFeatures_timescale=' + analysis_fiche['newFeatures_timescale'][0])

# welch frequency vector
frequencies = np.fft.rfftfreq(analysis_fiche['datasetScale_nfft'][0], 1 / analysis_fs)

# TOL frequency vector (should be read in /ongoing../compute.json)
upper_limit = analysis_fs/2
max_third_octave_index = np.floor(10 * np.log10(upper_limit))
tob_center_freqs = np.power(
    10, np.arange(0, max_third_octave_index + 1) / 10
)
all_tob = np.array([
    _tob_bounds_from_toc(toc_freq) for toc_freq in tob_center_freqs
])
frequency_TOL = all_tob[:,0]



json_files=[]
for subdir, dirs, files in os.walk(path_output_datasetScaleFeatures):
    for file in files:
        filepath = subdir + os.sep + file
        if filepath.endswith(".json"):
            json_files.append(filepath)    
# Sort files to have timestamps sorted not really needed
json_files = np.sort(json_files)  

if ind_min >= len(json_files):
    os.remove( os.path.join(path_analysisFolder, 'ongoing_pbsFiles', 'pbs_buildNewFeatures_'+str(ind_min)+'.pbs'))
    sys.exit() 
    
    
df_all = pd.concat([pd.read_json(f, lines=True) for f in json_files[ind_min:ind_max]], ignore_index=True)

# Put timestamps as index for data
df_all.set_index('timestamp', inplace=True, drop=True)
# Build two dataframes: one for spectrogram analysis, second for SPL
df_welch = df_all.drop(columns=['spl','tol'])
df_spl = df_all.drop(columns=['welch','tol'])
df_tol = df_all.drop(columns=['welch','spl'])

# Extract welch values
df_welch['welch'] = df_welch['welch'].apply(lambda l: l[0])
df_tol['tol'] = df_tol['tol'].apply(lambda l: l[0])    
    
    
    
# loop over the different time resolutions desired
for time_res in newFeatures_timescale:

    # Aggreagate welch values
    df_welch_agg = df_welch.groupby(pd.Grouper(freq=time_res)).apply(lambda x: x.values).to_frame(name='welch')
    df_tol_agg = df_tol.groupby(pd.Grouper(freq=time_res)).apply(lambda x: x.values).to_frame(name='tol')

    # Remove periods where recorder is off if needed
    df_welch_agg = df_welch_agg[df_welch_agg.astype(str)['welch'] != '[]']
    df_tol_agg = df_tol_agg[df_tol_agg.astype(str)['tol'] != '[]']

    ind_time = []
    welch_values = []
    for index, row in df_welch_agg.iterrows():
        welch_values.append(np.sum(np.array([np.array(xi[0]) for xi in row.welch]), axis=0))
        ind_time.append(index)
    del df_welch_agg
    welch_values = np.asarray(welch_values)
    welch_values = 10 * np.log10(welch_values)

    # for TOL
    ind_time = []
    tol_values = []
    for index, row in df_tol_agg.iterrows():

        # dummy way to remove '-Infinity' values in TOL 
        M=np.array([np.array(xi[0]).astype(float) for xi in row.tol])
        ind = np.where( np.isinf(M))
        for t in ind[0]:
            for y in ind[1]:
                M[t,y] = M[t,y-1]
                ind = np.where(np.isinf(M))

        tol_values.append(np.mean(M ,axis=0))
        ind_time.append(index)

    del df_tol_agg
    tol_values = np.asarray(tol_values)


    with open(os.path.join(path_analysisFolder, 'soundscape','raw_welch',time_res,'intermediary_'+str(ind_time[0])+'.pkl'), 'wb') as handle:
        pickle.dump([welch_values,ind_time,frequencies,tol_values,frequency_TOL], handle, protocol=4)

        
        
os.remove( os.path.join(path_analysisFolder, 'ongoing_pbsFiles', 'pbs_buildNewFeatures_' + str(ind_min) + '.pbs') )
