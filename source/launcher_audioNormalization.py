# current limits: 
# - in def main_resample , when you have eg 3 segments in your audio file, it will generate batches over nber audio files and not on generated segments 

import numpy as np
import os
import pandas as pd
import subprocess
import glob

path_osmose_home = "/home/datawork-osmose/"
path_osmose_dataset = "/home/datawork-osmose/dataset/"


def main(dataset_ID,analysis_fs,maxtime_display_spectro):

    global path_analysisFolder, path_pbsFiles, path_output_spectrograms
    path_analysisFolder = os.path.join(path_osmose_dataset, dataset_ID, 'analysis')
    path_pbsFiles = os.path.join(path_analysisFolder, 'ongoing_pbsFiles')
    
       
    ## initialize some metadata as global variables    
    global total_nber_audio_files, orig_fs, orig_fileDuration, orig_total_nber_audio_files, folderName_audioFiles
    metadata = pd.read_csv( os.path.join(path_osmose_dataset , dataset_ID , 'raw/metadata.csv') )
    orig_fs = metadata['orig_fs'][0]
    total_nber_audio_files = metadata['nberWavFiles'][0]        
    orig_total_nber_audio_files = metadata['nberWavFiles'][0]        
    orig_fileDuration = metadata['orig_fileDuration'][0]    
    
    if isinstance(maxtime_display_spectro,str):
        maxtime_display_spectro = int(orig_fileDuration)

#     if int(maxtime_display_spectro)>3600:    
#         print('for safety reasons, for the moment maxtime_display_spectro must be inferior or equal to 3600 s .. sorry !')
#         sys.exit()

    if int(maxtime_display_spectro) != int(orig_fileDuration):
        folderName_audioFiles = str(maxtime_display_spectro)+'_'+str(int(analysis_fs))        
        total_nber_audio_files = int(np.floor(orig_fileDuration / maxtime_display_spectro) * total_nber_audio_files)
    else:
#         folderName_audioFiles = str(int(analysis_fs))    
        folderName_audioFiles = str(int(orig_fileDuration))+'_'+str(int(analysis_fs))   

    
    if not os.path.exists(path_analysisFolder):
        os.makedirs(path_analysisFolder)     
    if not os.path.exists(path_pbsFiles):
        os.makedirs(path_pbsFiles)     
    if not os.path.exists(os.path.join(path_analysisFolder,'normaParams')):
        os.makedirs(os.path.join(path_analysisFolder,'normaParams'))  
    if not os.path.exists(os.path.join(path_analysisFolder,'normaParams',folderName_audioFiles)):
        os.makedirs(os.path.join(path_analysisFolder,'normaParams',folderName_audioFiles))  
        
        
    
    
    global output_path_audio_files
    output_path_audio_files = os.path.join(path_osmose_dataset, dataset_ID, 'raw','audio', folderName_audioFiles )     
    
    # write analysis.csv and make it global. Must be after the previous sys.exit() because ...
    global analysis_fiche
    data = {'dataset_ID' :dataset_ID
            ,'analysis_fs' :float(analysis_fs),'maxtime_display_spectro':maxtime_display_spectro}
    analysis_fiche = pd.DataFrame.from_records([data])
    analysis_fiche.to_csv( os.path.join(path_analysisFolder,'analysis_fiche.csv') )
            
    main_audioNormalization(dataset_ID,analysis_fs)    
    
    list_pbs_audioNormalization = sorted(glob.glob(os.path.join(path_pbsFiles, 'pbs_audioNormalization*') ))    
        

    job_audioNormalization=[]
    for job in list_pbs_audioNormalization:
        res = subprocess.run(['qsub', job], stdout=subprocess.PIPE).stdout.decode('utf-8').rstrip('\n')
        job_audioNormalization.append(res)        
        
        
    # set permissions to /analysis
    print('debug1:',os.path.join(path_osmose_dataset, dataset_ID))
    createQsub_setPermissions(dataset_ID,os.path.join(path_osmose_dataset, dataset_ID),folderName_audioFiles)


    if isinstance(job_audioNormalization,list):
        subprocess.run( ['qsub','-W depend=afterok:'+(':').join(job_audioNormalization),os.path.join(path_pbsFiles, 'pbs_setPermissions_0.pbs')],stdout=subprocess.PIPE).stdout.decode('utf-8')
    else:
        subprocess.run( ['qsub','-W depend=afterok:'+job_audioNormalization,os.path.join(path_pbsFiles, 'pbs_setPermissions_0.pbs')],stdout=subprocess.PIPE).stdout.decode('utf-8')
            
            
def createQsub_setPermissions(dataset_ID,path_to_set,folderName_audioFiles):

    cur_pbs_file = os.path.join(path_pbsFiles, 'pbs_setPermissions_0.pbs')
    logjob_outpath = os.path.join(path_pbsFiles, 'log_setPermissions_0.txt')
    
    with open(path_osmose_home+"osmoseNotebooks_v0/source/templateQsub_setPermissions.pbs", "r") as template:
        template_lines = template.readlines()
    
    with open(cur_pbs_file, "w") as pbs_file:
        for line in template_lines:
            pbs_file.write(
                line.replace("dataset_ID", dataset_ID)\
                    .replace("logjob_outpath",logjob_outpath)\
                    .replace("folderName_audioFiles",folderName_audioFiles)\
                    .replace("path_to_set",path_to_set)                
            )

     
        
            
        
def main_audioNormalization(dataset_ID,analysis_fs):
                 
       
    cur_total_nber_audio_files = len(glob.glob(output_path_audio_files+'/*wav'))
    
    size_batch = 200
    id_job=0

    if cur_total_nber_audio_files < size_batch:
        createQsub_audioNormalization(dataset_ID,analysis_fs,id_job, 0, cur_total_nber_audio_files)
    else:
        for ind in np.arange(0, cur_total_nber_audio_files, size_batch):
            
            if ind == np.arange(0, cur_total_nber_audio_files, size_batch)[-1]:
                ind_max = cur_total_nber_audio_files
            else:
                ind_max = ind + size_batch
            ind_min = ind

            createQsub_audioNormalization(dataset_ID,analysis_fs,id_job, ind_min, ind_max)
            id_job += 1        

def createQsub_audioNormalization(dataset_ID,analysis_fs,id_job, ind_min, ind_max):
        
    # build a few paths
    path_analysisFolder = os.path.join(path_osmose_dataset, dataset_ID, 'analysis')
    
    cur_pbs_file = os.path.join(path_pbsFiles, 'pbs_audioNormalization_' + str(ind_min) + '.pbs')
    logjob_outpath = os.path.join(path_pbsFiles, 'log_audioNormalization_' + str(ind_min) + '.txt')

    with open( os.path.join(path_osmose_home , "osmoseNotebooks_v0/source/templateQsub_audioNormalization.pbs"), "r") as template:
        template_lines = template.readlines()

    with open(cur_pbs_file, "w") as pbs_file:
        for line in template_lines:
            pbs_file.write(
                line.replace("dataset_ID", dataset_ID) \
                    .replace("analysis_fs", str(int(analysis_fs))) \
                    .replace("ind_min", str(ind_min)) \
                    .replace("ind_max", str(ind_max)) \
                    .replace("logjob_outpath", logjob_outpath)
            )

            


            