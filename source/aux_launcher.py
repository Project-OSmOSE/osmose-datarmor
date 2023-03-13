# current limits: 
# - in def main_resample , when you have eg 3 segments in your audio file, it will generate batches over nber audio files and not on generated segments 


import sys
import numpy as np
import os
import glob
import shutil
import pandas as pd
import subprocess
import math  

from ipywidgets import interact, interact_manual
from IPython.display import Image


path_osmose_home = "/home/datawork-osmose/"
path_osmose_dataset = "/home/datawork-osmose/dataset/"

size_batch = 100

def main(dataset_ID,analysis_fs, fileScale_nfft ,fileScale_winsize,nberAdjustSpectros,fileScale_overlap,colmapspectros,nber_zoom_levels,min_color_val,maxtime_display_spectro,formatForAPLOSE):

    ## initialize some metadata as global variables    
    global total_nber_audio_files, orig_fs, audioFile_duration, orig_total_nber_audio_files
    metadata = pd.read_csv( os.path.join(path_osmose_dataset , dataset_ID , 'raw/metadata.csv') )
    orig_fs = metadata['orig_fs'][0]
    total_nber_audio_files = metadata['nberWavFiles'][0]        
    orig_total_nber_audio_files = metadata['nberWavFiles'][0]        
    audioFile_duration = metadata['audioFile_duration'][0]    
    
    if int(maxtime_display_spectro) != int(audioFile_duration):
        folderName_audioFiles = str(maxtime_display_spectro)+'_'+str(int(analysis_fs))        
        total_nber_audio_files = int(np.floor(audioFile_duration / maxtime_display_spectro) * total_nber_audio_files)
        
    else:
        folderName_audioFiles = str(int(analysis_fs))    
    
    ## initialize some paths as global variables
    global path_analysisFolder, path_pbsFiles, path_output_spectrograms
    path_analysisFolder = os.path.join(path_osmose_dataset, dataset_ID, 'analysis')
    path_pbsFiles = os.path.join(path_analysisFolder, 'ongoing_pbsFiles')    
    path_output_spectrograms = os.path.join(path_analysisFolder, 'spectrograms', folderName_audioFiles)
    
    if not os.path.exists(path_pbsFiles):
        os.makedirs(path_pbsFiles)     
    if not os.path.exists(path_output_spectrograms):
        os.makedirs(path_output_spectrograms)     
        
    # write analysis.csv and make it global
    global analysis_fiche
    data = {'dataset_ID' :dataset_ID
            ,'analysis_fs' :float(analysis_fs)
            ,'fileScale_nfft' : fileScale_nfft,'fileScale_winsize' : fileScale_winsize,'fileScale_overlap' : fileScale_overlap
           ,'colmapspectros' : colmapspectros,'nber_zoom_levels' : nber_zoom_levels,'nberAdjustSpectros':nberAdjustSpectros,
           'min_color_val':min_color_val,'maxtime_display_spectro':maxtime_display_spectro,'formatForAPLOSE':formatForAPLOSE}
    analysis_fiche = pd.DataFrame.from_records([data])
    analysis_fiche.to_csv( os.path.join(path_analysisFolder,'analysis_fiche.csv') )
            
    # re-initialize the folder of pbs files 
    if len(glob.glob(os.path.join(path_pbsFiles, 'pbs_resample*'))) > 0:
        print('Ongoing RESAMPLING, check the Track Progress cell.. NO new generation possible for the moment')
        sys.exit()    

    elif len(glob.glob(os.path.join(path_pbsFiles, 'pbs_spectroGenerationFileScale*'))) > 0: 
        
        if analysis_fiche['nberAdjustSpectros'][0]==0:
            print('Ongoing generation of ALL spectrograms, check the Track Progress cell.. NO new generation possible for the moment')
        else:
            print('Ongoing generation of parameter setting spectrograms; wait it is done before running new spectrograms, it will not take long..')
            
        sys.exit()

    else:
        if not os.path.exists(path_pbsFiles):
            os.makedirs(path_pbsFiles)      
        else:
            shutil.rmtree(path_pbsFiles)
            os.makedirs(path_pbsFiles)        
        
    global output_path_audio_files
    output_path_audio_files = os.path.join(path_osmose_dataset, dataset_ID, 'raw','audio', folderName_audioFiles )
            
    # create all pbs files in /home/datawork-osmose/dataset/<dataset_ID>/analysis/ongoing_pbsFiles/
    if (not os.path.exists( output_path_audio_files )) and (analysis_fiche['nberAdjustSpectros'][0]==0):
        
        os.makedirs(output_path_audio_files)               

        main_resample(dataset_ID,analysis_fs,str(int(orig_fs)),folderName_audioFiles)    
                       
        print('OK we are generating all data for your annotation campaign now ! \n \n You can look at the progress bar in the cell below, or you can leave Jupyter and come back to Earth, your job is done, we will mail you when it is done ;) ')        
        
    elif analysis_fiche['nberAdjustSpectros'][0]!=0:
        print('Generating',analysis_fiche['nberAdjustSpectros'][0],'spectrograms for parameter selection! In a few seconds, your first spectrograms will be visible in the next cell..')
        
   # else:# (not os.path.exists( path_output_spectrograms )) or (analysis_fiche['nberAdjustSpectros'][0]!=0):
    main_spectroGenerationFileScale(dataset_ID,analysis_fs)    
                           
#     list_pbs_decoupe = sorted( glob.glob(os.path.join(path_pbsFiles, 'pbs_decoupe*')) )
    list_pbs_resample = sorted( glob.glob(os.path.join(path_pbsFiles, 'pbs_resample*')) )
    list_pbs_spectroGeneFileScale = sorted(glob.glob(os.path.join(path_pbsFiles, 'pbs_spectroGenerationFileScale*') ))
        
    job_resample = []
    for job in list_pbs_resample:    
        res = subprocess.run(['qsub', job], stdout=subprocess.PIPE).stdout.decode('utf-8').rstrip('\n')
        job_resample.append(res)

#     for job in list_pbs_spectroGeneFileScale:
#         res = subprocess.run( ['qsub','-W depend=afterok:'+(':').join(job_resample),job],stdout=subprocess.PIPE).stdout.decode('utf-8')
        
                       
        
## Resample        
        
def main_resample(dataset_ID,analysis_fs,folderIn,folderOut):
        
    # use multiple nodes with batch of size_batch files ; use it if number of files higher than 1000
    id_job=0 # used to identify the different jobs created in case total_nber_audio_files > size_batch
    if orig_total_nber_audio_files<size_batch:
        createQsub_resample(dataset_ID,analysis_fs,id_job,0,orig_total_nber_audio_files,folderIn,folderOut)
    else:    
        for ind in np.arange(0, orig_total_nber_audio_files, size_batch):
            if ind == np.arange(0, orig_total_nber_audio_files, size_batch)[-1]:
                ind_max = orig_total_nber_audio_files
            else:
                ind_max = ind + size_batch
            ind_min = ind
            createQsub_resample(dataset_ID,analysis_fs,id_job,ind_min, ind_max,folderIn,folderOut)
            id_job+=1       

            
            
def createQsub_resample(dataset_ID,analysis_fs,id_job,ind_min,ind_max,folderIn,folderOut):

    cur_pbs_file = os.path.join(path_pbsFiles, 'pbs_resample_'+str(ind_min)+'.pbs')
    logjob_outpath = os.path.join(path_pbsFiles, 'log_resample_'+str(ind_min)+'.txt')

    with open(path_osmose_home+"notebook_source/templateQsub_resample.pbs", "r") as template:
        template_lines = template.readlines()
    
    with open(cur_pbs_file, "w") as pbs_file:
        for line in template_lines:
            pbs_file.write(
                line.replace("dataset_ID", dataset_ID)\
                    .replace("folderIn", folderIn)\
                    .replace("analysis_fs", str(int(analysis_fs)))\
                    .replace("ind_min", str(ind_min)) \
                    .replace("ind_max", str(ind_max)) \
                    .replace("logjob_outpath",logjob_outpath) \
                    .replace("folderOut",folderOut)\
                    .replace("new_audioFileDuration",str(analysis_fiche['maxtime_display_spectro'][0]))\
                    .replace("orig_audioFileDuration",str(int(audioFile_duration)))\
                    .replace("nber_segments",str(int(audioFile_duration / analysis_fiche['maxtime_display_spectro'][0])))
            )

     
    


              
## spectroGenerationFileScale
        
def main_spectroGenerationFileScale(dataset_ID,analysis_fs):
               
    if analysis_fiche['nberAdjustSpectros'][0]!=0:
        if os.path.exists(os.path.join(path_output_spectrograms , 'spectro_adjustParams')):
            shutil.rmtree(os.path.join(path_output_spectrograms , 'spectro_adjustParams'))
        os.makedirs(os.path.join(path_output_spectrograms , 'spectro_adjustParams'))
        cur_total_nber_audio_files = min([total_nber_audio_files, analysis_fiche['nberAdjustSpectros'][0]])
        
    else:        
        cur_total_nber_audio_files = total_nber_audio_files
        
        # re-initialize the folder of spectrograms
        if os.path.exists(path_output_spectrograms):
            shutil.rmtree(path_output_spectrograms)
        os.makedirs(path_output_spectrograms)        
    
    # use multiple nodes with batch of size_batch files ; use it if number of files higher than 1000
    id_job = 0  # used to identify the different jobs created in case cur_total_nber_audio_files > size_batch
    if cur_total_nber_audio_files < size_batch:
        createQsub_spectroGenerationFileScale(dataset_ID,analysis_fs,id_job, 0, cur_total_nber_audio_files)
    else:
        for ind in np.arange(0, cur_total_nber_audio_files, size_batch):
            
            if ind == np.arange(0, cur_total_nber_audio_files, size_batch)[-1]:
                ind_max = cur_total_nber_audio_files
            else:
                ind_max = ind + size_batch
            ind_min = ind

            createQsub_spectroGenerationFileScale(dataset_ID,analysis_fs,id_job, ind_min, ind_max)
            id_job += 1        

def createQsub_spectroGenerationFileScale(dataset_ID,analysis_fs,id_job, ind_min, ind_max):
        
    # build a few paths
    path_analysisFolder = os.path.join(path_osmose_dataset, dataset_ID, 'analysis')
    
    cur_pbs_file = os.path.join(path_pbsFiles, 'pbs_spectroGenerationFileScale_' + str(ind_min) + '.pbs')
    logjob_outpath = os.path.join(path_pbsFiles, 'log_spectroGenerationFileScale_' + str(ind_min) + '.txt')

    with open( os.path.join(path_osmose_home , "notebook_source/templateQsub_spectroGenerationFileScale.pbs"), "r") as template:
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

            


            
## aux

def visu_adjustSpectros():

    fdir = os.path.join(path_output_spectrograms , 'spectro_adjustParams')
    
    if len(os.listdir(fdir))==0:
        print('Spectrograms not ready yet, please wait a bit..')
        return

    @interact
    def show_images(file=np.sort(os.listdir(fdir))):
        display(Image(os.path.join(fdir,file)))
        Image.height=12000
        Image.width=12000   
        
        
                            
def job_monitoring(dataset_ID,analysis_fs):
        
    if "total_nber_audio_files" not in globals():
        print('Hmm , I guess your notebook was restarted no ? well you have to rerun the first cells up to the cell <Generate spectrograms> to re-initialize me now! and do not worry it will not re-send your jobs ..')
        sys.exit()
                       
    # RESAMPLING
    if len( glob.glob(os.path.join(path_pbsFiles, 'pbs_resample*')) ):
        
        if len(glob.glob(os.path.join(output_path_audio_files, '*.wav'))) == total_nber_audio_files:
            resampling_status = 'DONE'
        else:
            resampling_status = 'ONGOING'
            
        if int(audioFile_duration) != int(analysis_fiche['maxtime_display_spectro'][0]):
            jobname = 'Segmenting & Resampling'
        else:
            jobname = 'Resampling'
            
        print(resampling_status +' (', len(glob.glob(os.path.join(output_path_audio_files, '*.wav'))), '/',str(total_nber_audio_files), ')' + ' -> '+   jobname  )   
    
    
    
    # FILESCALE_SPECTRO_GENERATION    
    if len( glob.glob(os.path.join(path_pbsFiles, 'pbs_spectroGenerationFileScale*')) ):
    
        if len(next(os.walk(path_output_spectrograms))[1]) == total_nber_audio_files:
            FileScaleSPECGENE_status = 'DONE'
        elif len(next(os.walk(path_output_spectrograms))[1]) == 0:
            FileScaleSPECGENE_status = 'WAITING'
        else:
            FileScaleSPECGENE_status = 'ONGOING'    

        print(FileScaleSPECGENE_status +' (',len(next(os.walk(path_output_spectrograms))[1]),'/',str(total_nber_audio_files),')' + ' -> Spectrogram Generation' )    
                
        


def next_power_of_2(x):
    return 1 if x == 0 else 2**(math.ceil(math.log2(x)))

def params_recommendation(analysis_fs,fileScale_timeResolution,fileScale_frequencyResolution):
            
    nfft = next_power_of_2(analysis_fs / fileScale_frequencyResolution)
    
    winsize= nfft
    tr = winsize / analysis_fs
    overlap = 0
    ct=0
    while tr > fileScale_timeResolution:
        if tr / fileScale_timeResolution > 10:
            winsize = winsize//8
            nfft = nfft//4
            tr = winsize / analysis_fs
        elif tr / fileScale_timeResolution > 5:
            winsize = winsize//2
            tr = winsize / analysis_fs
        else:
            overlap = min([90,round(100-( fileScale_timeResolution / tr)*100)])
            tr = tr * (100-overlap)/100
        ct+=1
        if ct>100:
            break
    
    return nfft,winsize,overlap


def display_metadata(dataset_ID):
    
    ## initialize some metadata as global variables    
    global total_nber_audio_files, orig_fs, audioFile_duration
    metadata = pd.read_csv( os.path.join(path_osmose_dataset , dataset_ID , 'raw/metadata.csv') )
    orig_fs = metadata['orig_fs'][0]
    total_nber_audio_files = metadata['nberWavFiles'][0]        
    audioFile_duration = metadata['audioFile_duration'][0]

    print('Original audio file duration (mins) :',round(audioFile_duration/60,2))
    print('Original sample frequency (Hz) :',orig_fs)    
    print('Total number of files:',total_nber_audio_files)    
    