import sys
import numpy as np
import os
import glob
import pickle
import shutil
import pandas as pd
import subprocess
import math  

from datetime import datetime
from aux_functions import custom_date_range

from ipywidgets import interact, interact_manual
from IPython.display import Image

from aux import *
from config import Config

import sys

path_osmose_home = "/home/datawork-osmose/"
path_osmose_dataset = "/home/datawork-osmose/dataset/"

size_batch = 500
        

    
def main(dataset_ID,analysis_fs,datasetScale_maxDisplaySpectro,datasetScale_timeResoAggregation,aux_variable,aux_file,spectroGO,newFeaturesGO,normalizeByMax_welch,delete_featuresLS):
    
    datasetScale_nfft = 4096
           
    ## initialize some metadata as global variables    
    global total_nber_audio_files, orig_fs, orig_fileDuration
    metadata = pd.read_csv( os.path.join(path_osmose_dataset , dataset_ID , 'raw/metadata.csv') )
    orig_fs = metadata['orig_fs'][0]
    total_nber_audio_files = metadata['nberWavFiles'][0]        
    orig_fileDuration = metadata['orig_fileDuration'][0]       
    
    # build a few paths
    global path_analysisFolder, path_pbsFiles, path_output_spectrograms, path_output_datasetScaleFeatures, path_newFeatures
    path_analysisFolder = os.path.join(path_osmose_dataset, dataset_ID, 'analysis')
    path_pbsFiles = os.path.join(path_analysisFolder, 'ongoing_pbsFiles')    
    path_newFeatures = os.path.join(path_analysisFolder, 'getFeatures', str(int(analysis_fs)),datasetScale_timeResoAggregation)
    path_output_datasetScaleFeatures =  os.path.join(path_analysisFolder, 'datasetScale_features', str(int(analysis_fs)) )
       
    global output_path_audio_files   
    
#     if debug_specFolder != 0:
#         output_path_audio_files = os.path.join(path_osmose_dataset, dataset_ID, 'raw','audio', debug_specFolder+'_'+str(int(analysis_fs))  )    
#     else:
    output_path_audio_files = os.path.join(path_osmose_dataset, dataset_ID, 'raw','audio', str(int(orig_fileDuration))+'_'+str(int(analysis_fs))  )
        
        
    ## initialize some paths as global variables
    path_output_spectrograms = os.path.join(path_analysisFolder, 'spectrograms',datasetScale_maxDisplaySpectro+'_'+str(int(analysis_fs)) )
    
    if not os.path.exists(path_pbsFiles):
        os.makedirs(path_pbsFiles)     
    if not os.path.exists(path_output_spectrograms):
        os.makedirs(path_output_spectrograms)                
    
        

    # write analysis.csv and make it global
    global analysis_fiche
    data = {'dataset_ID' :dataset_ID,'analysis_fs' :float(analysis_fs),'datasetScale_maxDisplaySpectro' : datasetScale_maxDisplaySpectro,'aux_variable' : aux_variable, 'datasetScale_nfft' : datasetScale_nfft , 'newFeatures_timescale':datasetScale_timeResoAggregation , 'normalizeByMax_welch':normalizeByMax_welch}
    analysis_fiche = pd.DataFrame.from_records([data])
    analysis_fiche.to_csv( os.path.join(path_analysisFolder,'analysis_fiche.csv') )
    
        
    # create all pbs files in /home/datawork-osmose/dataset/<dataset_ID>/analysis/ongoing_pbsFiles/
    if not os.path.exists( output_path_audio_files ):        
        os.makedirs(output_path_audio_files)               
        main_resample(dataset_ID,analysis_fs,str(int(orig_fileDuration))+'_'+str(int(orig_fs)),str(int(orig_fileDuration))+'_'+str(int(analysis_fs)))        

        
        if int(analysis_fs) != int(orig_fs):

            from shutil import copyfile
            src = os.path.join(path_osmose_dataset, dataset_ID, 'raw','audio',\
                                       str(int(orig_fileDuration))+'_'+str(int(orig_fs)), 'timestamp.csv')
            dst = os.path.join(output_path_audio_files, 'timestamp.csv')        
            copyfile(src, dst)
                
        list_pbs_resample = sorted( glob.glob(os.path.join(path_pbsFiles, 'pbs_resample*')) )
        job_resample = []
        for job in list_pbs_resample:    
            res = subprocess.run(['qsub', job], stdout=subprocess.PIPE).stdout.decode('utf-8').rstrip('\n')   
            job_resample.append(res)
                    
                
        # set permissions to /analysis
        path_to_set = os.path.join(path_osmose_dataset, dataset_ID, 'raw/audio',str(int(orig_fileDuration))+'_'+str(int(analysis_fs)))
        createQsub_setPermissions(dataset_ID,path_to_set)

        subprocess.run( ['qsub','-W depend=afterok:'+ (':').join(job_resample) ,os.path.join(path_pbsFiles, 'pbs_setPermissions_0.pbs')],stdout=subprocess.PIPE).stdout.decode('utf-8')            

        
    else:        
        job_resample = ''
   
    if (not os.path.exists(path_output_datasetScaleFeatures)) or (delete_featuresLS):
        
        if (os.path.exists(path_output_datasetScaleFeatures)) and (delete_featuresLS):
            shutil.rmtree(path_output_datasetScaleFeatures)
        os.makedirs(path_output_datasetScaleFeatures)

        config = Config()
        config.path_analysisFolder = path_analysisFolder
        config.sound_sampling_rate = int(metadata['orig_fs'][0])
        config.dataset_id = dataset_ID
        config.aux_file = aux_file      
        config.window_size = datasetScale_nfft
        config.nfft = datasetScale_nfft

        if analysis_fs < 800:
            config.segment_duration = 120
        elif analysis_fs < 10000:
            config.segment_duration = 60
        elif analysis_fs < 40000:
            config.segment_duration = 30            
        else:
            config.segment_duration = 1               

    #         if config.dataset_id=="SPMAuralB":
    #             config.segment_duration = 60
    #         elif config.dataset_id=="SPMAuralA":
    #             config.segment_duration = 60            
    #         elif config.dataset_id=="ohasisbio2015wker":
    #             config.segment_duration =120
    #         elif config.dataset_id=="chagos":
    #             config.segment_duration =60
    #         elif config.dataset_id=="azoresIfremer":
    #             config.segment_duration = 120
    #         elif config.dataset_id=="fromveur256000":
    #             config.segment_duration =1
    #         elif config.dataset_id=="synthetic":
    #             config.segment_duration =60
    #         elif config.dataset_id=="argoLOV":
    #             config.segment_duration =60     
    #         elif config.dataset_id=="DCLDE2020HFcinms17b":        
    #             config.segment_duration =1            
    #         elif config.dataset_id=="GliderSPAms":        
    #             config.segment_duration =60  
    #         elif config.dataset_id=="GliderWalterShoals":        
    #             config.segment_duration =60  
    #         elif config.dataset_id=="gliderWHOI":        
    #             config.segment_duration =60
    #         elif config.dataset_id=="ml17_280a":        
    #             config.segment_duration =60
    #         elif config.dataset_id=="ml18_294b":        
    #             config.segment_duration =60

        config.n_nodes = 4#2#4
        config.nber_exec_per_node = 9#14
        config.sound_sampling_rate_target = analysis_fs  
        config.high_freq_tol = 0.5 * analysis_fs

        config.pbs_timestampJsonCreation = path_analysisFolder + '/ongoing_pbsFiles/pbs_timestampJsonCreation.pbs'

        createQsub_timestampJsonCreation(dataset_ID,analysis_fs)
        createQsub_buildPKLlargescale(dataset_ID,analysis_fs)

        launch(config)    
        

        job_computeFeaturesDatasetScale = subprocess.run( ['qsub','-W depend=afterok:'+ (':').join(job_resample) ,os.path.join(path_pbsFiles, 'pbs_computeFeaturesDatasetScale.pbs')],stdout=subprocess.PIPE).stdout.decode('utf-8')            

        job_timestampJsonCreation = subprocess.run( ['qsub','-W depend=afterok:'+job_computeFeaturesDatasetScale,os.path.join(path_pbsFiles, 'pbs_timestampJsonCreation.pbs')],stdout=subprocess.PIPE).stdout.decode('utf-8')

#         job_buildPKLlargescale = subprocess.run( ['qsub','-W depend=afterok:'+job_computeFeaturesDatasetScale,os.path.join(path_pbsFiles, 'pbs_buildPKLlargescale.pbs')],stdout=subprocess.PIPE).stdout.decode('utf-8')

        
    else:
        job_timestampJsonCreation = ''        
#         job_buildPKLlargescale = ''        

    
    ##########

    global path_output_spectrogramsDatasetScale

#         path_output_spectrogramsDatasetScale =  os.path.join(path_osmose_dataset, dataset_ID, 'analysis', 'spectrograms', str(int(analysis_fs)) , 'datasetScale' , analysis_fiche['datasetScale_maxDisplaySpectro'][0] )
    path_output_spectrogramsDatasetScale =  path_output_spectrograms # os.path.join(path_osmose_dataset, dataset_ID, 'analysis', 'spectrograms', str(int(orig_fileDuration))+'_'+str(int(analysis_fs))  )
    
    
    if spectroGO:
        
        if os.path.exists(path_output_spectrogramsDatasetScale):
            shutil.rmtree(path_output_spectrogramsDatasetScale)
        os.makedirs(path_output_spectrogramsDatasetScale)

        main_spectroGenerationDatasetScale(dataset_ID,analysis_fs)

        for job in sorted(glob.glob(os.path.join(path_pbsFiles, 'pbs_spectroGenerationDatasetScale*') )):    
#             res_spectro = subprocess.run( ['qsub','-W depend=afterok:'+ (':').join([job_timestampJsonCreation , job_buildPKLlargescale]),job],stdout=subprocess.PIPE).stdout.decode('utf-8')
            res_spectro = subprocess.run( ['qsub','-W depend=afterok:'+ job_timestampJsonCreation,job],stdout=subprocess.PIPE).stdout.decode('utf-8')            
            
            
            
    else:
        res_spectro=''
        
        
    ##########
    
    if newFeaturesGO:
    
        path_newFeatures = os.path.join(path_analysisFolder, 'getFeatures', str(int(analysis_fs)),datasetScale_timeResoAggregation)    
        
        # re-initialize the folder of spectrograms
        if os.path.exists(path_newFeatures):
            shutil.rmtree(path_newFeatures)
        os.makedirs(path_newFeatures)

        createQsub_buildNewFeatures( dataset_ID,analysis_fs,0,0, 0)    
        createQsub_reaggregateGetFeatures( dataset_ID,analysis_fs,0)

        job_buildNewFeatures = subprocess.run( ['qsub','-W depend=afterok:'+ (':').join([job_timestampJsonCreation , job_buildPKLlargescale]),os.path.join(path_pbsFiles, 'pbs_buildNewFeatures_0.pbs')],stdout=subprocess.PIPE).stdout.decode('utf-8')
        
        res_newFeatures = subprocess.run( ['qsub','-W depend=afterok:'+job_buildNewFeatures,os.path.join(path_pbsFiles, 'pbs_reaggregateGetFeatures_0.pbs')],stdout=subprocess.PIPE).stdout.decode('utf-8')
    else:
        res_newFeatures=''
        

    # set permissions to /analysis
    path_to_set = os.path.join(path_osmose_dataset, dataset_ID, 'analysis')
    createQsub_setPermissions(dataset_ID,path_to_set)
    
    subprocess.run( ['qsub','-W depend=afterok:'+res_newFeatures+':'+res_spectro,os.path.join(path_pbsFiles, 'pbs_setPermissions_0.pbs')],stdout=subprocess.PIPE).stdout.decode('utf-8')


    
    
    if isinstance(res_spectro,list):       
        subprocess.run( ['qsub','-W depend=afterok:'+(':').join(res_spectro)+':'+res_newFeatures,os.path.join(path_pbsFiles, 'pbs_setPermissions_0.pbs')],stdout=subprocess.PIPE).stdout.decode('utf-8')        
    else:
        subprocess.run( ['qsub','-W depend=afterok:'+res_newFeatures+':'+res_spectro,os.path.join(path_pbsFiles, 'pbs_setPermissions_0.pbs')],stdout=subprocess.PIPE).stdout.decode('utf-8')
            
            
    
    
    
    
    
def createQsub_setPermissions(dataset_ID,path_to_set):

    cur_pbs_file = os.path.join(path_pbsFiles, 'pbs_setPermissions_0.pbs')
    logjob_outpath = os.path.join(path_pbsFiles, 'log_setPermissions_0.txt')
    
    with open(path_osmose_home+"notebook_source/templateQsub_setPermissions.pbs", "r") as template:
        template_lines = template.readlines()
    
    with open(cur_pbs_file, "w") as pbs_file:
        for line in template_lines:
            pbs_file.write(
                line.replace("dataset_ID", dataset_ID)\
                    .replace("logjob_outpath",logjob_outpath)\
                    .replace("path_to_set",path_to_set)                
            )

     

    
    
    
## Resample        
        
def main_resample(dataset_ID,analysis_fs,folderIn,folderOut):

    
    # use multiple nodes with batch of size_batch files ; use it if number of files higher than 1000
    id_job=0 # used to identify the different jobs created in case total_nber_audio_files > size_batch
    if total_nber_audio_files<size_batch:
        createQsub_resample(dataset_ID,analysis_fs,id_job,0,total_nber_audio_files,folderIn,folderOut)
    else:    
        for ind in np.arange(0, total_nber_audio_files, size_batch):
            if ind == np.arange(0, total_nber_audio_files, size_batch)[-1]:
                ind_max = total_nber_audio_files
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
                    .replace("new_audioFileDuration",str(int(orig_fileDuration)) )\
                    .replace("orig_audioFileDuration",str(int(orig_fileDuration)) )
            )

     

    
def createQsub_timestampJsonCreation(dataset_ID,analysis_fs):

    cur_pbs_file = os.path.join(path_pbsFiles, 'pbs_timestampJsonCreation.pbs')
    logjob_outpath = os.path.join(path_pbsFiles, 'log_timestampJsonCreation.txt')

    with open(path_osmose_home+"notebook_source/templateQsub_timestampJsonCreation.pbs", "r") as template:
        template_lines = template.readlines()
    
    with open(cur_pbs_file, "w") as pbs_file:
        for line in template_lines:
            pbs_file.write(
                line.replace("dataset_ID", dataset_ID)\
                    .replace("analysis_fs", str(int(analysis_fs)))\
                    .replace("logjob_outpath",logjob_outpath) 
            )
                

def createQsub_buildPKLlargescale(dataset_ID,analysis_fs):

    cur_pbs_file = os.path.join(path_pbsFiles, 'pbs_buildPKLlargescale.pbs')
    logjob_outpath = os.path.join(path_pbsFiles, 'log_buildPKLlargescale.txt')

    with open(path_osmose_home+"notebook_source/templateQsub_buildPKLlargescale.pbs", "r") as template:
        template_lines = template.readlines()
    
    with open(cur_pbs_file, "w") as pbs_file:
        for line in template_lines:
            pbs_file.write(
                line.replace("dataset_ID", dataset_ID)\
                    .replace("analysis_fs", str(int(analysis_fs)))\
                    .replace("logjob_outpath",logjob_outpath) 
            )
                
    
    
    
    
## spectroGenerationDatasetScale
        
def main_spectroGenerationDatasetScale(dataset_ID,analysis_fs):
                          
    
    df = pd.read_csv(os.path.join(output_path_audio_files, 'timestamp.csv'), header=None)
    
    if str(analysis_fiche['datasetScale_maxDisplaySpectro'][0]) != 'all':
        time_periods = custom_date_range(df[1][0], df[1][len(df[1]) - 1], analysis_fiche['datasetScale_maxDisplaySpectro'][0])
    else:
        time_periods = [ df[1][0] , df[1][len(df[1]) - 1] ]
        
    cur_total_nber_audio_files = len(time_periods)    
    
    
    total_nber_JSON = 0
    for base, dirs, files in os.walk( os.path.join(path_osmose_dataset, dataset_ID, 'analysis/datasetScale_features',str(int(analysis_fs))) ):
        for Files in files:
            if Files.lower().endswith('.json'):
                total_nber_JSON += 1    
    
    if total_nber_JSON / cur_total_nber_audio_files > 30:
       
        print('SORRY we have to stop here : your spectrogram time display window',analysis_fiche['datasetScale_maxDisplaySpectro'][0],'is too big for us, you will have to reduce it ..')

        print('so you know , the following ratio:',total_nber_JSON / cur_total_nber_audio_files,' must be inferior to 30')

        sys.exit()
        
        
    
    size_batch_spectroGene = 100 
    
    # use multiple nodes with batch of size_batch files ; use it if number of files higher than 1000
    id_job = 0  # used to identify the different jobs created in case cur_total_nber_audio_files > size_batch
    if cur_total_nber_audio_files < size_batch_spectroGene:
        createQsub_spectroGenerationDatasetScale(dataset_ID,analysis_fs,id_job, 0, cur_total_nber_audio_files)
    else:
        size_batch_spectroGene = np.floor(cur_total_nber_audio_files / 3)
        
        for ind in np.arange(0, cur_total_nber_audio_files, size_batch_spectroGene):
            
            if ind == np.arange(0, cur_total_nber_audio_files, size_batch_spectroGene)[-1]:
                ind_max = cur_total_nber_audio_files
            else:
                ind_max = ind + size_batch_spectroGene
            
            ind_min = ind

            createQsub_spectroGenerationDatasetScale(dataset_ID,analysis_fs,id_job, int(ind_min), int(ind_max))
            id_job += 1        

            
            
def createQsub_spectroGenerationDatasetScale(dataset_ID,analysis_fs,id_job, ind_min, ind_max):
        
    # build a few paths
    path_analysisFolder = os.path.join(path_osmose_dataset, dataset_ID, 'analysis')
    
    cur_pbs_file = os.path.join(path_pbsFiles, 'pbs_spectroGenerationDatasetScale_' + str(ind_min) + '.pbs')
    logjob_outpath = os.path.join(path_pbsFiles, 'log_spectroGenerationDatasetScale_' + str(ind_min) + '.txt')

    with open( os.path.join(path_osmose_home , "notebook_source/templateQsub_spectroGenerationDatasetScale.pbs"), "r") as template:
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

            
            
            

            
            
def createQsub_reaggregateGetFeatures(dataset_ID,analysis_fs,id_job):

    cur_pbs_file = os.path.join(path_pbsFiles, 'pbs_reaggregateGetFeatures_0.pbs')
    logjob_outpath = os.path.join(path_pbsFiles, 'log_reaggregateGetFeatures_0.txt')

    with open( os.path.join(path_osmose_home , "notebook_source/templateQsub_reaggregateGetFeatures.pbs"), "r") as template:
        template_lines = template.readlines()

    with open(cur_pbs_file, "w") as pbs_file:
        for line in template_lines:
            pbs_file.write(
                line.replace("dataset_ID", dataset_ID) \
                    .replace("analysis_fs", str(int(analysis_fs))) \
                    .replace("logjob_outpath", logjob_outpath)
            )

    
            
            
            
            
            
            
            
            
            
def createQsub_buildNewFeatures(dataset_ID,analysis_fs,id_job, ind_min, ind_max):
    
    # just need to estimate number of time periods here
    df = pd.read_csv(os.path.join(path_osmose_dataset, dataset_ID, 'raw/audio/timestamp.csv'), header=None)
    time_periods = custom_date_range(df[1][0], df[1][len(df[1]) - 1], '50H')   
    
#     print('for debug: number of reaggregated pkl files is:',len(time_periods))

    cur_pbs_file = os.path.join(path_pbsFiles, 'pbs_buildNewFeatures_' + str(ind_min) + '.pbs')
    logjob_outpath = os.path.join(path_pbsFiles, 'log_buildNewFeatures_' + str(ind_min) + '.txt')

    with open( os.path.join(path_osmose_home , "notebook_source/templateQsub_buildNewFeatures.pbs"), "r") as template:
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
            
        jobname = 'Resampling'
            
        print('o ' + resampling_status +' (', len(glob.glob(os.path.join(output_path_audio_files, '*.wav'))), '/',str(total_nber_audio_files), ')' + ' -> '+   jobname  )   


    if len( glob.glob(os.path.join(path_pbsFiles, 'pbs_resample*')) ):
        compute_status = 'WAITING'                       
    else:
        compute_status = 'ONGOING'        
    for root, dirs, files in os.walk(path_output_datasetScaleFeatures):  
        if '_SUCCESS' in files:
            compute_status = 'DONE'
                

    # Compute Features
    if len( glob.glob(os.path.join(path_pbsFiles, 'pbs_computeFeaturesDatasetScale*')) ):
        

                
        jobname = 'Compute dataset-scale features'
           
#         print( status +' (', count_nber_json(path_output_datasetScaleFeatures,'json'),')' + ' = '+   jobname  )  
        print('o ' +  compute_status + ' -> ' +   jobname  )  
        
        
        
    # Generate Spectrograms
    if len( glob.glob(os.path.join(path_pbsFiles, 'pbs_spectroGenerationDatasetScale*')) ):
        
#         ll = len(glob.glob(os.path.join(path_output_spectrogramsDatasetScale, '*.png')))
        
#         if ll == total_nber_audio_files:
#             status = 'DONE'
#         else:

        if compute_status=='DONE':
            status = 'ONGOING'
        else:
            status = 'WAITING'            
            
        jobname = 'Generate dataset scale spectrograms in: ' + path_output_spectrogramsDatasetScale
           
#         print( status +' (', ll, '/',str(total_nber_audio_files), ')' + ' = '+   jobname  )   
        print('o ' + status + ' -> ' +   jobname  )  
        
        
    # Re-aggreagate new features
    if len( glob.glob(os.path.join(path_pbsFiles, 'pbs_buildNewFeatures*')) ):
                
        if compute_status=='DONE':
            status = 'ONGOING'
        else:
            status = 'WAITING'         
            
        jobname = 'Re-aggreagate features in: ' + path_newFeatures
           
#         print( status +' (', len(glob.glob(os.path.join(path_newFeatures, '*.pkl'))) ,')' + ' = '+   jobname  )  
        print('o ' +  status + ' -> ' +   jobname  )  
        
        
        
        

def count_nber_json(path,ext):
    
    # wait compute_features is donepath_newFeatures = os.path.join(path_analysisFolder, 'getFeatures', str(int(analysis_fs)))    

    path_output_datasetScaleFeatures 
    json_files=[]
    for subdir, dirs, files in os.walk(path):
        if subdir.find("temporary")!=-1:# need this to avoid count temporary json files
            continue
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith("."+ext):
                json_files.append(filepath)       
    
    return len(json_files)        
        
def list_datasets(nargout=0):
    
    l_ds = sorted(os.listdir(path_osmose_dataset))
    
    if nargout == 0:
        print("Available datasets:")

        for ds in l_ds:
            print("  - {}".format(ds))

    else:
        return l_ds
    
        
        
## aux

def visu_adjustSpectros(dataset_ID , analysis_fs , datasetScale_maxDisplaySpectro):
    
    path_analysisFolder = os.path.join(path_osmose_dataset, dataset_ID, 'analysis')
    
    ## initialize some paths as global variables
    path_output_spectrograms = os.path.join(path_analysisFolder, 'spectrograms',datasetScale_maxDisplaySpectro+'_'+str(int(analysis_fs)) )    
    
    if len(os.listdir(path_output_spectrograms))==0:
        print('Spectrograms not ready yet, please wait a bit..')
        return

    @interact
    def show_images(file=np.sort(os.listdir(path_output_spectrograms))):
        display(Image(os.path.join(path_output_spectrograms,file)))
        Image.height=12000
        Image.width=12000   
        
        
        
        
        
def display_metadata(dataset_ID):
    
    ## initialize some metadata as global variables    
    global total_nber_audio_files, orig_fs, orig_fileDuration
    metadata = pd.read_csv( os.path.join(path_osmose_dataset , dataset_ID , 'raw/metadata.csv') )
    orig_fs = metadata['orig_fs'][0]
    total_nber_audio_files = metadata['nberWavFiles'][0]        
    orig_fileDuration = metadata['orig_fileDuration'][0]

    print('Original sample frequency (Hz) :',orig_fs)    
    print(metadata['start_date'][0][:16],' --> ',metadata['end_date'][0][:16])        
    print('Cumulated number of days :' , (pd.to_datetime(metadata['end_date'][0], utc=True) - pd.to_datetime(metadata['start_date'][0], utc=True)).days )                
    print('Original audio file duration (s) :',int(orig_fileDuration) )
    print('Duty cycle (%) :',round(metadata['dutyCycle_percent'][0],2))
    print('Total number of files:',total_nber_audio_files)      
    
    list_aux = glob.glob(os.path.join(path_osmose_dataset , dataset_ID , 'raw/auxiliary/*csv'))
    print('Auxiliary files :',[os.path.basename(ll) for ll in list_aux])      
    
    ll = np.sort(next(os.walk( os.path.join(path_osmose_dataset , dataset_ID , 'raw/audio/') ))[1])
    
    newll = []
    for el in ll:
        if not str(int(orig_fileDuration))+'_'+str(int(orig_fs)) == el:
            newll.append(el)
    
    print('***************************')      
    print('Existing analysis paramaters (fileDuration_sampleFrequency) :', newll)    
        
    