

import os, glob
import pandas as pd
import shutil

import numpy as np

import subprocess

from ipywidgets import interact, interact_manual
from IPython.display import Image

from config import Config

import sys

path_osmose_home = "/home/datawork-osmose/"
path_osmose_dataset = "/home/datawork-osmose/dataset/"

with open('path_osmose_dataset.txt') as f:
    path_osmose_dataset = f.readlines()[0]

size_batch = 500
        

    
def check_segmenteduration(analysis_fs,segment_duration):

    sol = False

    if 'S' not in segment_duration:
        print('SORRY but your smallest time scale must be in seconds')
        sol= True
    else: 
        segment_duration = int(segment_duration[:-1])
        
    
    if analysis_fs < 800:

        if (segment_duration>200) | (segment_duration<30):
            
            print('SORRY but with your analysis frequency you have to choose a minimal time resolution between 30 and 200 s')
            sol = True

    elif analysis_fs < 10000:

        if (segment_duration>100) | (segment_duration<10):
            print('SORRY but with your analysis frequency you have to choose a minimal time resolution between 10 and 100 s')
            sol = True

    elif analysis_fs < 40000:

        if (segment_duration>30) | (segment_duration<5):
            print('SORRY but with your analysis frequency you have to choose a minimal time resolution between 5 and 30 s')
            sol = True

    else:
        if (segment_duration>10) | (segment_duration<1):
            print('SORRY but with your analysis frequency you have to choose a minimal time resolution between 1 and 10 s')
            sol = True

     
    return sol, segment_duration


def launch(config):

    datawork_base_path = "/home/datawork-osmose/FeatureEngine/"


    list_sev = []
    on_nber_exec_per_node = 0
    on_segment_duration = 0
    on_n_nodes = 0
    on_window_size = 0
    on_window_overlap = 0

    if type(config.nber_exec_per_node) == list:
        list_sev = config.nber_exec_per_node
        on_nber_exec_per_node = 1
    elif type(config.segment_duration) == list:
        list_sev = config.segment_duration
        on_segment_duration = 1
    elif type(config.n_nodes) == list:
        list_sev = config.n_nodes
        on_n_nodes = 1
    elif type(config.window_size) == list:
        list_sev = config.window_size
        on_window_size = 1
    elif type(config.window_overlap) == list:
        list_sev = config.window_overlap
        on_window_overlap = 1

    for ll in list_sev:

        if on_nber_exec_per_node:
            config.nber_exec_per_node = ll
        elif on_segment_duration:
            config.segment_duration = ll
        elif on_n_nodes:
            config.n_nodes = ll
        elif on_window_size:
            config.window_size = ll
            config.nfft = ll
        elif on_window_overlap:
            config.window_overlap = ll

        config_js = config.as_json
        with open(config.job_config_file_location, "w") as job_file:
            job_file.write(config_js)

        with open(datawork_base_path + "jobLauncher/template.pbs", "r") as template:
            template_lines = template.readlines()

        with open(config.pbs_file_location, "w") as pbs_file:
            for line in template_lines:
                pbs_file.write(
                    line.replace("JOBNAME", 'compute') \
                        .replace("N_NODES", str(config.n_nodes)) \
                        .replace("PATHOUTPUT", config.path_analysisFolder + '/ongoing_pbsFiles/') \
                        .replace("NUM_EXEC", str(config.nber_exec_per_node * config.n_nodes)) \
                        .replace("JAR_LOCATION", config.jar_location) \
                        .replace("JOB_CONFIG_FILE_LOCATION", config.job_config_file_location)
                )

        r = os.system("qsub " + config.pbs_file_location)

    if not list_sev:

        config_js = config.as_json

        with open(config.job_config_file_location, "w") as job_file:
            job_file.write(config_js)

        with open(datawork_base_path + "jobLauncher/template.pbs", "r") as template:
            template_lines = template.readlines()

        with open(config.pbs_file_location, "w") as pbs_file:
            for line in template_lines:
                pbs_file.write(
                    line.replace("JOBNAME", 'compute') \
                        .replace("N_NODES", str(config.n_nodes)) \
                        .replace("PATHOUTPUT",
                                 config.path_analysisFolder + '/ongoing_pbsFiles/log_computeFeaturesDatasetScale.txt') \
                        .replace("NUM_EXEC", str(config.nber_exec_per_node * config.n_nodes)) \
                        .replace("JAR_LOCATION", config.jar_location) \
                        .replace("JOB_CONFIG_FILE_LOCATION", config.job_config_file_location)
                )




def main(dataset_ID,analysis_fs,datasetScale_timeResoAggregation,aux_variable,aux_file,plot_LTAS,plot_EPD,plot_timeSPL,plot_recurBOX,normalizeByMax_welch,delete_featuresLS,warp_timePeriod,fmin,fmax,sequential_timePeriod,bigwarp_timePeriod_recurBox,smallwarp_timePeriod_recurBox):
        
    segment_duration = datasetScale_timeResoAggregation[0]
#     datasetScale_timeResoAggregation = datasetScale_timeResoAggregation[1:]
    
    datasetScale_nfft = 4096
    
#     datasetScale_timeResoAggregation.append(str(segment_duration)+'S')
           
    path_log = os.path.join(path_osmose_home, 'log')

    ## get unique analID
    if not os.path.exists(path_log):
        os.makedirs(path_log)

    ll = len(sorted(glob.glob(os.path.join(path_log, 'analysis_fiche*'))))
    if ll < 10:
        analID = '0000' + str(ll)
    elif ll < 100:
        analID = '000' + str(ll)
    elif ll < 1000:
        analID = '00' + str(ll)
    elif ll < 10000:
        analID = '0' + str(ll)
    else:
        analID = str(ll)
             
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
    path_output_datasetScaleFeatures =  os.path.join(path_analysisFolder, 'datasetScale_features', str(int(analysis_fs)) )
        

    global output_path_audio_files   
    output_path_audio_files = os.path.join(path_osmose_dataset, dataset_ID, 'raw','audio', str(int(orig_fileDuration))+'_'+str(int(analysis_fs))  )

    
    if len(glob.glob( os.path.join(path_pbsFiles,'pbs_*')  ))>0:
            print('SORRY but there are ongoing analysis on this dataset and we currently do not support concurrent analysis on a same dataset')
            sys.exit()       
    
    
    # do this kind of checking before laucnhing any job
    if (not os.path.exists(path_output_datasetScaleFeatures)) or (delete_featuresLS):
        
        sol, segment_duration = check_segmenteduration(analysis_fs,segment_duration)        
        if sol:
            sys.exit()
        
        
           
    
    if not os.path.exists(path_pbsFiles):
        os.makedirs(path_pbsFiles)     
    
    # dummy regle de trois to compute analysis stat from orig ones..
    anal_totalVolume = metadata['orig_totalVolume'][0]  * (analysis_fs / orig_fs)
    anal_totalDurationMins = metadata['orig_totalDurationMins'][0] * (analysis_fs / orig_fs)
            
    # write analysis.csv and make it global
    global analysis_fiche
    data = {'dataset_ID' :dataset_ID,'analysis_fs' :float(analysis_fs),'aux_variable' : aux_variable, 'datasetScale_nfft' : datasetScale_nfft , 'newFeatures_timescale':datasetScale_timeResoAggregation , 'normalizeByMax_welch':normalizeByMax_welch, 'plot_LTAS':plot_LTAS, 'plot_recurBOX':plot_recurBOX,'plot_timeSPL':plot_timeSPL,'plot_EPD':plot_EPD,'dataset_Volume':anal_totalVolume,'dataset_totalDurationMins':anal_totalDurationMins,'total_nber_audio_files':total_nber_audio_files,'segment_duration':segment_duration,'warp_timePeriod':warp_timePeriod,'fmin':fmin,'fmax':fmax,'sequential_timePeriod':sequential_timePeriod,'bigwarp_timePeriod_recurBox':bigwarp_timePeriod_recurBox,'smallwarp_timePeriod_recurBox':smallwarp_timePeriod_recurBox}
    
    
    
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
        
        jobID_resample=[]
        for job in list_pbs_resample:    
            res = subprocess.run(['qsub', job], stdout=subprocess.PIPE).stdout.decode('utf-8').rstrip('\n')   
            jobID_resample.append(res)
                                    
        # set permissions to /analysis
        createQsub_setPermissions(dataset_ID,os.path.join(path_osmose_dataset, dataset_ID),str(int(orig_fileDuration))+'_'+str(int(analysis_fs)))
        
        res = subprocess.run( ['qsub','-W depend=afterok:'+ (':').join(jobID_resample) ,os.path.join(path_pbsFiles, 'pbs_setPermissions_0.pbs')],stdout=subprocess.PIPE).stdout.decode('utf-8')     
        jobID_setPermissionResample=[]
        jobID_setPermissionResample.append(res)
        
    else:        
        jobID_resample = ['']
        jobID_setPermissionResample = ['']
   
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
        config.segment_duration = segment_duration

        config.n_nodes = 4#2#4
        config.nber_exec_per_node = 9#14
        config.sound_sampling_rate_target = analysis_fs  
        config.high_freq_tol = 0.5 * analysis_fs

        config.pbs_timestampJsonCreation = path_analysisFolder + '/ongoing_pbsFiles/pbs_timestampJsonCreation.pbs'

#         createQsub_timestampJsonCreation(dataset_ID,analysis_fs)
        
        launch(config)    
        
        res = subprocess.run( ['qsub','-W depend=afterok:'+ (':').join(jobID_resample) ,os.path.join(path_pbsFiles, 'pbs_computeFeaturesDatasetScale.pbs')],stdout=subprocess.PIPE).stdout.decode('utf-8')            
        jobID_computeFeaturesDatasetScale=[]
        jobID_computeFeaturesDatasetScale.append(res)
    
    
#         res = subprocess.run( ['qsub','-W depend=afterok:'+ jobID_computeFeaturesDatasetScale[0] ,os.path.join(path_pbsFiles, 'pbs_timestampJsonCreation.pbs')],stdout=subprocess.PIPE).stdout.decode('utf-8')           
#         jobID_timestampJsonCreation=[]
#         jobID_timestampJsonCreation.append(res)
                
    else:
        jobID_computeFeaturesDatasetScale=['']
#         jobID_timestampJsonCreation=['']

        
        

    ##########

    global path_output_soundscape_results

    path_output_soundscape_results =  os.path.join(path_analysisFolder, 'soundscape' )
    
    if os.path.exists(path_output_soundscape_results):
        shutil.rmtree(path_output_soundscape_results)

    if plot_LTAS and (not os.path.exists(os.path.join(path_output_soundscape_results,'LTAS'))):
        os.makedirs(os.path.join(path_output_soundscape_results,'LTAS')) 
    if plot_EPD and (not os.path.exists(os.path.join(path_output_soundscape_results,'EPD'))):
        os.makedirs(os.path.join(path_output_soundscape_results,'EPD'))
    if plot_timeSPL and (not os.path.exists(os.path.join(path_output_soundscape_results,'timeSPL'))):
        os.makedirs(os.path.join(path_output_soundscape_results,'timeSPL'))            
    if plot_recurBOX and (not os.path.exists(os.path.join(path_output_soundscape_results,'recurBOX'))):
        os.makedirs(os.path.join(path_output_soundscape_results,'recurBOX')) 
                           

    for dddd in datasetScale_timeResoAggregation:
        path_newFeatures = os.path.join(path_output_soundscape_results,'raw_welch',dddd)
        
        if os.path.exists(path_newFeatures):
            shutil.rmtree(path_newFeatures)
        os.makedirs(path_newFeatures)                
            
            

    ##########
    
    if plot_LTAS+plot_EPD+plot_timeSPL+plot_recurBOX:

        main_buildNewFeatures( dataset_ID,analysis_fs)    
        createQsub_reaggregateGetFeatures( dataset_ID,analysis_fs,0)
        
        jobID_buildNewFeatures = []
        for job in sorted(glob.glob(os.path.join(path_pbsFiles, 'pbs_buildNewFeatures*'))):            
            res_buildNewFeatures = subprocess.run( ['qsub','-W depend=afterok:'+ jobID_computeFeaturesDatasetScale[0],job],stdout=subprocess.PIPE).stdout.decode('utf-8').rstrip('\n')           
            jobID_buildNewFeatures.append(res_buildNewFeatures)

        res = subprocess.run(['qsub', '-W depend=afterok:' + (':').join(jobID_buildNewFeatures),os.path.join(path_pbsFiles, 'pbs_reaggregateGetFeatures_0.pbs')],stdout=subprocess.PIPE).stdout.decode('utf-8')
        jobID_reaggregateFeat = []
        jobID_reaggregateFeat.append(res)
      
        
    else:
        jobID_reaggregateFeat = ['']
        jobID_buildNewFeatures = ['']
        

        
    # create the log file on the side of analysis , ie job_id with analysis paremeters
    data = {'analID':analID,'jobID_resample':[ll.split('.')[0] for ll in jobID_resample],'jobID_setPermissionResample':[ll.split('.')[0] for ll in jobID_setPermissionResample],'jobID_computeFeaturesDatasetScale':[ll.split('.')[0] for ll in jobID_computeFeaturesDatasetScale],'jobID_buildNewFeatures' :[ll.split('.')[0] for ll in jobID_buildNewFeatures],'jobID_reaggregateFeat':[ll.split('.')[0] for ll in jobID_reaggregateFeat]}       
    df = pd.DataFrame.from_records([data])
    df.to_csv( os.path.join(path_log,'jobID_'+analID+'.csv') , index=False)      
    analysis_fiche.to_csv( os.path.join(path_log,'analysis_fiche_'+analID+'.csv'))
                                            
    
    

    
    
    
    
    
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
                    .replace("path_to_set",path_to_set)\
                    .replace("folderName_audioFiles",folderName_audioFiles)                      
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
    
    with open(path_osmose_home+"osmoseNotebooks_v0/source/templateQsub_resample.pbs", "r") as template:
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

     
            
            
            


            
def createQsub_reaggregateGetFeatures(dataset_ID,analysis_fs,id_job):

    cur_pbs_file = os.path.join(path_pbsFiles, 'pbs_reaggregateGetFeatures_0.pbs')
    logjob_outpath = os.path.join(path_pbsFiles, 'log_reaggregateGetFeatures_0.txt')

    with open( os.path.join(path_osmose_home , "osmoseNotebooks_v0/source/templateQsub_reaggregateGetFeatures.pbs"), "r") as template:
        template_lines = template.readlines()

    with open(cur_pbs_file, "w") as pbs_file:
        for line in template_lines:
            pbs_file.write(
                line.replace("dataset_ID", dataset_ID) \
                    .replace("analysis_fs", str(int(analysis_fs))) \
                    .replace("logjob_outpath", logjob_outpath)
            )
            
            
            
def main_buildNewFeatures(dataset_ID,analysis_fs):
                
    size_batch = 8 # this corresponds to the number of json files processed in qsub_buildNewFeatures
    
    # we do not know total_nber_JSON as compute has not been done yet ! so we assume a high number to be sure , and job without data will be automatically killed, but i know this is shitty stuff hmm .. 
    total_nber_JSON = 300
    
    for ind in np.arange(0, total_nber_JSON, size_batch):
        if ind == np.arange(0, total_nber_JSON, size_batch)[-1]:# deal with last batch
            ind_max = total_nber_JSON
        else:
            ind_max = ind + size_batch

        ind_min = ind
        createQsub_buildNewFeatures(dataset_ID,analysis_fs,ind_min, ind_max)
        

def createQsub_buildNewFeatures(dataset_ID,analysis_fs, ind_min, ind_max):
    
    cur_pbs_file = os.path.join(path_pbsFiles, 'pbs_buildNewFeatures_' + str(ind_min) + '.pbs')
    logjob_outpath = os.path.join(path_pbsFiles, 'log_buildNewFeatures_' + str(ind_min) + '.txt')

    with open( os.path.join(path_osmose_home , "osmoseNotebooks_v0/source/templateQsub_buildNewFeatures.pbs"), "r") as template:
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
        
        
    # convert_json_to_pkl
    if len( glob.glob(os.path.join(path_pbsFiles, 'pbs_convert_json_to_pkl*')) ):        

        if compute_status=='DONE':
            status = 'ONGOING'
        else:
            status = 'WAITING'            
            
        jobname = 'Generate pkl files with minimal resolution in: ' + path_newFeatures
           
#         print( status +' (', ll, '/',str(total_nber_audio_files), ')' + ' = '+   jobname  )   
        print('o ' + status + ' -> ' +   jobname  )  
    
    
    
        
        
        
    # Generate Spectrograms
    if len( glob.glob(os.path.join(path_pbsFiles, 'pbs_spectroGenerationDatasetScale*')) ):
        
#         ll = len(glob.glob(os.path.join(path_output_soundscape_results, '*.png')))
        
#         if ll == total_nber_audio_files:
#             status = 'DONE'
#         else:

        if compute_status=='DONE':
            status = 'ONGOING'
        else:
            status = 'WAITING'            
            
        jobname = 'Generate dataset scale spectrograms in: ' + path_output_soundscape_results
           
#         print( status +' (', ll, '/',str(total_nber_audio_files), ')' + ' = '+   jobname  )   
        print('o ' + status + ' -> ' +   jobname  )  
        
        
    # Re-aggreagate new features
    if len( glob.glob(os.path.join(path_pbsFiles, 'pbs_buildNewFeatures*')) ):
                
        if compute_status=='DONE':
            status = 'ONGOING'
        else:
            status = 'WAITING'         


        jobname = 'Compute welch spectra'# + os.path.join(path_analysisFolder, 'getFeatures', str(int(analysis_fs)))
           
#         print( status +' (', len(glob.glob(os.path.join(path_newFeatures, '*.pkl'))) ,')' + ' = '+   jobname  )  
        print('o ' +  status + ' -> ' +   jobname  )  
        
        
    print('\n\n just so you know you have',len( glob.glob(os.path.join(path_pbsFiles, 'pbs_*')) ),'ongoing jobs working for you! thank you M. DATARMOR!')
        

def count_nber_json(path,ext):
    
    # wait compute_features is donepath_newFeatures = os.path.join(path_analysisFolder, 'getFeatures', str(int(analysis_fs)))    

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

def visu_adjustSpectros(dataset_ID , analysis_fs , datasetScale_maxDisplaySpectro,soundscapeType_im):
    
    path_analysisFolder = os.path.join(path_osmose_dataset, dataset_ID, 'analysis')

    ## initialize some paths as global variables
    path_output_spectrograms = os.path.join(path_analysisFolder, 'soundscape',datasetScale_maxDisplaySpectro+'_'+str(int(analysis_fs)) , soundscapeType_im)    
    
    
    if os.path.exists(path_output_spectrograms):
    
        if len(os.listdir(path_output_spectrograms))==0:
            print('Spectrograms not ready yet, please wait a bit..')
            return

        @interact
        def show_images(file=np.sort(os.listdir(path_output_spectrograms))):
            display(Image(os.path.join(path_output_spectrograms,file)))
            Image.height=12000
            Image.width=12000   
        
    else:
        
        print('No LTAS to display')
        
        
        
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
    print('Total volume (GB):',metadata['orig_totalVolume'][0])      
    
    list_aux = glob.glob(os.path.join(path_osmose_dataset , dataset_ID , 'raw/auxiliary/*csv'))
    print('Auxiliary files :',[os.path.basename(ll) for ll in list_aux])      
    
    ll = np.sort(next(os.walk( os.path.join(path_osmose_dataset , dataset_ID , 'raw/audio/') ))[1])
    
    newll = []
    for el in ll:
        if not str(int(orig_fileDuration))+'_'+str(int(orig_fs)) == el:
            newll.append(el)
    
    print('***************************')      
    print('Existing analysis paramaters (fileDuration_sampleFrequency) :', newll)    
        
    