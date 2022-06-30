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

from termcolor import colored

path_osmose_home = "/home/datawork-osmose/"
path_osmose_dataset = "/home/datawork-osmose/dataset/"


def main(dataset_ID,analysis_fs, fileScale_nfft ,fileScale_winsize,nberAdjustSpectros,fileScale_overlap,colmapspectros,nber_zoom_levels,min_color_val,max_color_val,maxtime_display_spectro,norma_gliding_zscore,fmin_HighPassFilter):

    
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
    global total_nber_audio_files, orig_fs, orig_fileDuration, orig_total_nber_audio_files, folderName_audioFiles
    metadata = pd.read_csv( os.path.join(path_osmose_dataset , dataset_ID , 'raw/metadata.csv') )
    orig_fs = metadata['orig_fs'][0]
    total_nber_audio_files = metadata['nberWavFiles'][0]        
    orig_total_nber_audio_files = metadata['nberWavFiles'][0]        
    orig_fileDuration = metadata['orig_fileDuration'][0]    
    
    if isinstance(maxtime_display_spectro,str):
        maxtime_display_spectro = int(orig_fileDuration)

    if int(maxtime_display_spectro)>3600:    
        print('for safety reasons, for the moment maxtime_display_spectro must be inferior or equal to 3600 s .. sorry !')
        sys.exit()
    if int(maxtime_display_spectro)>int(orig_fileDuration):    
        print('your maxtime_display_spectro must be inferior or equal to your audio file duration')
        sys.exit()
        
    
    if int(maxtime_display_spectro) != int(orig_fileDuration):
        folderName_audioFiles = str(maxtime_display_spectro)+'_'+str(int(analysis_fs))        
        total_nber_audio_files = int(np.floor(orig_fileDuration / maxtime_display_spectro) * orig_total_nber_audio_files)
    else:
#         folderName_audioFiles = str(int(analysis_fs))    
        folderName_audioFiles = str(int(orig_fileDuration))+'_'+str(int(analysis_fs))       
    
    ## initialize some paths as global variables
    global path_analysisFolder, path_pbsFiles, path_output_spectrograms, orig_folderName_audioFiles
    path_analysisFolder = os.path.join(path_osmose_dataset, dataset_ID, 'analysis')
    path_pbsFiles = os.path.join(path_analysisFolder, 'ongoing_pbsFiles')    
    path_output_spectrograms = os.path.join(path_analysisFolder, 'spectrograms', folderName_audioFiles)
    path_output_DC = os.path.join(path_analysisFolder, 'detectionClassification')
    
    orig_folderName_audioFiles = str(int(orig_fileDuration))+'_'+str(int(orig_fs))
    

    
    
    if nberAdjustSpectros>0:
        path_summstats = os.path.join(path_analysisFolder,'normaParams',orig_folderName_audioFiles)
    else:
        path_summstats = os.path.join(path_analysisFolder,'normaParams',folderName_audioFiles)
            
    
    
    global output_path_audio_files
    output_path_audio_files = os.path.join(path_osmose_dataset, dataset_ID, 'raw','audio', folderName_audioFiles )
    
    if not os.path.exists(path_pbsFiles):
        os.makedirs(path_pbsFiles)     
    if not os.path.exists(path_output_spectrograms):
        os.makedirs(path_output_spectrograms)     
    if not os.path.exists(path_output_DC):
        os.makedirs(path_output_DC)                      

        
        
    # if pbs from audio normalization still exist it means that the calculation of normalization parameters and/or permissions launched automatically at dataset importation is not finished, so you will have to make the user wait
    if (len(glob.glob( os.path.join(path_pbsFiles,'pbs_audioNormalization*')  ))>0) or (len(glob.glob( os.path.join(path_pbsFiles,'pbs_setPermissions*')  ))>0):
        print('SORRY the calculation of normalization parameters and/or permissions launched automatically at dataset importation is not finished, so you will have to wait a bit..')
        sys.exit()    

    elif len(glob.glob(os.path.join(path_pbsFiles, 'pbs_resample*'))) > 0:
        print('Ongoing RESAMPLING, check the Track Progress cell.. NO new generation possible for the moment')
        sys.exit()    

    elif len(glob.glob(os.path.join(path_pbsFiles, 'pbs_spectroGenerationFileScale*'))) > 0: 
        
        print('Ongoing generation of parameter setting spectrograms; wait it is done before running new spectrograms, it will not take long..')
            
        sys.exit()

    else:
        if not os.path.exists(path_pbsFiles):
            os.makedirs(path_pbsFiles)      
        else:
            shutil.rmtree(path_pbsFiles)
            os.makedirs(path_pbsFiles)        
 

        
    
    
    # write analysis.csv and make it global. Must be after the previous sys.exit() because ...
    global analysis_fiche
    data = {'dataset_ID' :dataset_ID
            ,'analysis_fs' :float(analysis_fs)
            ,'fileScale_nfft' : fileScale_nfft,'fileScale_winsize' : fileScale_winsize,'fileScale_overlap' : fileScale_overlap
           ,'colmapspectros' : colmapspectros,'nber_zoom_levels' : nber_zoom_levels,'nberAdjustSpectros':nberAdjustSpectros,
           'min_color_val':min_color_val,'max_color_val':max_color_val,'maxtime_display_spectro':maxtime_display_spectro, 'folderName_audioFiles':folderName_audioFiles, 'norma_gliding_zscore':norma_gliding_zscore,'fmin_HighPassFilter':fmin_HighPassFilter}
    analysis_fiche = pd.DataFrame.from_records([data])
    analysis_fiche.to_csv( os.path.join(path_analysisFolder,'analysis_fiche.csv') )
    
        

    

    # create all pbs files in /home/datawork-osmose/dataset/<dataset_ID>/analysis/ongoing_pbsFiles/
    if (not os.path.exists( output_path_audio_files )) and (analysis_fiche['nberAdjustSpectros'][0]==0):
        
        os.makedirs(output_path_audio_files)               

        main_resample(dataset_ID,analysis_fs, str(int(orig_fileDuration)) + '_' + str(int(orig_fs)),folderName_audioFiles)   
        
        metadata['dataset_fs'] = analysis_fs
        metadata['dataset_fileDuration'] = maxtime_display_spectro
        
        metadata.to_csv( os.path.join(output_path_audio_files, 'metadata.csv') )      
        
        
        print('OK we are generating all data for your annotation campaign now ! \n \n You can look at the progress bar in the cell below, or you can leave Jupyter and come back to Earth, your job is done, we will mail you when it is done ;) ')   
        
        
    elif analysis_fiche['nberAdjustSpectros'][0]!=0:
        print('Generating',analysis_fiche['nberAdjustSpectros'][0],'spectrograms for parameter selection! In a few seconds, your first spectrograms will be visible in the next cell..')
        
              

    main_spectroGenerationFileScale(dataset_ID,analysis_fs)    
    
    list_pbs_resample = sorted( glob.glob(os.path.join(path_pbsFiles, 'pbs_resample*')) )
    list_pbs_spectroGeneFileScale = sorted(glob.glob(os.path.join(path_pbsFiles, 'pbs_spectroGenerationFileScale*') ))

    
    print('\n\n %%%%%%%%% FOR DEBUG : processing chain')

    print('Launch: pbs_resample')
    
    jobID_resample = []
    for job in list_pbs_resample:    
        res = subprocess.run(['qsub', job], stdout=subprocess.PIPE).stdout.decode('utf-8').rstrip('\n')
        jobID_resample.append(res)
        
        
    isnorma = 0
    for cc in ['D','M','H','S','W']:
        if cc in norma_gliding_zscore:
            isnorma = 1
               
    if (not os.path.exists( path_summstats )) and isnorma:

        os.makedirs(path_summstats)          

        main_audioNormalization(dataset_ID,analysis_fs)    

        list_pbs_audioNormalization = sorted(glob.glob(os.path.join(path_pbsFiles, 'pbs_audioNormalization*') ))    

        print('Launch: pbs_audioNormalization')

        jobID_audioNormalization=[]
        for job in list_pbs_audioNormalization:
            res = subprocess.run( ['qsub','-W depend=afterok:'+(':').join(jobID_resample),job],stdout=subprocess.PIPE).stdout.decode('utf-8').rstrip('\n')
            jobID_audioNormalization.append(res)


        jobID_resample = jobID_audioNormalization
        
    else:
        jobID_audioNormalization = ''
                
               
        
        
    # for ADJUSTING PARAMS SPECTROS generation
    if analysis_fiche['nberAdjustSpectros'][0]>0:
        
            # launch spectro generation            
            for job in list_pbs_spectroGeneFileScale:
                res = subprocess.run( ['qsub','-W depend=afterok:'+(':').join(jobID_resample),job],stdout=subprocess.PIPE).stdout.decode('utf-8')        
        
        
    # for COMPLETE spectrogam generation
    else:
        
        name_folder = 'nfft=' + str(fileScale_nfft) + ' winsize=' + str(fileScale_winsize) + \
                              ' overlap=' + str(int( fileScale_overlap)) + ' cvr='+ str(min_color_val) +':0'
        path_output_spectrograms_for_monitoring = os.path.join(path_analysisFolder, 'spectrograms', folderName_audioFiles,name_folder)

        os.makedirs(path_output_spectrograms_for_monitoring)                
        
        # case 1 : use of the original folder
        if (int(maxtime_display_spectro) == int(orig_fileDuration)) and (int(analysis_fs) == int(orig_fs)):
            
            print('CASE 1')
            
            # do nothing on timestamp.csv
            
            print('Launch: pbs_spectroGeneFileScale')
            
            # launch spectro generation
            for job in list_pbs_spectroGeneFileScale:
                jobID_spectroGenerationFileScale = subprocess.run( ['qsub','-W depend=afterok:'+(':').join(jobID_resample),job],stdout=subprocess.PIPE).stdout.decode('utf-8')            
            

        # case 2 : if no resegmentation the timestamp.csv remains the same , so just copy it into the analysis folder
        elif int(maxtime_display_spectro) == int(orig_fileDuration):
            print('CASE 2')
                        
            from shutil import copyfile
            src = os.path.join(path_osmose_dataset, dataset_ID, 'raw','audio',\
                                       str(int(orig_fileDuration))+'_'+str(int(orig_fs)), 'timestamp.csv')
            dst = os.path.join(path_osmose_dataset, dataset_ID, 'raw','audio',folderName_audioFiles, 'timestamp.csv')        
            copyfile(src, dst)
   
            
            print('Launch: pbs_spectroGeneFileScale')
    
            # launch spectro generation            
            for job in list_pbs_spectroGeneFileScale:
                jobID_spectroGenerationFileScale = subprocess.run( ['qsub','-W depend=afterok:'+(':').join(jobID_resample),job],stdout=subprocess.PIPE).stdout.decode('utf-8')             
                
                
        # case 3 : you have a resegmentation so launch qsub_timestampAfterDecoupe.py
        else:
            print('CASE 3')
            
            # this case is : 1) if you only generate spectrograms to adjust parameters or 2) your wav have already been segmented, THEN do not launch the generation of the new timestamp
            if not os.path.isfile(os.path.join(path_osmose_dataset, dataset_ID, 'raw','audio',folderName_audioFiles, 'timestamp.csv') ):
                createQsub_timestampAfterDecoupe(dataset_ID,analysis_fs)

                print('Launch: pbs_timestampAfterDecoupe_0')
                jobID_timestampAfterDecoupe = subprocess.run( ['qsub','-W depend=afterok:'+(':').join(jobID_resample),os.path.join(path_pbsFiles, 'pbs_timestampAfterDecoupe_0.pbs')],stdout=subprocess.PIPE).stdout.decode('utf-8')
            else:
                jobID_timestampAfterDecoupe = jobID_resample # here we need to set jobID_timestampAfterDecoupe to jobID_resample instead of '' because you may want to use jobID_audioNormalization with already existing resampled wav , in that case you will have jobID_resample = jobID_audioNormalization then  jobID_timestampAfterDecoupe = jobID_resample , so list_pbs_spectroGeneFileScale can wait for jobID_audioNormalization jobs..
            
            print('Launch: pbs_spectroGeneFileScale')        
            
            if not isinstance(jobID_timestampAfterDecoupe,str):
                jobID_timestampAfterDecoupe = ''

            for job in list_pbs_spectroGeneFileScale:
                jobID_spectroGenerationFileScale = subprocess.run( ['qsub','-W depend=afterok:'+jobID_timestampAfterDecoupe,job],stdout=subprocess.PIPE).stdout.decode('utf-8')
            

        # set permissions to /analysis
        createQsub_setPermissions(dataset_ID,os.path.join(path_osmose_dataset, dataset_ID) , folderName_audioFiles)


        if isinstance(jobID_spectroGenerationFileScale,list):
            subprocess.run( ['qsub','-W depend=afterok:'+(':').join(jobID_spectroGenerationFileScale),os.path.join(path_pbsFiles, 'pbs_setPermissions_0.pbs')],stdout=subprocess.PIPE).stdout.decode('utf-8')
        else:
            subprocess.run( ['qsub','-W depend=afterok:'+jobID_spectroGenerationFileScale,os.path.join(path_pbsFiles, 'pbs_setPermissions_0.pbs')],stdout=subprocess.PIPE).stdout.decode('utf-8')


#         ## update the file datasets.csv
#         if write_datasets_csv:

#             datasets_path = os.path.join(path_osmose_dataset,'datasets.csv')

#             nn = dataset_ID+' ('+folderName_audioFiles+')'

#             df2 = {'name': nn, 'folder_name': dataset_ID, \
#                    'conf_folder': folderName_audioFiles, 'dataset_type_name':'' , 'dataset_type_desc':'' , \
#                    'files_type': '.wav', 'location_name': '', 'location_desc': '', 'location_lat':'' , 'location_lon':''}

#             if os.path.exists(datasets_path):
# #                 # here just delete current file
# #                 os.remove(datasets_path)

#                 # here add new dataset
#                 met = pd.read_csv(datasets_path)
#                 if nn not in met['name'].values:
#                     met = met.append(df2, ignore_index = True)
#                     met.to_csv(os.path.join(path_osmose_dataset,'datasets.csv') , index=False)

#             else:
#                 met=pd.DataFrame.from_records([df2]) 
#                 met.to_csv(os.path.join(path_osmose_dataset,'datasets.csv') , index=False)        

#                 os.system('chgrp gosmose ' + os.path.join(path_osmose_dataset,'datasets.csv') )
#                 os.system('chmod g+rwx ' + os.path.join(path_osmose_dataset,'datasets.csv') )


    #if analysis_fiche['nberAdjustSpectros'][0]==0:

        #if not 'jobID_timestampAfterDecoupe' in locals():
        #    jobID_timestampAfterDecoupe = ''
        
        # create the log file on the side of analysis , ie job_id with analysis paremeters
        #data = {'analID':analID,'jobID_resample':[ll.split('.')[0] for ll in jobID_resample],'jobID_spectroGenerationFileScale':[ll.split('.')[0] for ll in jobID_spectroGenerationFileScale],'jobID_timestampAfterDecoupe':[ll.split('.')[0] for ll in jobID_timestampAfterDecoupe],'jobID_audioNormalization':[ll.split('.')[0] for ll in jobID_audioNormalization]}       
        #df = pd.DataFrame.from_records([data])
        #df.to_csv( os.path.join(path_log,'jobID_'+analID+'.csv') , index=False)      
        #analysis_fiche.to_csv( os.path.join(path_log,'analysis_fiche_'+analID+'.csv'))

    

def main_audioNormalization(dataset_ID,analysis_fs):
                 
           
    size_batch = 200
    id_job=0

    if total_nber_audio_files < size_batch:
        createQsub_audioNormalization(dataset_ID,analysis_fs,id_job, 0, total_nber_audio_files)
    else:
        for ind in np.arange(0, total_nber_audio_files, size_batch):
            
            if ind == np.arange(0, total_nber_audio_files, size_batch)[-1]:
                ind_max = total_nber_audio_files
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
    
    size_batch = 600
    
    nber_batch = min(orig_total_nber_audio_files / size_batch , 20)
    size_batch = int(round(orig_total_nber_audio_files / nber_batch))
    
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
                    .replace("new_audioFileDuration",str(analysis_fiche['maxtime_display_spectro'][0]))\
                    .replace("orig_audioFileDuration",str(int(orig_fileDuration)))\
                    .replace("nber_segments",str(int(orig_fileDuration / analysis_fiche['maxtime_display_spectro'][0])))
            )

     

def createQsub_timestampAfterDecoupe(dataset_ID,analysis_fs):

    cur_pbs_file = os.path.join(path_pbsFiles, 'pbs_timestampAfterDecoupe_0.pbs')
    logjob_outpath = os.path.join(path_pbsFiles, 'log_timestampAfterDecoupe_0.txt')

    with open(path_osmose_home+"osmoseNotebooks_v0/source/templateQsub_timestampAfterDecoupe.pbs", "r") as template:
        template_lines = template.readlines()
    
    with open(cur_pbs_file, "w") as pbs_file:
        for line in template_lines:
            pbs_file.write(
                line.replace("dataset_ID", dataset_ID)\
                    .replace("logjob_outpath",logjob_outpath) \
                    .replace("analysis_fs", str(int(analysis_fs)))
            )

     


    ## spectroGenerationFileScale

def main_spectroGenerationFileScale(dataset_ID,analysis_fs):
                 
            
    size_batch = 100

    if analysis_fiche['nberAdjustSpectros'][0]!=0:
        if os.path.exists(os.path.join(path_output_spectrograms , 'spectro_adjustParams')):
            shutil.rmtree(os.path.join(path_output_spectrograms , 'spectro_adjustParams'))
        os.makedirs(os.path.join(path_output_spectrograms , 'spectro_adjustParams'))
        cur_total_nber_audio_files = min([orig_total_nber_audio_files, analysis_fiche['nberAdjustSpectros'][0]])
        
    else:        
        cur_total_nber_audio_files = total_nber_audio_files
        
        # re-initialize the folder of spectrograms
        if os.path.exists(path_output_spectrograms):
            shutil.rmtree(path_output_spectrograms)
        os.makedirs(path_output_spectrograms)        
        
    nber_batch = min([cur_total_nber_audio_files / size_batch , 10])
    size_batch = int(round(cur_total_nber_audio_files / nber_batch))
        
    # use multiple nodes with batch of size_batch files ; use it if number of files higher than 1000
    id_job = 0  # used to identify the different jobs created in case cur_total_nber_audio_files > size_batch

#     createQsub_spectroGenerationFileScale(dataset_ID,analysis_fs,id_job, 0, cur_total_nber_audio_files)

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

    with open( os.path.join(path_osmose_home , "osmoseNotebooks_v0/source/templateQsub_spectroGenerationFileScale.pbs"), "r") as template:
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

            



# # aux

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

        

def job_monitoring(dataset_ID,analysis_fs,nfft,winsize,overlap,min_color_val):
        
    if "total_nber_audio_files" not in globals():
        print('Hmm , I guess your notebook was restarted no ? well you have to rerun the first cells up to the cell <Generate spectrograms> to re-initialize me now! and do not worry it will not re-send your jobs ..')
        sys.exit()
                       
    # RESAMPLING
    if len( glob.glob(os.path.join(path_pbsFiles, 'pbs_resample*')) ):
        
        if len(glob.glob(os.path.join(output_path_audio_files, '*.wav'))) == total_nber_audio_files:
            resampling_status = 'DONE'
        else:
            resampling_status = 'ONGOING'
            
        if int(orig_fileDuration) != int(analysis_fiche['maxtime_display_spectro'][0]):
            jobname = 'Segmenting & Resampling'
        else:
            jobname = 'Resampling'
            
        print('o ' + resampling_status +' (', len(glob.glob(os.path.join(output_path_audio_files, '*.wav'))), '/',str(total_nber_audio_files), ')' + ' -> '+   jobname + ' output in : ' + output_path_audio_files)     
    
    
    
    
    
    name_folder = 'nfft=' + str(nfft) + ' winsize=' + str(winsize) + \
                          ' overlap=' + str(int( overlap)) + ' cvr='+ str(min_color_val) +':0'
    path_output_spectrograms_for_monitoring = os.path.join(path_analysisFolder, 'spectrograms', folderName_audioFiles,name_folder)

    

    # FILESCALE_SPECTRO_GENERATION    
    if len( glob.glob(os.path.join(path_pbsFiles, 'pbs_spectroGenerationFileScale*')) ):

        ind_count = 1

        if not os.path.exists(path_output_spectrograms_for_monitoring):
            FileScaleSPECGENE_status = 'WAITING'
            ll = '0'
            
        else:
            
            ll = len(next(os.walk(path_output_spectrograms_for_monitoring))[ind_count])

            if ll == 0:
                FileScaleSPECGENE_status = 'WAITING'

            elif ll == total_nber_audio_files:
                FileScaleSPECGENE_status = 'DONE'            

            else:
                FileScaleSPECGENE_status = 'ONGOING'    

        print('o ' + FileScaleSPECGENE_status +' (',ll,'/',str(total_nber_audio_files),')' + ' -> Spectrogram Generation'  + ' output in : ' + path_output_spectrograms_for_monitoring)    


        
        print('\n\n just so you know you have',len( glob.glob(os.path.join(path_pbsFiles, 'pbs_*')) ),'ongoing jobs working for you! thank you M. DATARMOR!')


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



def params_checkingNberPixels(analysis_fs,maxtime_display_spectro,nfft,winsize,overlap,nber_zoom_levels,dataset_ID):

    if isinstance(maxtime_display_spectro,str):
        metadata = pd.read_csv( os.path.join(path_osmose_dataset , dataset_ID , 'raw/metadata.csv') )     
        maxtime_display_spectro = int(metadata['orig_fileDuration'][0])
     
        
    ############# params checking to be sure that your nber of windows below screen pixel
    
    if nfft > 2048:
        print('your nfft is :',nfft)
        print( colored('PLEASE REDUCE IT UNLESS YOU HAVE A VERY HD SCREEN WITH MORE THAN 1k pixels vertically !!!! ', 'red') )        

#     for level in range(nber_zoom_levels):

    pct_overlap = overlap     
    
    tile_duration = maxtime_display_spectro / 2**(nber_zoom_levels-1)

    data = np.zeros([int(tile_duration * analysis_fs),1 ])

#     cur_win = winsize
#     noverlap = int(cur_win * cur_over / 100)
#     nstep = cur_win - noverlap
#     window_type = 'hamming'

#     win = signal.get_window(window_type, cur_win)
#     x = np.asarray(data)
#     shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // nstep, cur_win)
#     strides = x.strides[:-1] + (nstep * x.strides[-1], x.strides[-1])
#     xinprewin = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
#     xinwin = win * xinprewin

    x= data
    fs=analysis_fs

    Nwin = winsize
    Nfft=nfft
    Noverlap = int(Nwin * pct_overlap / 100)
    
    win = np.hamming(Nwin)
    if Nfft < (0.5*Nwin):
        scale_psd = 2.0 * Nwin / (Nfft * ((win * win).sum() / Nwin))
    else:
        scale_psd = 2.0 / ( (win * win).sum())
    Nbech = np.size(x)
    Noffset = Nwin-Noverlap
    Nbwin = int((Nbech-Nwin) / Noffset)
    Freq =  np.fft.rfftfreq(Nfft, d=1/fs)
    Time = np.linspace(0, Nbech/fs, Nbwin)

    print('your smallest tile has a duration of:',tile_duration,'(s)')
    print('\n')

    if Nbwin>3500:
        print('your number of time windows:',Nbwin )
        print( colored('PLEASE REDUCE IT UNLESS YOU HAVE A VERY HD SCREEN WITH MORE THAN 2k pixels horizontally !!!! ', 'red') )

    else:
        print('your number of time windows in this tile is:',Nbwin )        
        print('\n')
        print('your resolutions : time = ',round(Time[1]-Time[0],3),'(s) / frequency = ', round(Freq[1]-Freq[0],3) ,'(Hz)' )    


def list_datasets(nargout=0):
    
    l_ds = [ss for ss in sorted(os.listdir(path_osmose_dataset)) if '.csv' not in ss ]
    
    if nargout == 0:
        print("Available datasets:")

        for ds in l_ds:
            print("  - {}".format(ds))

    else:
        return l_ds


def display_metadata(dataset_ID):
    
    ## initialize some metadata as global variables    
    global total_nber_audio_files, orig_fs, orig_fileDuration
    metadata = pd.read_csv( os.path.join(path_osmose_dataset , dataset_ID , 'raw/metadata.csv') )
    orig_fs = metadata['orig_fs'][0]
    total_nber_audio_files = metadata['nberWavFiles'][0]        
    orig_fileDuration = metadata['orig_fileDuration'][0]

    print('Original sample frequency (Hz) :',int(orig_fs) )    
    print(metadata['start_date'][0][:16],' --> ',metadata['end_date'][0][:16])        
    print('Cumulated number of days :' , (pd.to_datetime(metadata['end_date'][0], utc=True) - pd.to_datetime(metadata['start_date'][0], utc=True)).days )                
    print('Original audio file duration (s) :',int(orig_fileDuration) )
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
