import os
import wave
import datetime
from datetime import timedelta
import csv
import numpy as np
import time
import pandas as pd
import glob

import json
import matplotlib.pyplot as plt

from datetime import datetime

import sys

from tqdm import tqdm

from aux import *

path_osmose_dataset = "/home/datawork-osmose/dataset/"
path_osmose_home = "/home/datawork-osmose/"



def get_platform_storage_infos():

    import shutil

    usage=shutil.disk_usage(path_osmose_home)
    print("Total storage space (TB):",round(usage.total / (1024**4),1))
    print("Used storage space (TB):",round(usage.used / (1024**4),1))
    print('-----------------------')
    print("Available storage space (TB):",round(usage.free / (1024**4),1))
    
def builder_dataset(dataset_ID,gps):

#     if not os.path.isfile(os.path.join(path_osmose_dataset,dataset_ID, 'raw/metadata.csv')):

    path_timestamp_formatted = os.path.join(path_osmose_dataset, dataset_ID ,'raw' ,'audio' ,'original','timestamp.csv')
    path_raw_audio = os.path.join(path_osmose_dataset, dataset_ID ,'raw' ,'audio','original')

    if not isinstance(gps, list):
        
        csvFileArray = pd.read_csv(os.path.join(path_osmose_dataset, dataset_ID ,'raw' ,'auxiliary' ,gps))
        gps = [(np.min(csvFileArray['lat']) , np.max(csvFileArray['lat'])) , (np.min(csvFileArray['lon']) , np.max(csvFileArray['lon']))]

    csvFileArray = pd.read_csv(path_timestamp_formatted, header=None)

    timestamp_csv = csvFileArray[1].values
    filename_csv= csvFileArray[0].values
       

    list_duration =[]
    list_volumeFile =[]
    list_samplingRate =[]
    list_interWavInterval =[]
    list_size =[]
    list_sampwidth = []
    list_filename = []
    for ind_dt in tqdm(range(len(timestamp_csv))):
#         if np.mod(ind_dt , 200)==0:
#             print(ind_dt , '/',len(timestamp_csv))

        if ind_dt <len(timestamp_csv ) -1:
            diff = datetime.strptime(timestamp_csv[ind_dt +1], '%Y-%m-%dT%H:%M:%S.%fZ') - datetime.strptime \
                (timestamp_csv[ind_dt], '%Y-%m-%dT%H:%M:%S.%fZ')
            list_interWavInterval.append(diff.total_seconds())

        filewav = os.path.join(path_raw_audio ,filename_csv[ind_dt])
        
        list_filename.append(filename_csv[ind_dt])

        with wave.open(filewav, "rb") as wave_file:
            params = wave_file.getparams()
            sr = params.framerate
            frames = params.nframes
            sampwidth = params.sampwidth

        list_size.append(os.path.getsize(filewav) / 1e6)

        list_duration.append(frames / float(sr))
        #     list_volumeFile.append( np.round(sr * params.nchannels * (sampwidth) * frames / float(sr) /1024 /1000))
        list_samplingRate.append( float(sr) )
        list_sampwidth.append(sampwidth)

    dutyCycle_percent = None
    flag_sample_rate = 0
    flag_duration =0
    list_anomalies =dict()




    dd = pd.DataFrame(list_interWavInterval).describe()
    print('%%%%%%%%%%%%%INTERWAV DURATION%%%%%%%%%%%%%%%%%%%%')
    print('file inter-wav duration : ',dd,'\n')
    if (dd[0]['std' ] <1e-10) & (flag_duration==0):
        dutyCycle_percent =  round(100 *pd.DataFrame(list_duration).values.flatten().mean( ) /pd.DataFrame
            (list_interWavInterval).values.flatten().mean() ,1)
    else:
        list_anomalies['interWav' ] =list_interWavInterval

    dd = pd.DataFrame(list_samplingRate).describe()
    print('%%%%%%%%%%%%%SAMPLING RATE%%%%%%%%%%%%%%%%%%%%')
    print('file sampling rate : ',dd,'\n')
    if dd[0]['std' ] >1e-10:
        flag_sample_rate = 1
        list_anomalies['samplingRate' ] =list_samplingRate

            
    data = {'orig_fs' :float(pd.DataFrame(list_samplingRate).values.flatten().mean())
            ,'sound_sample_size_in_bits' :int( 8 *pd.DataFrame(list_sampwidth).values.flatten().mean())
            ,'nchannels' :int(params.nchannels) ,'nberWavFiles': len(filename_csv) ,'start_date' :timestamp_csv[0]
            ,'end_date' :timestamp_csv[-1] ,'dutyCycle_percent' :dutyCycle_percent
            ,'orig_fileDuration' :round(pd.DataFrame(list_duration).values.flatten().mean() ,2)
            ,'orig_fileVolume' :pd.DataFrame(list_size).values.flatten().mean()
            ,'orig_totalVolume' :round(pd.DataFrame(list_size).values.flatten().mean() * len(filename_csv) /1000, 1),
            'orig_totalDurationMins': round(pd.DataFrame(list_duration).values.flatten().mean() * len(filename_csv) / 60, 2),'lat':gps[0],'lon':gps[1]}

    df = pd.DataFrame.from_records([data])
    df.to_csv( os.path.join(path_osmose_dataset, dataset_ID ,'raw' ,'metadata.csv') , index=False)  
    
    
    df['dataset_fs'] = float(pd.DataFrame(list_samplingRate).values.flatten().mean()) 
    df['dataset_fileDuration']=round(pd.DataFrame(list_duration).values.flatten().mean() ,2)

    df.to_csv( os.path.join(path_osmose_dataset, dataset_ID ,'raw' ,'audio','original', 'metadata.csv') , index=False)  
    

    nominalVal_size= int(np.percentile(list_size, 90))

    print('%%%%%%%%%%%%%SIZE%%%%%%%%%%%%%%%%%%%%')
    dd_size = pd.DataFrame(list_size).describe()    
    print('file size : ',dd_size,'\n')    # go through the duration and check whether anormal files

    
    ct_anormal_size = 0
    for name,size in zip(list_filename,list_size):

        if int(size) !=  nominalVal_size :
            ct_anormal_size+=1
            print(name,'has different size',str(int(size)),'from average',str(nominalVal_size))
            

    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('anormal files on SIZE:',ct_anormal_size,'/',len(list_filename))
    
    


    nominalVal_duration= int(np.percentile(list_duration, 90))

    print('%%%%%%%%%%%%%DURATION%%%%%%%%%%%%%%%%%%%%')
    dd_duration = pd.DataFrame(list_duration).describe()
    print('file duration : ',dd_duration,'\n')

    # go through the duration and check whether anormal files
    ct_anormal_duration=0
    list_filename_anormal_duration = []

    for name,duration in zip(list_filename,list_duration):

        if int(duration) != int(nominalVal_duration):
            ct_anormal_duration+=1
            print(name,'has different duration',str(int(duration)),'from 90th percentile',str(int(nominalVal_duration)))

            list_filename_anormal_duration.append(name)
            
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('anormal files on DURATION:',ct_anormal_duration,'/',len(list_filename))
        
            
            
    if (ct_anormal_duration>0) | (ct_anormal_size>0):
        print('\n \n SORRY BUT YOUR DATASET CONTAINS ANORMAL FILES , so IT HAS NOT BEEN IMPORTED ON OSMOSE PLATFORM .. you can use the next cell to directly delete them')
        return list_filename_anormal_duration
    
    else:

        # change name of the original wav folder
        os.rename( os.path.join(path_osmose_dataset, dataset_ID ,'raw' ,'audio','original') , os.path.join(path_osmose_dataset, dataset_ID ,'raw' ,'audio',str(int(pd.DataFrame(list_duration).values.flatten().mean()))+'_'+str(int(float(pd.DataFrame(list_samplingRate).values.flatten().mean())))))

        # change permission on the dataset
        print('\n Now setting OSmOSE permissions ; wait a bit ...')
        os.system('chgrp -R gosmose /home/datawork-osmose/dataset/'+dataset_ID)
        os.system('chmod -R g+rw /home/datawork-osmose/dataset/'+dataset_ID)
        print('\n DONE ! you dataset is on OSmOSE platform !')
    
    
    
# def keep_only_filesAlmostSameDura_deleteOthers(dataset_ID,list_filename_anormal_duration):
        
#     path_raw_audio = os.path.join(path_osmose_dataset, dataset_ID ,'raw' ,'audio','original')

#     csvFileArray = pd.read_csv(os.path.join(path_osmose_dataset, dataset_ID ,'raw' ,'audio' ,'original','timestamp.csv'), header=None)
    
#     almostDuraIs = 

#     for ll in list_filename_anormal_duration:

#         filewav = os.path.join(path_raw_audio ,ll)

#         csvFileArray=csvFileArray.drop(csvFileArray[ csvFileArray[0].values == os.path.basename(ll)].index)    

#         print('removing : ',os.path.basename(ll))
#         os.remove(filewav)

#     csvFileArray.sort_values(by=[1], inplace=True)
#     csvFileArray.to_csv(os.path.join(path_osmose_dataset, dataset_ID ,'raw' ,'audio' ,'original','timestamp.csv'), index=False,na_rep='NaN',header=None)

#     print('\n ALL ANORMAL FILES REMOVED ! you can now re-run the previous file to finish importing it on OSmOSE platform')
    
def list_not_builded_datasets(nargout=0):

    l_ds = [ss for ss in sorted(os.listdir(path_osmose_dataset)) if '.csv' not in ss ]

    list_not_builded_datasets = []

    for dd in l_ds:

        if os.path.exists( os.path.join(path_osmose_dataset,dd,'raw/audio/original/') ):
            list_not_builded_datasets.append(dd)

    print("List of the datasets not built yet:")

    for ds in list_not_builded_datasets:
        print("  - {}".format(ds))    
        
    
def delete_anormal_files(dataset_ID,list_filename_anormal_duration):
        
    path_raw_audio = os.path.join(path_osmose_dataset, dataset_ID ,'raw' ,'audio','original')

    csvFileArray = pd.read_csv(os.path.join(path_osmose_dataset, dataset_ID ,'raw' ,'audio' ,'original','timestamp.csv'), header=None)

    for ll in list_filename_anormal_duration:

        filewav = os.path.join(path_raw_audio ,ll)

        csvFileArray=csvFileArray.drop(csvFileArray[ csvFileArray[0].values == os.path.basename(ll)].index)    

        print('removing : ',os.path.basename(ll))
        os.remove(filewav)

    csvFileArray.sort_values(by=[1], inplace=True)
    csvFileArray.to_csv(os.path.join(path_osmose_dataset, dataset_ID ,'raw' ,'audio' ,'original','timestamp.csv'), index=False,na_rep='NaN',header=None)

    print('\n ALL ANORMAL FILES REMOVED ! you can now re-run the previous file to finish importing it on OSmOSE platform')


# To delete : test push on github