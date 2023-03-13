import wave
import datetime

from datetime import datetime

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

    path_timestamp_formatted = os.path.join(path_osmose_dataset, dataset_ID ,'raw' ,'audio' ,'timestamp.csv')
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
    for ind_dt in tqdm(range(len(timestamp_csv))):
#         if np.mod(ind_dt , 200)==0:
#             print(ind_dt , '/',len(timestamp_csv))

        if ind_dt <len(timestamp_csv ) -1:
            diff = datetime.strptime(timestamp_csv[ind_dt +1], '%Y-%m-%dT%H:%M:%S.%fZ') - datetime.strptime \
                (timestamp_csv[ind_dt], '%Y-%m-%dT%H:%M:%S.%fZ')
            list_interWavInterval.append(diff.total_seconds())

        filewav = os.path.join(path_raw_audio ,filename_csv[ind_dt])

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

    dd = pd.DataFrame(list_duration).describe()
    if dd[0]['std' ] >1e-10:
        flag_duration =1
        list_anomalies['duration' ] =list_duration

    dd = pd.DataFrame(list_interWavInterval).describe()
    if (dd[0]['std' ] <1e-10) & (flag_duration==0):
        dutyCycle_percent =  round(100 *pd.DataFrame(list_duration).values.flatten().mean( ) /pd.DataFrame
            (list_interWavInterval).values.flatten().mean() ,1)
    else:
        list_anomalies['interWav' ] =list_interWavInterval

    dd = pd.DataFrame(list_samplingRate).describe()
    if dd[0]['std' ] >1e-10:
        flag_sample_rate = 1
        list_anomalies['samplingRate' ] =list_samplingRate

        

    
    data = {'orig_fs' :float(pd.DataFrame(list_samplingRate).values.flatten().mean())
            ,'sound_sample_size_in_bits' :int( 8 *pd.DataFrame(list_sampwidth).values.flatten().mean())
            ,'nchannels' :int(params.nchannels) ,'nberWavFiles': len(filename_csv) ,'start_date' :timestamp_csv[0]
            ,'end_date' :timestamp_csv[-1] ,'dutyCycle_percent' :dutyCycle_percent
            ,'audioFile_duration' :round(pd.DataFrame(list_duration).values.flatten().mean() ,2)
            ,'audioFile_volume' :pd.DataFrame(list_size).values.flatten().mean()
            ,'total_volume' :round(pd.DataFrame(list_size).values.flatten().mean() * len(filename_csv) /1000, 1),
            'total_minutes': round(pd.DataFrame(list_duration).values.flatten().mean() * len(filename_csv) / 60, 2),'lat':gps[0],'lon':gps[1]}

#         # go through the duration and check whether anormal files
#         dd = pd.DataFrame(list_duration).describe()
#         ct=0
#         list_anormFiles = []
#         ct_anormal = 0
#         for ll in list_duration:

# #             if (ll < 0.5 * dd[0]['mean']) or (ll > 1.5 * dd[0]['mean']):
#             if int(ll) != 3600:# dd[0]['mean']) or (ll > 1.5 * dd[0]['mean']):
#                 print('please remove file:',filename_csv[ct])
#                 ct_anormal+=1
#                 list_anormFiles.append(path_raw_audio+'/'+filename_csv[ct])

#             ct+=1

#         if ct_anormal==0:      

    df = pd.DataFrame.from_records([data])
    df.to_csv( os.path.join(path_osmose_dataset,dataset_ID, 'raw/metadata.csv') )

#     if pd.DataFrame(list_interWavInterval).describe()[0]['std' ] > 1e-10:
#         pd.DataFrame(list_interWavInterval).describe().to_csv( os.path.join(path_osmose_dataset,dataset_ID, 'raw/metadata_interWavInterval.csv') )        

#     if pd.DataFrame(list_duration).describe()[0]['std' ] > 1e-10:
#         pd.DataFrame(list_duration).describe().to_csv( os.path.join(path_osmose_dataset,dataset_ID, 'raw/metadata_duration.csv') )              

#     if pd.DataFrame(list_samplingRate).describe()[0]['std' ] > 1e-10:
#         pd.DataFrame(list_samplingRate).describe().to_csv( os.path.join(path_osmose_dataset,dataset_ID, 'raw/metadata_samplingRate.csv') ) 


    # change name of the original wav folder
    os.rename( os.path.join(path_osmose_dataset, dataset_ID ,'raw' ,'audio','original') , os.path.join(path_osmose_dataset, dataset_ID ,'raw' ,'audio',str(int(pd.DataFrame(list_duration).values.flatten().mean()))+'_'+str(int(float(pd.DataFrame(list_samplingRate).values.flatten().mean())))))
    
    # change permission on the dataset
    print('Now setting OSmOSE permissions ; wait a bit ...')
    os.system('chgrp -R gosmose /home/datawork-osmose/dataset/'+dataset_ID)
    os.system('chmod -R g+rw /home/datawork-osmose/dataset/'+dataset_ID)
    print('DONE !')
    
    
        
        