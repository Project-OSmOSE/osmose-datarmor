
import glob
import os
import pandas as pd
import datetime
import sys
    
path_osmose_dataset = "/home/datawork-osmose/dataset/"

# HYPOTHESIS : your segmented wav should have the form : 'longname_seg*int*.wav' where *int* is a number, possible to start with zeroes, eg 002. Also note that longname can contain the substring '_seg'


def multi_find(s, r):
    # find multiple occurrences of a substring r in a string s
    return [pos for pos in range(len(s)) if s.startswith(r,pos)]


if __name__ == "__main__":

    dataset_ID = sys.argv[1]
    analysis_fs = float(sys.argv[2])
        
    path_analysisFolder = os.path.join(path_osmose_dataset, dataset_ID, 'analysis')
    analysis_fiche = pd.read_csv(os.path.join(path_analysisFolder, 'analysis_fiche.csv'), header=0)
    segment_duration = analysis_fiche['maxtime_display_spectro'][0]
    folderName_audioFiles = analysis_fiche['folderName_audioFiles'][0]

    output_path_audio_files = os.path.join(path_osmose_dataset, dataset_ID, 'raw','audio', folderName_audioFiles )

    # list of wavfile names
    listwav = glob.glob(output_path_audio_files+'/*.wav')

    metadata = pd.read_csv( os.path.join(path_osmose_dataset , dataset_ID , 'raw/metadata.csv') )
    orig_fs = int(metadata['orig_fs'][0])
    orig_fileDuration = int(metadata['orig_fileDuration'][0])

    # load original timestamp.csv
    df = pd.read_csv( os.path.join(path_osmose_dataset, dataset_ID, 'raw','audio',\
                                   str(orig_fileDuration)+'_'+str(orig_fs), 'timestamp.csv' ),header=None)
    
    
    timestamp=[]
    filename_rawaudio=[]
    for audio_file in listwav:

        ind_seg = multi_find(audio_file , '_seg')

        if len(ind_seg)==1: # just in case original files have already _seg in their names, eg ml80a , this occurs if real original files are very very big , so have been segmented in smaller wav first, containing _seg
            ind_seg=ind_seg[0]
        else:
            ind_seg=ind_seg[-1]
        
        # get current original audio filename
        cur_filename = os.path.basename(audio_file[:ind_seg])

        # get current original timestamp
        cur_timestamp = df[df[0].values == cur_filename+'.wav'][1].values[0]

        # get current index segment
        index_cur_segment = int(audio_file[ind_seg+4:-4])

        dates_str = str(datetime.datetime.strftime( datetime.datetime.strptime(cur_timestamp, '%Y-%m-%dT%H:%M:%S.%fZ') \
                                                   + datetime.timedelta(seconds=index_cur_segment * int(segment_duration)) , '%Y-%m-%dT%H:%M:%S'))
        # simply chopping !
        dates_final = dates_str + '.000Z'    

        timestamp.append(dates_final)
        filename_rawaudio.append( os.path.basename(audio_file) )        


    df = pd.DataFrame({'filename':filename_rawaudio,'timestamp':timestamp})
    df.sort_values(by=['timestamp'], inplace=True)
    df.to_csv( os.path.join(output_path_audio_files,'timestamp.csv'), index=False,na_rep='NaN',header=None)
    
    os.remove( os.path.join(path_analysisFolder, 'ongoing_pbsFiles', 'pbs_timestampAfterDecoupe_0.pbs') )

