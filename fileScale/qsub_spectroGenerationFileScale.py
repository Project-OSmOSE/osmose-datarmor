import soundfile
import matplotlib.pyplot as plt
from scipy import signal
import pickle
import sys
import numpy as np
import os
import glob
from multiprocessing import Pool
import time
import pandas as pd
import math
import shutil
from PIL import Image

from module_activityFuncs import naive_PSDbased

plt.switch_backend('agg')

fontsize = 16
ticksize = 12
plt.rc('font', size=fontsize)  # controls default text sizes
plt.rc('axes', titlesize=fontsize)  # fontsize of the axes title
plt.rc('axes', labelsize=fontsize)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=ticksize)  # fontsize of the tick labels
plt.rc('ytick', labelsize=ticksize)  # fontsize of the tick labels
plt.rc('legend', fontsize=ticksize)  # legend fontsize
plt.rc('figure', titlesize=ticksize)  # fontsize of the figure title

path_osmose_dataset = "/home/datawork-osmose/dataset/"

import imageio
import matplotlib.image as mpimg


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def gen_spectro(max_w, min_color_val, data, sample_rate, output_file, win_size, nfft, pct_overlap,
                main_ref):
    
    noverlap = int(win_size * pct_overlap / 100)
    nperseg = win_size
    nstep = nperseg - noverlap

    window_type = 'hamming'

    win = signal.get_window(window_type, nperseg)

    x = np.asarray(data)
    shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // nstep, nperseg)
    strides = x.strides[:-1] + (nstep * x.strides[-1], x.strides[-1])
    
    print('shape: ',shape)
    print('strides: ',strides)
    
    xinprewin = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    xinwin = win * xinprewin

    result = xinwin.real
    func = np.fft.rfft
    fftraw = func(result, n=nfft)

    scale_psd = 1.0 / (sample_rate * (win * win).sum())
    vPSD_noBB = np.conjugate(fftraw) * fftraw
    vPSD_noBB *= scale_psd

    if nfft % 2:
        vPSD_noBB[..., 1:] *= 2
    else:
        vPSD_noBB[..., 1:-1] *= 2

    spectro = vPSD_noBB.real
    segment_times = np.arange(nperseg / 2, x.shape[-1] - nperseg / 2 + 1, nperseg - noverlap) / float(sample_rate)
    frequencies = np.fft.rfftfreq(nfft, 1 / sample_rate)
    spectro = spectro.transpose()
    
    # Setting self.max_w and normalising spectro as needed
    if main_ref:
        # Restricting spectro frenquencies for dynamic range
        freqs_to_keep = (frequencies == frequencies)
        if min_freq_dyn:
            freqs_to_keep *= min_freq_dyn <= frequencies
        if max_freq_dyn:
            freqs_to_keep *= frequencies <= max_freq_dyn
        max_w = np.amax(spectro[freqs_to_keep, :])
    elif not isinstance(max_over_dataset, list):
        max_w = max_over_dataset

    if main_ref and autofind_minw:
        temp_log_spectro = 10 * np.log10(np.array(spectro / max_w))
        min_color_val = np.min([min_color_val, np.amin(temp_log_spectro) + autofind_minw])

    spectro = spectro / max_w

    # Switching to log spectrogram
    log_spectro = 10 * np.log10(np.array(spectro))
        
    if performDetection:
        naive_PSDbased(np.array(spectro),frequencies,path_analysisFolder,output_file)
    

    # Ploting spectrogram
    my_dpi = 100
    fact_x = 1.3
    fact_y = 1.3
    fig,ax = plt.subplots(nrows=1, ncols=1,figsize=(fact_x * 1800 / my_dpi, fact_y * 512 / my_dpi), dpi=my_dpi)
    color_map = plt.cm.get_cmap(colmapspectros).reversed()
    plt.pcolormesh(segment_times, frequencies, log_spectro, cmap=color_map)
    plt.clim(vmin=min_color_val, vmax=max_color_val)

    if nberAdjustSpectros==0:    
        fig.axes[0].get_xaxis().set_visible(False)
        fig.axes[0].get_yaxis().set_visible(False)
        
        if forAPLOSEcampaign==False:            
            output_file = os.path.join(path_output_spectrograms , os.path.basename(output_file) ) 
                
    else:
        
        if not performDetection:
            fig.axes[0].get_xaxis().set_visible(True)
            fig.axes[0].get_yaxis().set_visible(True)        
            ax.set_ylabel('Frequency (Hz)')        
            ax.set_xlabel('Time (s)')       
            plt.colorbar()
        else:
            fig.axes[0].get_xaxis().set_visible(False)
            fig.axes[0].get_yaxis().set_visible(False)            

    # Saving spectrogram plot to file
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0, dpi=my_dpi)
    plt.close()    

    return max_w, min_color_val



def next_power_of_2(x):
    return 1 if x == 0 else 2**(math.ceil(math.log2(x)))

def next_power_of_2_minus(x):
    return 1 if x == 0 else 2**(math.ceil(math.log2(x))-1)


def gen_tiles(tile_levels, data, sample_rate, output, winsize, nfft, overlap,min_color_val,segment_str):

    """Generates multiple spectrograms for zoom tiling"""
    duration = len(data) / int(sample_rate)

    max_w = 0
                
    for level in range(tile_levels):

        zoom_level = 2 ** level
        tile_duration = duration / zoom_level
        main_ref = (isinstance(max_over_dataset, list)) and (level == 0)

        for tile in range(0, zoom_level):
            
            print(tile)
            
            start = tile * tile_duration
            end = start + tile_duration
            
            if forAPLOSEcampaign:              
                output_file = output[:-4]+ segment_str + '_' + str(zoom_level) + '_' + str(tile) + '.png'                  
            else:                
                output_file = output[:-4] + '.png'                  

            sample_data = data[int(start * sample_rate):int((end + 1) * sample_rate)]
            
            if isinstance(winsize,list):
                cur_win = winsize[level]
                cur_over = overlap[level]
            else:
                cur_win = winsize
                cur_over = overlap                 
            
            max_w, min_color_val = gen_spectro(max_w, min_color_val, sample_data, sample_rate, output_file, cur_win,nfft, cur_over,main_ref)


def process_file(audio_file):
        
    segment_str = ''

    if nberAdjustSpectros==0:        
        data, sample_rate = soundfile.read( os.path.join(path_audio_files,audio_file)  )  
                
        if forAPLOSEcampaign:     
            os.mkdir( os.path.join(path_output_spectrograms , name_folder , audio_file[:-4]) )            
            
        output_file = os.path.join(path_output_spectrograms , name_folder , audio_file[:-4] , audio_file)
        
        gen_tiles(nber_zoom_levels, data, sample_rate, output_file, fileScale_winsize, fileScale_nfft,fileScale_overlap,min_color_val,segment_str)   
        
    else:                
        
        if int(orig_fs) != int(analysis_fs):
            os.system("/appli/sox/sox-14.4.2_gcc-7.2.0/bin/sox '"+ path_audio_files +'/'+ audio_file +"' -r "+ str(analysis_fs) +" -t wavpcm "+ path_audio_files +"'/temp_'"+audio_file+";")    
            data, sample_rate = soundfile.read( os.path.join(path_audio_files,'temp_'+audio_file )  ) 
        else:
            data, sample_rate = soundfile.read( os.path.join(path_audio_files,audio_file)  )  
            
        output_file = os.path.join(path_output_spectrograms , 'spectro_adjustParams', audio_file)   
                        
            
        if int(maxtime_display_spectro) == int(orig_fileDuration):
            gen_tiles(nber_zoom_levels, data, sample_rate, output_file, fileScale_winsize,fileScale_nfft,fileScale_overlap,min_color_val,segment_str)
            
        elif len(data) > maxtime_display_spectro * int(sample_rate):        
            seg = np.arange(0,len(data)+1,maxtime_display_spectro * int(sample_rate))                
            ct=0
            for t1,t2 in zip(seg[:-1],seg[1:]):     
                segment_str = '_seg'+str(ct)
                gen_tiles(nber_zoom_levels, data[t1:t2], sample_rate, output_file, fileScale_winsize, fileScale_nfft,fileScale_overlap,min_color_val,segment_str)
                ct+=1                
                if ct == nberAdjustSpectros:
                    break                
                
        else:
            print('maxtime_display_spectro (',maxtime_display_spectro,' s) must be smaller than your audio file duration (',orig_fileDuration,' s)')
            

if __name__ == "__main__":

    dataset_ID = sys.argv[1]
    analysis_fs = float(sys.argv[2])
    ind_min = int(sys.argv[3]) 
    ind_max = int(sys.argv[4]) 

    # load needed variables from raw metadata
    metadata = pd.read_csv( os.path.join(path_osmose_dataset , dataset_ID , 'raw/metadata.csv') )
    orig_fileDuration = metadata['orig_fileDuration'][0]
    orig_fs = metadata['orig_fs'][0]
    total_nber_audio_files = metadata['nberWavFiles'][0]        

    # build a few paths
    path_analysisFolder = os.path.join(path_osmose_dataset, dataset_ID, 'analysis')
    path_analysis_fiche = os.path.join(path_analysisFolder,'analysis_fiche.csv')

    # load needed variables from analysis fiche
    analysis_fiche = pd.read_csv(path_analysis_fiche,header=0)
    fileScale_nfft = analysis_fiche['fileScale_nfft'][0]
    
    exec('fileScale_winsize=' + analysis_fiche['fileScale_winsize'][0])
    exec('fileScale_overlap=' + analysis_fiche['fileScale_overlap'][0])
  
    #fileScale_winsize = analysis_fiche['fileScale_winsize'][0]
#     fileScale_overlap = analysis_fiche['fileScale_overlap'][0]
    colmapspectros = analysis_fiche['colmapspectros'][0]
    nber_zoom_levels = analysis_fiche['nber_zoom_levels'][0]
    min_color_val = analysis_fiche['min_color_val'][0]
    nberAdjustSpectros = analysis_fiche['nberAdjustSpectros'][0]    
    maxtime_display_spectro = analysis_fiche['maxtime_display_spectro'][0]      
    forAPLOSEcampaign = analysis_fiche['forAPLOSEcampaign'][0] 
    performDetection = analysis_fiche['performDetection'][0] 

    max_over_dataset = []
    max_freq_dyn = analysis_fs/2
    min_freq_dyn = 0
    autofind_minw = []

    
    # selection mode of audio files
    random_selection = False
    nber_files_to_process = 0
    indexation_selection = True

    # internal variable initialization
    equalize_spectro = True
    max_color_val = 0
    
    if autofind_minw:
        name_folder = 'nfft=' + str(fileScale_nfft) + ' winsize=' + str(fileScale_winsize[0]) + \
                      ' overlap=' + str(int( fileScale_overlap[0])) + ' cvr=autofind_min:0'
    else:
        name_folder = 'nfft=' + str(fileScale_nfft) + ' winsize=' + str(fileScale_winsize[0]) + \
                      ' overlap=' + str(int( fileScale_overlap[0])) + ' cvr='+ str(min_color_val) +':0'
        
#     if int(maxtime_display_spectro) != int(orig_fileDuration):
#         folderName_audioFiles = str(maxtime_display_spectro)+'_'+str(int(analysis_fs))
#     else:
#         folderName_audioFiles = str(orig_fileDuration)+'_'+str(int(analysis_fs))
    folderName_audioFiles = str(maxtime_display_spectro)+'_'+str(int(analysis_fs))
    
    path_audio_files = os.path.join(path_osmose_dataset, dataset_ID, 'raw/audio', folderName_audioFiles )
    path_output_spectrograms = os.path.join(path_analysisFolder, 'spectrograms', folderName_audioFiles)
    
    if not os.path.exists(os.path.join(path_output_spectrograms ,'spectrograms.csv')):
        data = {'name' :name_folder , 'nfft':fileScale_nfft , 'window_size' : fileScale_winsize[0], 'overlap' : fileScale_overlap[0]/100 , 'zoom_level': 2**(nber_zoom_levels-1) , 'desc':''}
        df = pd.DataFrame.from_records([data])
        df.to_csv( os.path.join(path_output_spectrograms ,'spectrograms.csv') , index=False)     
    
    
    # when you want to adjust your spectro params, we suppose that your wav files with new fs and duration file are just not ready, and we work with original ones:
    if nberAdjustSpectros>0:
        ind_max = int(np.ceil(nberAdjustSpectros / np.round(orig_fileDuration / maxtime_display_spectro) ) )
        path_audio_files = os.path.join(path_osmose_dataset, dataset_ID, 'raw/audio', str(int(orig_fileDuration))+'_'+str(int(orig_fs)) )

                
    list_wav_withEvent_comp = glob.glob(os.path.join( path_audio_files , '*wav'))


    if nberAdjustSpectros>0:
        list_ind = np.random.randint(len(list_wav_withEvent_comp),size=ind_max-ind_min)
#         list_ind = range(14 , 14+nberAdjustSpectros)        
        
        list_wav_withEvent = [list_wav_withEvent_comp[ii] for ii in list_ind]
    elif indexation_selection:
        list_wav_withEvent = list_wav_withEvent_comp[ind_min:ind_max]

    list_wav_withEvent = [os.path.basename(x) for x in list_wav_withEvent]

    if nberAdjustSpectros==0:
 
        ncpus = 10
        with Pool(processes=ncpus) as pool:
            pool.map(process_file, list_wav_withEvent)
            pool.close()
        
#         for file in list_wav_withEvent:
#             print(file)
#             process_file(file)

    else:
        for file in list_wav_withEvent:
            process_file(file)
           


    os.remove( os.path.join(path_analysisFolder, 'ongoing_pbsFiles', 'pbs_spectroGenerationFileScale_' + str(ind_min) + '.pbs') )

    if nberAdjustSpectros>0:
        for ff in glob.glob(os.path.join(path_audio_files, 'temp_*')):
            os.remove( ff )
