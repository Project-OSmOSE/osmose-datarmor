import soundfile
import matplotlib.pyplot as plt
from scipy import signal
import sys
import numpy as np
import os
import glob
from multiprocessing import Pool
import pandas as pd

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



def generate_and_save_figures(colmapspectros, segment_times, Freq, log_spectro, min_color_val, max_color_val,
                              output_file):
    # Ploting spectrogram
    my_dpi = 100
    fact_x = 1.3
    fact_y = 1.3
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fact_x * 1800 / my_dpi, fact_y * 512 / my_dpi), dpi=my_dpi)
    color_map = plt.cm.get_cmap(colmapspectros)  # .reversed()
    plt.pcolormesh(segment_times, Freq, log_spectro, cmap=color_map)
    plt.clim(vmin=min_color_val, vmax=max_color_val)
    # plt.colorbar()

    if nberAdjustSpectros == 0:
        fig.axes[0].get_xaxis().set_visible(False)
        fig.axes[0].get_yaxis().set_visible(False)

    else:

        fig.axes[0].get_xaxis().set_visible(True)
        fig.axes[0].get_yaxis().set_visible(True)
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (s)')
        plt.colorbar()

    # Saving spectrogram plot to file
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0, dpi=my_dpi)
    plt.close()


def gen_spectro(max_w, min_color_val, data, sample_rate, win_size, nfft, pct_overlap, duration, output_file):
    x = data
    fs = sample_rate

    Nwin = win_size
    Nfft = nfft
    Noverlap = int(win_size * pct_overlap / 100)

    win = np.hamming(Nwin)
    if Nfft < (0.5 * Nwin):
        scale_psd = 2.0 
    else:
        scale_psd = 2.0 / ((win * win).sum())
    Nbech = np.size(x)
    Noffset = Nwin - Noverlap
    Nbwin = int((Nbech - Nwin) / Noffset)
    Freq = np.fft.rfftfreq(Nfft, d=1 / fs)
    Sxx = np.zeros([np.size(Freq), Nbwin])
    Time = np.linspace(0, Nbech / fs, Nbwin)
    for idwin in range(Nbwin):
        if Nfft < (0.5 * Nwin):
            x_win = x[idwin * Noffset:idwin * Noffset + Nwin]
            _, Sxx[:, idwin] = signal.welch(x_win, fs=fs, window='hamming', nperseg=Nfft,
                                            noverlap=int(Nfft / 2) , scaling='density')
        else:
            x_win = x[idwin * Noffset:idwin * Noffset + Nwin] * win
            Sxx[:, idwin] = (np.abs(np.fft.rfft(x_win, n=Nfft)) ** 2)
        Sxx[:, idwin] *= scale_psd

    log_spectro = 10 * np.log10(Sxx)

    segment_times = np.linspace(0, duration, Sxx.shape[
        1])  # np.arange(win_size / 2, x.shape[-1] - win_size / 2 + 1, win_size - Noverlap) / float(sample_rate)

    print('DIM LOG SPECTRO:', log_spectro.shape)
    generate_and_save_figures(colmapspectros, segment_times, Freq, log_spectro, min_color_val, max_color_val,
                              output_file)

    return Sxx, Freq


def gen_tiles(nber_level, data, sample_rate, output, winsize, nfft, overlap, min_color_val, segment_str):
    if not isnorma:
        data = (data - np.mean(data)) / np.std(data)
    else:
        data = (data - zscore_mean) / zscore_std

    duration = len(data) / int(sample_rate)

    max_w = 0

    nber_tiles_lowest_zoom_level = 2 ** (nber_level - 1)
    tile_duration = duration / nber_tiles_lowest_zoom_level

    print('LEVEL:', str(nber_tiles_lowest_zoom_level))

    Sxx_2 = np.empty((int(nfft / 2) + 1, 1))
    for tile in range(0, nber_tiles_lowest_zoom_level):
        start = tile * tile_duration
        end = start + tile_duration

        sample_data = data[int(start * sample_rate):int((end + 1) * sample_rate)]

        output_file = output[:-4] + segment_str + '_' + str(nber_tiles_lowest_zoom_level) + '_' + str(tile) + '.png'

        Sxx, Freq = gen_spectro(max_w, min_color_val, sample_data, sample_rate, winsize, nfft, overlap,
                                len(sample_data) / sample_rate, output_file)

        Sxx_2 = np.hstack((Sxx_2, Sxx))

    Sxx_lowest_level = Sxx_2[:, 1:]

    segment_times = np.linspace(0, len(data) / sample_rate, Sxx_lowest_level.shape[1])[np.newaxis, :]

    for ll in range(nber_level)[::-1]:

        print('LEVEL:', ll)

        nberspec = Sxx_lowest_level.shape[1] // (2 ** ll)

        for kk in range(2 ** ll):
            Sxx_int = Sxx_lowest_level[:, kk * nberspec: (kk + 1) * nberspec][:, ::2 ** (nber_level - ll)]

            segment_times_int = segment_times[:, kk * nberspec: (kk + 1) * nberspec][:, ::2 ** (nber_level - ll)]

            log_spectro = 10 * np.log10(Sxx_int)

            print('DIM LOG SPECTRO:', log_spectro.shape)

            output_file = output[:-4] + segment_str + '_' + str(2 ** ll) + '_' + str(kk) + '.png'

            generate_and_save_figures(colmapspectros, segment_times_int, Freq, log_spectro, min_color_val,
                                      max_color_val, output_file)


def process_file(audio_file):
    print(audio_file)

    global zscore_mean, zscore_std

    if isnorma:
        zscore_mean = summStats[summStats['filename'] == audio_file]['mean_avg'].values[0]
        zscore_std = summStats[summStats['filename'] == audio_file]['std_avg'].values[0]

    segment_str = ''

    if nberAdjustSpectros == 0:
        data, sample_rate = soundfile.read(os.path.join(path_audio_files, audio_file))

        bpcoef = signal.butter(20, np.array([fmin_HighPassFilter, sample_rate / 2 - 1]), fs=sample_rate, output='sos',
                               btype='bandpass')
        data = signal.sosfilt(bpcoef, data)

        if not os.path.exists(os.path.join(path_output_spectrograms, name_folder, audio_file[:-4])):
            os.makedirs(os.path.join(path_output_spectrograms, name_folder, audio_file[:-4]))

        output_file = os.path.join(path_output_spectrograms, name_folder, audio_file[:-4], audio_file)

        gen_tiles(nber_zoom_levels, data, sample_rate, output_file, fileScale_winsize, fileScale_nfft,
                  fileScale_overlap, min_color_val, segment_str)

    else:

        if int(orig_fs) != int(analysis_fs):
            os.system("/appli/sox/sox-14.4.2_gcc-7.2.0/bin/sox '" + path_audio_files + '/' + audio_file + "' -r " + str(
                analysis_fs) + " -t wavpcm " + path_audio_files + "'/temp_'" + audio_file + ";")
            data, sample_rate = soundfile.read(os.path.join(path_audio_files, 'temp_' + audio_file))
        else:
            data, sample_rate = soundfile.read(os.path.join(path_audio_files, audio_file))

        output_file = os.path.join(path_output_spectrograms, 'spectro_adjustParams', audio_file)

        bpcoef = signal.butter(20, np.array([fmin_HighPassFilter, sample_rate / 2 - 1]), fs=sample_rate, output='sos',
                               btype='bandpass')
        data = signal.sosfilt(bpcoef, data)

        if int(maxtime_display_spectro) == int(orig_fileDuration):
            gen_tiles(nber_zoom_levels, data, sample_rate, output_file, fileScale_winsize, fileScale_nfft,
                      fileScale_overlap, min_color_val, segment_str)

        elif len(data) > maxtime_display_spectro * int(sample_rate):
            seg = np.arange(0, len(data) + 1, maxtime_display_spectro * int(sample_rate))
            ct = 0
            for t1, t2 in zip(seg[:-1], seg[1:]):
                segment_str = '_seg' + str(ct)
                gen_tiles(nber_zoom_levels, data[t1:t2], sample_rate, output_file, fileScale_winsize, fileScale_nfft,
                          fileScale_overlap, min_color_val, segment_str)
                ct += 1
                if ct == nberAdjustSpectros:
                    break

        else:
            print('maxtime_display_spectro (', maxtime_display_spectro,
                  ' s) must be smaller than your audio file duration (', orig_fileDuration, ' s)')


if __name__ == "__main__":

    # if you want to use this script in local, set am_local to 1 and review lines from 227 to 244
    am_local = 1

    # if in locals
    if am_local:

        ## parameters to BE FILLED
        path_osmose_dataset = '/home/cazaudo/Desktop/new_git_2/osmose_dataset_sample/' # put here the root path of the dataset folder, eg '/home/cazaudo/Desktop/new_git_2/osmose_dataset_sample/'

        analysis_fs = 240
        dataset_ID = 'gliderSPAms_sample1'

        maxtime_display_spectro = 600
        fileScale_nfft = 2048
        fileScale_winsize = 512
        fileScale_overlap = 90
        colmapspectros = 'jet'
        nber_zoom_levels = 2
        min_color_val = -20
        max_color_val = 20

        nber_wav_to_be_processed = 4

        ## default parameters
        nberAdjustSpectros = 0
        norma_gliding_zscore = ''
        fmin_HighPassFilter = 10
        ind_min = 0
        ind_max = nber_wav_to_be_processed


    # else you are on datarmor
    else:
        path_osmose_dataset = "/home/datawork-osmose/dataset/"
        dataset_ID = sys.argv[1]
        analysis_fs = float(sys.argv[2])
        ind_min = int(sys.argv[3])
        ind_max = int(sys.argv[4])

        path_analysis_fiche = os.path.join(path_osmose_dataset, dataset_ID, 'analysis','analysis_fiche.csv')

        # load needed variables from analysis fiche
        analysis_fiche = pd.read_csv(path_analysis_fiche,header=0)
        fileScale_nfft = analysis_fiche['fileScale_nfft'][0]
        fileScale_winsize = analysis_fiche['fileScale_winsize'][0]
        fileScale_overlap = analysis_fiche['fileScale_overlap'][0]
        colmapspectros = analysis_fiche['colmapspectros'][0]
        nber_zoom_levels = analysis_fiche['nber_zoom_levels'][0]
        min_color_val = analysis_fiche['min_color_val'][0]
        max_color_val = analysis_fiche['max_color_val'][0]
        nberAdjustSpectros = analysis_fiche['nberAdjustSpectros'][0]
        maxtime_display_spectro = analysis_fiche['maxtime_display_spectro'][0]
        norma_gliding_zscore = analysis_fiche['norma_gliding_zscore'][0]
        fmin_HighPassFilter = analysis_fiche['fmin_HighPassFilter'][0]


    # load needed variables from raw metadata
    metadata = pd.read_csv( os.path.join(path_osmose_dataset , dataset_ID , 'raw/metadata.csv') )
    orig_fileDuration = metadata['orig_fileDuration'][0]
    orig_fs = metadata['orig_fs'][0]
    total_nber_audio_files = metadata['nberWavFiles'][0]

    # build a few paths
    path_analysisFolder = os.path.join(path_osmose_dataset, dataset_ID, 'analysis')



    if nberAdjustSpectros==0:
        folderName_audioFiles = str(maxtime_display_spectro)+'_'+str(int(analysis_fs))
    else:
        folderName_audioFiles = str(int(orig_fileDuration))+'_'+str(int(orig_fs))
        
    
    path_audio_files = os.path.join(path_osmose_dataset, dataset_ID, 'raw/audio', folderName_audioFiles )
    
    path_output_spectrograms = os.path.join(path_analysisFolder, 'spectrograms', str(maxtime_display_spectro)+'_'+str(int(analysis_fs)))    
    
    
    path_summstats = os.path.join(path_analysisFolder,'normaParams',folderName_audioFiles)    
    
    
    # shitty stuff ! in case where norma_gliding_zscore is the empty list , need to convert it from str to list, a bit weird, to be changed..
    # isnorma is 1 when it is a time period
    isnorma = 0
    for cc in ['D','M','H','S','W']:
        if cc in norma_gliding_zscore:
            isnorma = 1
                    
#     if isinstance(analysis_fiche['norma_gliding_zscore'][0],str):
#         exec('norma_gliding_zscore=' + analysis_fiche['norma_gliding_zscore'][0] )            
    if isnorma:
        
        if nberAdjustSpectros>0:
            dura=orig_fileDuration
        else:
            dura=maxtime_display_spectro
        
        average_over_H = int(round(pd.to_timedelta(norma_gliding_zscore).total_seconds() / dura))
        print(average_over_H)
        
        print(os.path.join(path_summstats,'summaryStats*'))
        
        print(glob.glob( os.path.join(path_summstats,'summaryStats*') ))
        
        df=pd.DataFrame()
        for dd in glob.glob( os.path.join(path_summstats,'summaryStats*') ):        
            df = pd.concat([ df , pd.read_csv(dd,header=0) ])

        print(df.head)
            
        df['mean_avg'] = df['mean'].rolling(average_over_H, min_periods=1).mean()
        df['std_avg'] = df['std'].rolling(average_over_H, min_periods=1).std()

#         nn = os.path.join(path_summstats,'summaryStats_'+norma_gliding_zscore+'.csv')

#         df.to_csv(nn,index=False)
                
        summStats = df#pd.read_csv(os.path.join(path_summstats,'summaryStats_'+norma_gliding_zscore+'.csv'))
    
    
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
    
    if autofind_minw:
        name_folder = 'nfft=' + str(fileScale_nfft) + ' winsize=' + str(fileScale_winsize) + \
                      ' overlap=' + str(int( fileScale_overlap)) + ' cvr=autofind_min:0'
    else:
        name_folder = 'nfft=' + str(fileScale_nfft) + ' winsize=' + str(fileScale_winsize) + \
                      ' overlap=' + str(int( fileScale_overlap)) + ' cvr='+ str(min_color_val) +':0'
        
    

    
    if not os.path.exists(os.path.join(path_output_spectrograms ,'spectrograms.csv')):
        data = {'name' :name_folder , 'nfft':fileScale_nfft , 'window_size' : fileScale_winsize , 'overlap' : fileScale_overlap /100 , 'zoom_level': 2**(nber_zoom_levels-1) , 'desc':''}
        df = pd.DataFrame.from_records([data])
        df.to_csv( os.path.join(path_output_spectrograms ,'spectrograms.csv') , index=False)     
    
    

    # when you want to adjust your spectro params, we work with original wav files because in this mode we segment and resample directly in this script
    if nberAdjustSpectros>0:
        ind_max = int(np.ceil(nberAdjustSpectros / np.round(orig_fileDuration / maxtime_display_spectro) ) )


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
           

    if not am_local:
        os.remove( os.path.join(path_analysisFolder, 'ongoing_pbsFiles', 'pbs_spectroGenerationFileScale_' + str(ind_min) + '.pbs') )

    if nberAdjustSpectros>0:
        for ff in glob.glob(os.path.join(path_audio_files, 'temp_*')):
            os.remove( ff )
            
            
            
            
            
            
            
            
        
