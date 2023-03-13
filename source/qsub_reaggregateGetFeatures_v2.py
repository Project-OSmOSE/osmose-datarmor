
from module_soundscape import *


import sys
import glob

import numpy as np
import pickle
import os
import pandas as pd
from datetime import datetime

def append_new_line(file_name, text_to_append):
    """Append given text as a new line at the end of file"""
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)
        
        

def flatten(nasted_list):
    list_of_lists = []
    for item in nasted_list:
        list_of_lists.extend(item)
    return list_of_lists
    

def get_list_label_of_timeres(timeres):

    weekday_list=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]    
    month_list = ['January', 'February', 'March', 'April', 'May', 'June', 'July','August', 'September', 'October', 'November', 'December']

    if 'w' in timeres:
        list_label = weekday_list
    elif 'm' in timeres:
        list_label = month_list
    else:
        list_label = None    
    
    return list_label
    
    
def filter_to_timeres(x,timeres):

    return  datetime.strptime(str(x)[:19], '%Y-%m-%d %H:%M:%S').strftime(timeres)



    
path_osmose_dataset = "/home/datawork-osmose/dataset/"



dataset_ID = sys.argv[1]
analysis_fs = float(sys.argv[2])

path_analysisFolder = os.path.join(path_osmose_dataset, dataset_ID, 'analysis')

# load needed variables from analysis fiche
analysis_fiche = pd.read_csv( os.path.join(path_analysisFolder,'analysis_fiche.csv'),header=0)
exec('newFeatures_timescale=' + analysis_fiche['newFeatures_timescale'][0])
normalizeByMax_welch = False # analysis_fiche['normalizeByMax_welch'][0]


if str(analysis_fiche['aux_variable'][0])=='nan':
    aux_variable = ['']    
else:
    aux_variable = analysis_fiche['aux_variable'][0].split('-')

nfft=analysis_fiche['datasetScale_nfft'][0] 


# loop over the different time resolutions desired
for time_res in newFeatures_timescale:

    path_output_newFeatures = os.path.join(path_analysisFolder, 'soundscape','raw_welch',time_res)
    
    total_time = []
    total_welch = np.empty((1, int(nfft/ 2 + 1)))

    ct=0
    for gg in sorted(glob.glob(os.path.join(path_output_newFeatures,'*.pkl'))):

        [welch_values,ind_time,frequencies,tol_values,frequency_TOL] = pickle.load(open(gg, 'rb'))
        
        # bricolage sale : parce qu'on ne connait pas a priori length de frequency_TOL ...
        if ct==0:
            total_tol = np.empty((1, len(frequency_TOL) ))
        
        ct+=1

        total_welch = np.vstack((total_welch, welch_values))
        total_tol = np.vstack((total_tol, tol_values))
        
        total_time.append(ind_time)

    # remove first elements because of np.empty , which creates a weird floating number
    total_welch = total_welch[1:,:]
    total_tol = total_tol[1:,:]    

    total_time = flatten(total_time)
    
    if normalizeByMax_welch:
        total_welch = total_welch - np.min(total_welch)

    with open( os.path.join(path_output_newFeatures,'complete.pkl'), 'wb') as handle:
        pickle.dump([total_welch,total_time,frequencies,total_tol,frequency_TOL], handle, protocol=4)

    
        
exec('warp_timePeriod=' + analysis_fiche['warp_timePeriod'][0])
exec('sequential_timePeriod=' + analysis_fiche['sequential_timePeriod'][0])

df_total_welch = pd.DataFrame(data = {'total_welch' : [total_welch[i] for i in range(len(total_welch))],'timestamp':total_time})
df_total_welch.set_index('timestamp', inplace=True, drop=True)

  
# filter welch spectra, ie keep only coefs between fmin and fmax
df_total_welch['total_welch_filtered']=df_total_welch['total_welch'].apply(lambda x: x[np.arange(np.argmin(abs(frequencies-analysis_fiche['fmin'][0])),np.argmin(abs(frequencies-analysis_fiche['fmax'][0])))])

# frequencies_filtered = frequencies[np.arange(np.argmin(abs(frequencies-analysis_fiche['fmin'][0])),np.argmin(abs(frequencies-analysis_fiche['fmax'][0])))]

# step_spectro = analysis_fs / (2 * (nfft // 2)) if np.mod(nfft, 2) == 0 else analysis_fs / (2 * (nfft // 2 + 1))
# end_spectro = analysis_fs / 2 + 1 / (nfft // 2) if np.mod(nfft, 2) == 0 else analysis_fs / 2 + 1 / (nfft // 2 + 1)
    
    
warp_yes_no = np.hstack((np.ones(len(warp_timePeriod)).astype(int) , np.zeros(len(sequential_timePeriod)).astype(int))).tolist()
    
# WARPING figures
ct=0
for wp in flatten( [warp_timePeriod , sequential_timePeriod]) :
    
    if warp_yes_no[ct]: # warp mode
        df_total_welch['group_bs'] = df_total_welch.index.to_series().apply(filter_to_timeres,timeres=wp)
        filename_type='warping'
    else: # sequential mode
        if wp=='all':
            df_total_welch['group_bs'] = 'all'
        else:
            df_total_welch['group_bs'] = df_total_welch.index.round(freq=wp)
        filename_type='sequential'
   

    big_group = np.unique(df_total_welch['group_bs'])

    print('wp:',wp)
    big_group_label = get_list_label_of_timeres(wp)
    print('big_group_label',big_group_label)
    if big_group_label==None:
        big_group_label=big_group   
    print('big_group',big_group)
    print('big_group_label',big_group_label)

    # for each big group (eg day)
    for ind_bg in range(len(big_group)):
        
        if warp_yes_no[ct]:           
            filna = big_group_label[int(ind_bg)-1] # substract 1 to make it an index          
            bg = big_group[int(ind_bg)-1]            
        else:
            bg = big_group[int(ind_bg)-1]
            filna = str(bg)[0:13].replace(' ','T').replace('-','')

        cur_time = df_total_welch.loc[df_total_welch['group_bs'] == bg].index  

        # format xticklabels with timestamps
        format_date = '%Y-%m-%d %H:%M'
        date = []
        for cc in cur_time:
            date.append(pd.to_datetime(cc).strftime(format_date))   

            

        # median averaged of all welch within current group, ie get only one SPL value per timestep, eg used by plot_timeSPL
        cur_SPL = df_total_welch[ df_total_welch['group_bs']==bg ]['total_welch_filtered'].apply(lambda x: np.median(x)).values   
        if cur_SPL.shape[0]>1e6:
            
            
            content = "Problem in sequential soundscape figures : you have way too many welch spectra into your timeseries : "+str(cur_SPL.shape[0])
            
            
            if os.path.exists(os.path.join(path_analysisFolder, 'log_error.txt')):
                
                append_new_line(os.path.join(path_analysisFolder, 'log_error.txt'), content)
                
            else:
            
                with  open(os.path.join(path_analysisFolder, 'log_error.txt'), "w") as file:
                    file.write(content)
                    file.close()    
                            
            continue
            

        if analysis_fiche['plot_timeSPL'][0]:
            
            if analysis_fiche['plot_timeSPL'][0]:
                if wp == 'all':
                    ff = filename_type+'_'+wp+'.png'
                else:
                    ff = filename_type+'_'+wp+'_'+filna+'.png'  
            
            plot_timeSPL(cur_SPL,date,os.path.join(path_analysisFolder, 'soundscape','timeSPL' , ff))


        # aggregated welch in current group
        cur_welch = np.vstack(df_total_welch.loc[df_total_welch['group_bs'] == bg, 'total_welch'].values)

        if analysis_fiche['plot_EPD'][0]:
            if wp == 'all':
                ff = filename_type+'_'+wp+'.png'
            else:
                ff = filename_type+'_'+wp+'_'+filna+'.png'                
                
            plot_EPD(cur_welch,date,frequencies,os.path.join(path_analysisFolder, 'soundscape','EPD' , ff))

            
        if (analysis_fiche['plot_LTAS'][0]):

            screen_res_pixel = 2000
                        
            cur_welch = np.vstack(df_total_welch.loc[df_total_welch['group_bs'] == bg, 'total_welch'].values)
            ind_av = round(cur_welch.shape[0] / screen_res_pixel)
            
            print(cur_welch.shape)
            
            if cur_welch.shape[0]>screen_res_pixel:
                            
                if ind_av>0: # you must have more than screen_res_pixel in cur_welch

                    mm=cur_welch[0::ind_av,:]
                    bb=cur_welch[1::ind_av,:]

                    if mm.shape[0]>bb.shape[0]:
                        mm=mm[:-1,:]
                    elif bb.shape[0]>mm.shape[0]:
                        bb=bb[:-1,:]

                    cur_welch = 0.5*(mm + bb)

                    cur_ind_time=df_total_welch.loc[df_total_welch['group_bs'] == bg].index[0::ind_av]

                else:

                    cur_ind_time = df_total_welch.loc[df_total_welch['group_bs'] == bg].index  


                # format xticklabels with timestamps
                format_date = '%Y-%m-%d %H:%M'
                date = []
                for cc in cur_ind_time:
                    date.append(pd.to_datetime(cc).strftime(format_date))       

                if wp == 'all':
                    ff = filename_type+'_'+wp+'.png'
                else:
                    ff = filename_type+'_'+wp+'_'+filna+'.png'  

                plot_LTAS(cur_welch,date,frequencies,os.path.join(path_analysisFolder, 'soundscape','LTAS' , ff))
                
                
            else:
                
                print('Not enough welch spectra in your sequential time period to plot a LTAS')
                

        
    ct+=1

            
## PLOT the recurBox figure 
if analysis_fiche['plot_recurBOX'][0]:

    # here because problem using exec within a function https://stackoverflow.com/questions/41100196/exec-not-working-inside-function-python3-x
    analysis_fiche = pd.read_csv(os.path.join(path_analysisFolder, 'analysis_fiche.csv'), header=0)
    exec('warp_timePeriod=' + analysis_fiche['bigwarp_timePeriod_recurBox'][0])
    warp_timePeriod = warp_timePeriod[0] # make warp_timePeriod a str from list    
    exec('small_timeres=' + analysis_fiche['smallwarp_timePeriod_recurBox'][0])
    small_timeres = small_timeres[0] # make warp_timePeriod a str from list
    
    
    
    plot_recurBOX(total_welch,total_time,path_analysisFolder,os.path.join(path_analysisFolder, 'soundscape','recurBOX'),warp_timePeriod,small_timeres)
    
    
if os.path.exists(os.path.join(path_analysisFolder, 'ongoing_pbsFiles', 'pbs_computeFeaturesDatasetScale.pbs')):
    os.remove( os.path.join(path_analysisFolder, 'ongoing_pbsFiles', 'pbs_computeFeaturesDatasetScale.pbs'))

    
os.remove( os.path.join(path_analysisFolder, 'ongoing_pbsFiles', 'pbs_reaggregateGetFeatures_0.pbs') )

