import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from datetime import datetime
import matplotlib.cm as cm


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

dataset_base_path = "/home/datawork-osmose/dataset/"

my_dpi = 100
fact_x = 1
fact_y = 1



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
    # class is month name
    return  datetime.strptime(str(x)[:19], '%Y-%m-%d %H:%M:%S').strftime(timeres)


def plot_recurBOX(total_welch,ind_time,path_analysisFolder,path_output_newFeatures,warp_timePeriod,small_timeres):
    


    # here total_welch not needed in the DataFrame but cannot remove it , not clear why
    df_total_welch = pd.DataFrame(data = {'total_welch' : [total_welch[i] for i in range(len(total_welch))],'timestamp':ind_time})
    df_total_welch.set_index('timestamp', inplace=True, drop=True)

    df_total_welch['group_bs'] = df_total_welch.index.to_series().apply(filter_to_timeres,timeres=warp_timePeriod)
    df_total_welch['group_ss'] = df_total_welch.index.to_series().apply(filter_to_timeres,timeres=small_timeres)

    small_group = np.sort(np.unique(df_total_welch['group_ss']))
    big_group = np.sort(np.unique(df_total_welch['group_bs']))



    small_group_label = get_list_label_of_timeres(small_timeres)
    if small_group_label==None:# case where neither day or month, eg hour that is already numeric
        small_group_label=small_group
        
    big_group_label = get_list_label_of_timeres(warp_timePeriod)
    if big_group_label==None:
        big_group_label=big_group
         
    fig, ax = plt.subplots(1, 1, figsize=(fact_x * 1800 / my_dpi, fact_y * 512 / my_dpi),
                           dpi=my_dpi, constrained_layout=True)

    vec_col = cm.get_cmap('jet', len(small_group))

    coefmul= len(small_group) * 3 / 24 # arbitrary coefficient for barplot interspaces

    decal=0
    ct=0
    data=[]
    vmax=0
    vmin=10e6
    # for each small group (eg hour)   
    ct_col=0
    for sg in small_group:

        data=[]
        # for each big group (eg day)
        for bg in big_group:

            valdd=df_total_welch[ (df_total_welch['group_bs']==bg) &  (df_total_welch['group_ss']==sg)]['total_welch'].apply(lambda x: np.median(x)).values        

            if len(valdd)>0:        
                data.append(np.mean(valdd))
            else:
                data.append(np.nan)

        # plot data in grouped manner of bar type
        plt.bar(np.arange(len(big_group))*coefmul-1 + decal, data, 0.1, color=vec_col(ct_col))

        vmax = max([vmax , max(data)] )
        vmin = min([vmin , min(data)] )

        decal += 0.1
        ct_col+=1


    plt.xticks(np.arange(len(big_group))*coefmul-1)
    ax.set_xticklabels(big_group_label)
    ax.tick_params(axis='x', rotation=20)
    plt.ylabel("relative SPL (dB)")
    
    print(small_group_label)
    if len(small_group_label)>1:
        plt.legend(small_group_label , loc='best')   
    
    print(vmin)
    print(vmax)
#     if not (np.isinf(vmin) or np.isinf(vmax) or np.isnan(vmin) or np.isnan(vmax)):
#         plt.ylim([vmin*0.95 , vmax*1.01])
    plt.grid()

    if len(warp_timePeriod) == 0:
        nn='recurBox_'+small_timeres+'.png'
    else:
        nn='recurBox_'+small_timeres+'_'+warp_timePeriod+'.png'
        
    fig.savefig(os.path.join(path_output_newFeatures , nn),bbox_inches='tight',dpi=my_dpi)
    plt.close(fig)        

    

    
def plot_timeSPL(cur_welch,date,filename):
    
    
        fig, ax = plt.subplots(1, 1, figsize=(fact_x * 1800 / my_dpi, fact_y * 512 / my_dpi),
                               dpi=my_dpi, constrained_layout=True)

        plt.plot(cur_welch, linewidth=2)     
        
        if len(date)>10:
            int_sep = int(len(date) / 10)
        else:
            int_sep = 1
        
        plt.xticks(np.arange(0, len(date), int_sep), date[::int_sep])
        labels = [l.get_text().split(',')[0].replace('(', '').split('.')[0] for l in ax.get_xticklabels()]
        ax.tick_params(axis='x', rotation=20)   
        plt.autoscale(enable=True, axis='y', tight=True)
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.ylabel("relative SPL (dB)")

        fig.savefig(filename,bbox_inches='tight',dpi=my_dpi)

        plt.close(fig)
        
        
        
        
        
        
def plot_EPD(cur_welch,date,fPSD,filename):

    RMSlevel = 10 * np.log10(np.nanmean(10 ** (cur_welch / 10), axis=0))

    fig, ax = plt.subplots(1, 1, figsize=(fact_x * 1800 / my_dpi, fact_y * 512 / my_dpi),
                           dpi=my_dpi, constrained_layout=True)    

    ax.plot(fPSD, RMSlevel, color='k', label='RMS level')

    percen = [1, 5, 50, 95, 99]
    p = np.nanpercentile(cur_welch, percen, 0, interpolation='linear')
    for i in range(len(p)):
        plt.plot(fPSD, p[i, :], linewidth=2, label='%s %% percentil' % percen[i])

    ax.semilogx()
    plt.legend()    

    plt.autoscale(enable=True, axis='y', tight=True)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.ylabel("relative SPL (dB)")
    plt.xlabel("Frequency (Hz)")

    fig.savefig(filename,bbox_inches='tight',dpi=my_dpi)

    plt.close(fig)
    
    
def plot_LTAS(cur_welch,date,fPSD,filename):
    
    fig, ax = plt.subplots(1, 1, figsize=(fact_x * 1800 / my_dpi, fact_y * 512 / my_dpi),
                           dpi=my_dpi, constrained_layout=True)

#     x, y = np.mgrid[slice(0, cur_welch.shape[0], 1), fpsd]

    im = ax.pcolormesh( np.arange(0,cur_welch.shape[0]) , fPSD , cur_welch.T)

    cb=plt.colorbar(im,ax=ax)
    #         cb.set_label(label='PSD (relative dB)')

    int_sep = int(len(date) / 10)
    plt.xticks(np.arange(0, len(date), int_sep), date[::int_sep])
    labels = [l.get_text().split(',')[0].replace('(', '').split('.')[0] for l in ax.get_xticklabels()]
    ax.tick_params(axis='x', rotation=20)

    ax.set_ylabel("Frequency (Hz)")

    fig.savefig(filename,bbox_inches='tight',dpi=my_dpi)

    plt.close(fig)

        

