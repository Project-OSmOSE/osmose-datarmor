# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 12:22:13 2022

@author: gabri
"""
'''
Include Functions to splitt DEV (developpment) and EVAL (evaluation) sets from already existing dataset for network (as ALL_annotation.csv)

Librairies : Please, check the file "requierments.txt"

functions here : 
        - SplitDataset_DevEval_main() : MAIN

Paths needed : 
        - path_osmose_dataset : path of the datset with OSmOSE architecture
        - codes_path : path with all the functions needed 
'''

import numpy as np
import os
import pandas as pd
import random
import sys
from tqdm import tqdm


with open('path_osmose_dataset.txt') as f:
    path_osmose_dataset = f.readlines()[0]
    
with open('path_codes.txt') as f:
    codes_path = f.readlines()[0]
sys.path.append(codes_path)


#%% MAIN
def SplitDataset_DevEval_main(dataset_ID, LenghtFile, Fs, Task_ID, BM_Name, SelectionMethod, DeveloppmentDatasetPortion, SplitName, LenghtSequence):

    '''
        INPUTS :
            - dataset_ID : name of dataset (already existing)
            - Task_ID : name of task (already existing)
            - BM_Name : name of Benchmark (already existing)
            - LengthFile : duration of input files (in second)
            - Fs : Sampling Rate (in Hz)
            - SelectionMethod : Method of selection of file for developpment set (for now only 'FullyRandom')
            - DeveloppmentDatasetPortion : Proportion of developpment set from all files (between 0 and 1)
            - SplitName : label you want to give to save the split 
           
        '''
    #%% Dataset Path
    with open('path_osmose_dataset.txt') as f:
        path_osmose_dataset = f.readlines()[0]
    base_path = path_osmose_dataset + dataset_ID + os.sep
     
    #%% Import Metadata, Set Variables
    # general path of the dataset location
    base_path = path_osmose_dataset + dataset_ID + os.sep
    folderName_audioFiles = str(LenghtFile)+'_'+str(int(Fs))
    
    Annot_metadata = np.load(base_path+'AI'+os.sep + Task_ID + os.sep + BM_Name + os.sep + folderName_audioFiles + os.sep + 'info_datasplit' + os.sep +  'Annotations_metadata.npz')
    train_df = pd.read_csv(base_path+'AI'+os.sep + Task_ID + os.sep + BM_Name + os.sep + folderName_audioFiles + os.sep + 'info_datasplit' + os.sep +    'ALLannotations.csv')
    LabelsList = Annot_metadata['LabelsList']
    NbFile = len(train_df)
    
    if not os.path.exists(base_path+'AI'+os.sep + Task_ID + os.sep + BM_Name + os.sep + folderName_audioFiles + os.sep + 'info_datasplit' + os.sep + SplitName + os.sep):
        os.makedirs(base_path+'AI'+os.sep + Task_ID + os.sep + BM_Name + os.sep + folderName_audioFiles + os.sep + 'info_datasplit' + os.sep + SplitName + os.sep)
    
    #%% Separate Dev, Validation
    # Label to be saved in the .csv files
    columns_name = list(LabelsList.copy())
    columns_name.insert(0, "filename")
    
    
    
    #%% Split Method 
        # For now, only 'FullyRandom' : all files are mixed and then we split according to DeveloppmentDatasetPortion
    if  SelectionMethod == 'FullyRandom':
        random_sequence = list(np.linspace(0,NbFile-1, NbFile, dtype=int))
        random.shuffle(random_sequence)
        
        DevSetArg = (random_sequence[:int(DeveloppmentDatasetPortion*NbFile)])
        EvalSetArg = (random_sequence[int(DeveloppmentDatasetPortion*NbFile):])
        
    if  SelectionMethod == 'RandomBySequence':
        NbFileInSequence = round(LenghtSequence/LenghtFile)
        NbSequence = round(NbFile/NbFileInSequence)
        
        ord_sequence = list(np.linspace(0,NbFile-1, NbFile, dtype=int))
        random_start_clust = list(np.arange(0,NbFile-1, 10, dtype=int))
        random.shuffle(random_start_clust)# = sorted(random.choices(ord_sequence, k=round(DeveloppmentDatasetPortion*NbSequence)))
        DevSetArg = []
        
        for file_id in tqdm(random_start_clust[:int(DeveloppmentDatasetPortion*NbSequence)]):
            for i in range(10):
                if (file_id + i) not in DevSetArg:
                    DevSetArg.append(file_id + i)
                    
        EvalSetArg = []
        for file_id in tqdm(ord_sequence):   
            if file_id not in DevSetArg:
                EvalSetArg.append(file_id)
    
        EvalSetArg = sorted(EvalSetArg)
        DevSetArg = sorted(DevSetArg)
    
    #%%
    
    # Create Dataframe that will be save as .csv for devellopment annotations
    train_df_dev = pd.DataFrame(columns=columns_name)
    train_df_dev["filename"] = [[]] * len(DevSetArg)
    
    # Get files for devellopment annotations
    for i in range(len(DevSetArg)):
        train_df_dev['filename'][i] = train_df['filename'][DevSetArg[i]]
        for label in LabelsList:
            train_df_dev[label][i] =  train_df[label][DevSetArg[i]]
    
    # Create Dataframe that will be save as .csv for evaluation annotations
    train_df_eval = pd.DataFrame(columns=columns_name)
    train_df_eval["filename"] = [[]] * len(EvalSetArg)
    
    # Get files for evaluation annotations
    for i in range(len(EvalSetArg)):
        train_df_eval['filename'][i] = train_df['filename'][EvalSetArg[i]]
        for label in LabelsList:
            train_df_eval[label][i] =  train_df[label][EvalSetArg[i]]
    
    # Reorganize Dataframe
    train_df_eval.dropna(subset = [columns_name[1]], inplace=True)
    train_df_dev.dropna(subset = [columns_name[1]], inplace=True)
    
    # Save dataframe as .csv
    train_df_dev.to_csv(base_path+'AI'+os.sep + Task_ID + os.sep + BM_Name + os.sep + folderName_audioFiles + os.sep + 'info_datasplit' + os.sep + SplitName + os.sep + 'DEVannotations.csv', index = False, header=True)
    train_df_eval.to_csv(base_path+'AI'+os.sep + Task_ID + os.sep + BM_Name + os.sep + folderName_audioFiles + os.sep + 'info_datasplit' + os.sep + SplitName + os.sep + 'EVALannotations.csv', index = False, header=True)
    
    
    #%% Print number of files in all subset and % of positive files for each label
    print('DEV :')
    
    print('Nombre de fichier : ', len(train_df_dev))
    for label in LabelsList:
        x = 100*np.sum(train_df_dev[label])/len(train_df_dev)
        print(label,' -> pourcentage de Positif : ', "{:10.3f}".format(x),'%')
    
    print('EVAL :')
    
    print('Nombre de fichier : ', len(train_df_eval))
    for label in LabelsList:
        x = 100*np.sum(train_df_eval[label])/len(train_df_eval)
        print(label,' -> pourcentage de Positif : ', "{:10.3f}".format(x),'%')
        
    print(' ')
    print('Split is done ! You now can train a network on the development set and apply it on the evaluation set.')
    
    
   
    
    
