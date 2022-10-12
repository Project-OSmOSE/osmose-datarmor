# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 13:56:55 2022

@author: gabri
"""
'''
Include Functions to apply a trainned model on the evaluation set and compute evaluation metrics to evaluate the network

Librairies : Please, check the file "requierments.txt"

functions here : 
        - ApplyModelOnEvalSet_main() : MAIN
        - LoadModelHyperParameters() : Load the models hyperparameters, weight, architecture, ...

Paths needed : 
        - path_osmose_dataset : path of the datset with OSmOSE architecture
        - codes_path : path with all the functions needed
            - ClasseDatasetForTorch.py
            - Transfer_Learning/Functions.py
            - EvaluationMetrics_ComputeAndPlot.py
'''



#%% Import 
import numpy as np
import os
import pandas as pd
import sys
import torch
from tqdm import tqdm

with open('path_osmose_dataset.txt') as f:
    path_osmose_dataset = f.readlines()[0]
    
with open('path_codes.txt') as f:
    codes_path = f.readlines()[0]
sys.path.append(codes_path)
from ClasseDatasetForTorch import ClassDataset
from EvaluationMetrics_ComputeAndPlot import ComputeEvaluationMetrics, plot_PR_curve, plot_ROC_curve, plot_DET_curve, plot_COST_curve
from TransferLearning_Functions import transform


#%% MAIN

def LoadModelHyperParameters(dataset_ID, Task_ID, BM_Name, LengthFile, Fs, Version_name):
    
    '''
        INPUTS :
            - dataset_ID : name of dataset (already existing)
            - Task_ID : name of task (already existing)
            - BM_Name : name of Benchmark (already existing)
            - LengthFile : duration of input files (in second)
            - Fs : Sampling Rate (in Hz)
            - Version_name : Name of the detection network that will be trainnned 
            
        OUTPUTS :
            - Dyn : array with minimum and maximum levels for the spectrograms (in dB) 
            - batch_size : Number of file in one batch
            - learning_rate : learning rate
            - ModelName : name of the reference model to be used (one in already existing list, check in Transfer_Learning/Functions.py file)
            - use_pretrained : True or False if you want to initialize your model with pre-trainned weight - please check PyTorch documentation
            - TrainSetRatio : Ratio between 0 and 1 for the train set from the dev set
            - SplitName : label the Dev/Eval split to use

        '''
    
    with open('path_osmose_dataset.txt') as f:
        path_osmose_dataset = f.readlines()[0]
        
    folderName_audioFiles = str(LengthFile)+'_'+str(int(Fs))

    base_path = path_osmose_dataset + dataset_ID + os.sep + 'AI' + os.sep + Task_ID + os.sep + BM_Name + os.sep + folderName_audioFiles + os.sep + 'models'
     
    Parameters = np.load(base_path + os.sep + Version_name + os.sep + 'hyper_parameters' + os.sep + Version_name + '_hyperparameters.npz')
    
    Dyn = Parameters['Dyn']
    batch_size = Parameters['batch_size']
    learning_rate = Parameters['learning_rate']
    LabelsList = Parameters['LabelsList']
    ModelName = Parameters['ModelName']
    use_pretrained = Parameters['use_pretrained']
    SplitName = Parameters['SplitName']
    TrainSetRatio = Parameters['TrainSetRatio']
    return Dyn, batch_size, learning_rate, LabelsList, ModelName, use_pretrained, SplitName, TrainSetRatio
        

def ApplyModelOnEvalSet_main(dataset_ID, Task_ID, BM_Name, LengthFile, Fs, Version_name, Dyn, LabelsList, SplitName):

    '''
        INPUTS :
            - dataset_ID : name of dataset (already existing)
            - Task_ID : name of task (already existing)
            - BM_Name : name of Benchmark (already existing)
            - LengthFile : duration of input files (in second)
            - Fs : Sampling Rate (in Hz)
            - Version_name : Name of the detection network that will be trainnned 
            - Dyn : array with minimum and maximum levels for the spectrograms (in dB) 
            - LabelsList : List of label to be detected
            - SplitName : label the Dev/Eval split to use
        '''    

    #%% DEVICE 
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    
    base_path = "E:/PhD/Datasets/Glider/"
    
    #%% Dataset Path
    with open('path_osmose_dataset.txt') as f:
        path_osmose_dataset = f.readlines()[0]
        
    base_path = path_osmose_dataset + dataset_ID + os.sep
    
    folderName_audioFiles = str(LengthFile)+'_'+str(int(Fs))
        
        
    csv_anaylsis_fiche = pd.read_csv(path_osmose_dataset+dataset_ID+os.sep+'analysis'+os.sep + 'spectrograms' + os.sep + folderName_audioFiles + os.sep +'spectrograms.csv')
    
    #Load Parameters
    Nfft = csv_anaylsis_fiche['nfft'][0]
    winsize = csv_anaylsis_fiche['window_size'][0]
    overlap = csv_anaylsis_fiche['overlap'][0]
    name_folder = csv_anaylsis_fiche['name'][0]
    cvr = csv_anaylsis_fiche['cvr_min'][0]
    cvr_max = csv_anaylsis_fiche['cvr_max'][0]
    
    # problem with ':' in file name with wondows
    name_folder_windows = name_folder.replace(':','_')
    
    if os.path.exists(path_osmose_dataset + dataset_ID + os.sep + 'analysis' + os.sep + 'spectrograms_mat' + os.sep + folderName_audioFiles + os.sep + name_folder):
        folderName_Param_Spectro = name_folder
    
    elif os.path.exists(path_osmose_dataset + dataset_ID + os.sep + 'analysis' + os.sep + 'spectrograms_mat' + os.sep + folderName_audioFiles + os.sep + name_folder_windows):
        folderName_Param_Spectro = name_folder_windows
        
    else: print('Path issu, please contact OSmOSE team')    

    path_npz_files =path_osmose_dataset + dataset_ID + os.sep + 'analysis' + os.sep + 'spectrograms_mat' + os.sep + folderName_audioFiles + os.sep + folderName_Param_Spectro
    
    #%% Load Model and annotation
    model_path = base_path+'AI'+os.sep + Task_ID + os.sep + BM_Name + os.sep + folderName_audioFiles + os.sep + 'models' + os.sep + Version_name
    model = torch.jit.load(model_path + os.sep + 'model_state' + os.sep + Version_name + '_Scripted_model.pt')
    model.eval()
    
    raw_data_path = path_npz_files
    CSV_annotations_path = base_path+'AI' + os.sep + Task_ID + os.sep + BM_Name + os.sep + folderName_audioFiles + os.sep + 'info_datasplit' + os.sep + SplitName + os.sep + 'EVALannotations.csv'
    dataset = ClassDataset(raw_data_path,CSV_annotations_path,Dyn = Dyn, transform=transform)
    
    #%% Apply Model On All Dataset
    
    labels = np.zeros([len(dataset), len(LabelsList)])
    outputs = np.zeros([len(dataset), len(LabelsList)])
        
    for i in tqdm(range(len(dataset))):
        
        #get data and label
        imgs, label = dataset.__getitem__(i)
        
        #to device
        imgs = imgs.to(device)
        labels_batch = label.to(device)
        #apply model
        outputs_batch = model(imgs[None,:].float())
        
        labels[i] = labels_batch.cpu().detach().numpy()[0]
        outputs[i] = outputs_batch.cpu().detach().numpy()
    
    #%% Compute Evaluation Index
    Recall, Precision, FP_rate, TP_rate, FN_rate, NormalizedExpectedCost, ProbabilityCost = ComputeEvaluationMetrics(LabelsList, labels, outputs)
    
    #%% PLOT PR DET ROC Curve
    np.savez(model_path + os.sep + 'train_curves' + os.sep + Version_name + '_EvaluationMetrics_DATA.npz', Recall=Recall, Precision=Precision, FP_rate=FP_rate, TP_rate=TP_rate, FN_rate=FN_rate, NormalizedExpectedCost=NormalizedExpectedCost, ProbabilityCost=ProbabilityCost, LabelsList=LabelsList, labels=labels, outputs=outputs)
    for id_specie in range(len(LabelsList)):
        
        savepath = model_path + os.sep + 'train_curves' + os.sep + str(Version_name) + '_' + str(LabelsList[id_specie]) + '_PRCurves.png'
        plot_PR_curve(Recall[:,id_specie], Precision[:,id_specie], savepath = savepath, color='b', xlim=[0,1], ylim=[0,1], figsize=(4,4))
        
        savepath = model_path + os.sep + 'train_curves' + os.sep + str(Version_name) + '_' + str(LabelsList[id_specie]) + '_ROCCurves.png'
        plot_ROC_curve(FP_rate[:,id_specie], TP_rate[:,id_specie], savepath = savepath, color='b', xlim=[0,1], ylim=[0,1], figsize=(4,4))
   
        savepath = model_path + os.sep + 'train_curves' + os.sep + str(Version_name) + '_' + str(LabelsList[id_specie]) + '_DETCurves.png'
        plot_DET_curve(FP_rate[:,id_specie], FN_rate[:,id_specie], savepath = savepath, color='b', xlim=[0.005,0.8], ylim=[0.005,0.8], figsize=(4,4))
        
        savepath = model_path + os.sep + 'train_curves' + os.sep + str(Version_name) + '_' + str(LabelsList[id_specie]) + '_CostCurves.png'
        plot_COST_curve(ProbabilityCost, NormalizedExpectedCost[:,id_specie,:].T, savepath = savepath, color='b', xlim=[0,1], ylim=[0,0.5], figsize=(4,4))
       
        
