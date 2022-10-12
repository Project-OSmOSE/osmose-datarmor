# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 12:22:13 2022

@author: gabri
"""
'''
Include Functions to Train Network on already existing DEVset on datarmore

Librairies : Please, check the file "requierments.txt"

functions here : 
        - TrainNetwork_main() : MAIN
        - train() : loop over train and test loader

Paths needed : 
        - path_osmose_dataset : path of the datset with OSmOSE architecture
        - codes_path : path with all the functions needed 
            - ClasseDatasetForTorch.py
            - Transfer_Learning/Functions.py
'''

import numpy as np
from torch.utils.data import DataLoader
import os
import pandas as pd
import sys
import torch

with open('path_osmose_dataset.txt') as f:
    path_osmose_dataset = f.readlines()[0]
    
with open('path_codes.txt') as f:
    codes_path = f.readlines()[0]
sys.path.append(codes_path)
from ClasseDatasetForTorch import ClassDataset
from TransferLearning_Functions import initialize_model, transform, bce_loss


#Train Loop
def train(device, model_ft, optimizer, num_epochs, train_loader, test_loader, weight):
    
        '''
        INPUTS :
            - device : device cpu or gpu to optimize computation
            - model_ft : torch model to be trainned
            - num_epochs : number of iteration in the trainning
            - train_loader : torch dataset with trainning files and labels for train
            - test_loader : torch dataset with trainning files and labels for test
            - weight : weight on each label - now, there are at put ones 
            
        OUTPUTS :
            - loss_tab_train : loss value for test at each iteration
            - loss_tab_test : loss value for train at each iteration
        '''
    
        #initialize loss tab for plot
        loss_tab_train = []
        loss_tab_test = []
        model_ft.train()
        for epoch in range(num_epochs):
            model_ft.train()
            ite = 1
            loss_sum_train = 0
            #Loop For over train set
            for imgs, labels in train_loader:
                #load data and label, send them to device
                imgs = imgs.to(device)
                labels = labels.to(device)
                #apply model
                outputs = model_ft(imgs.float())
                #compute loss and backward (gradient)
                loss = bce_loss(outputs, labels, weight)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #loss mean over the epoch
                loss_sum_train += loss.item()
                PrintedLine = f"Epoch TRAIN [{epoch}/{num_epochs}]" + '  -- Loss = '+ str(loss_sum_train/ite) + '  --  ' + f"iteration [{ite}/{len(train_loader)}]" 
                sys.stdout.write('\r'+PrintedLine)
                ite += 1
            loss_tab_train.append(loss_sum_train/ite)
            
            print('  ')
            
            #TEST - SAME AS PREVIEWS ONE WITHOUT BACKWARD
            model_ft.eval()
            ite = 1
            loss_sum_test = 0
            for imgs, labels in test_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model_ft(imgs.float())
                loss = bce_loss(outputs, labels, weight)
                loss_sum_test += loss.item()
                PrintedLine = f"Epoch TEST [{epoch}/{num_epochs}]" + '  -- Loss = '+ str(loss_sum_test/ite) + '  --  ' + f"iteration [{ite}/{len(test_loader)}]" 
                sys.stdout.write('\r'+PrintedLine)
                ite += 1
            loss_tab_test.append(loss_sum_test/ite)
            print('  ')
        return loss_tab_train, loss_tab_test

def TrainNetwork_main(dataset_ID, Task_ID, BM_Name, LengthFile, Fs, SplitName, Version_name, ModelName, use_pretrained, TrainSetRatio, batch_size, learning_rate, num_epochs, Dyn):
  
    '''
        INPUTS :
            - dataset_ID : name of dataset (already existing)
            - Task_ID : name of task (already existing)
            - BM_Name : name of Benchmark (already existing)
            - LengthFile : duration of input files (in second)
            - Fs : Sampling Rate (in Hz)
            - SplitName : label the Dev/Eval split to use
            - Version_name : Name of the detection network that will be trainnned 
            - ModelName : name of the reference model to be used (one in already existing list, check in Transfer_Learning/Functions.py file)
            - use_pretrained : True or False if you want to initialize your model with pre-trainned weight - please check PyTorch documentation
            - TrainSetRatio : Ratio between 0 and 1 for the train set from the dev set
            - batch_size : Number of file in one batch
            - learning_rate : learning rate
            - num_epochs : number of iteration in trainning loop
            - Dyn : array with minimum and maximum levels for the spectrograms (in dB) 
        '''
        
        
    #%% DEVICE 
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    #device = 'cpu'
    
    #%% LOAD PARAMETERS AND DEFINE PATHS
    with open('path_osmose_dataset.txt') as f:
        path_osmose_dataset = f.readlines()[0]
        
    #Define some paths
    #General Path of the dataset
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

    
    
    
    # Initialyse some parameters if not specified before
    if Dyn is None:
        Dyn = np.array([cvr, cvr_max])
        
    if batch_size == None:
        batch_size = 10
        
    if learning_rate == None:
        learning_rate = 1e-4
    
    if TrainSetRatio == None:
        TrainSetRatio = 0.9
             
    if num_epochs == None:
        num_epochs = 10
    
    
    
    #path of spectrograms data as .npz
    path_npz_files = path_osmose_dataset + dataset_ID + os.sep + 'analysis' + os.sep + 'spectrograms_mat' + os.sep + folderName_audioFiles + os.sep + folderName_Param_Spectro

    #path for save trainng data
    model_path = path_osmose_dataset + dataset_ID + os.sep + 'AI' + os.sep + Task_ID + os.sep + BM_Name + os.sep + folderName_audioFiles + os.sep + 'models'
    if not os.path.exists(model_path + os.sep + Version_name):
        os.makedirs(model_path + os.sep + Version_name)
        os.makedirs(model_path + os.sep + Version_name + os.sep + 'hyper_parameters')
        os.makedirs(model_path + os.sep + Version_name + os.sep + 'train_curves')
        os.makedirs(model_path + os.sep + Version_name + os.sep + 'model_state')

    #Import some param
    #Annotations
    annot_param = np.load(base_path+'AI'+os.sep + Task_ID + os.sep + BM_Name + os.sep + folderName_audioFiles + os.sep + 'info_datasplit' + os.sep +  'Annotations_metadata.npz')
    #List of labels to be detected
    LabelsList = annot_param['LabelsList']
    #Number of labels
    num_classes = len(LabelsList)
    
    #%% Def AI Param and import model
    print('MODEL INFORMATION : ')
    print(' ')
    shuffle = False
    pin_memory = True
    feature_extract = True
    num_workers = 1
    drop_last = True
    #Useless for now, but it will be possible to change weight in next release
    weight = torch.ones([num_classes]).to(device)

    # Initialize the model for this run
    model_ft, input_size = initialize_model(ModelName, num_classes, feature_extract, use_pretrained=use_pretrained)
    
    #send to device
    model_ft.to(device)
    # Print the model we just instantiated
    print(model_ft)
#%% Import DEV Annotation, Set Variables
    train_df_dev = pd.read_csv(base_path+'AI'+os.sep + Task_ID + os.sep + BM_Name + os.sep + folderName_audioFiles + os.sep + 'info_datasplit' + os.sep +  SplitName + os.sep + 'DEVannotations.csv')
    # Number of files for developpment
    NbFile = len(train_df_dev)

#%% DEF TEST AND TRAIN SET

    '''
    Nb : For the moment, the split is simply done by cutting in one spot, new possibilities will be added
    '''
    
    raw_data_path = path_npz_files
    CSV_annotations_path = base_path+'AI' + os.sep + Task_ID + os.sep + BM_Name + os.sep + folderName_audioFiles + os.sep + 'info_datasplit' + os.sep + SplitName + os.sep + 'DEVannotations.csv'
    dataset = ClassDataset(raw_data_path,CSV_annotations_path, Dyn = Dyn, transform=transform)

    # Created using indices from 0 to train_size.
    train_set = torch.utils.data.Subset(dataset, range(int(TrainSetRatio*NbFile)))
    train_loader = DataLoader(dataset=train_set, shuffle=shuffle, batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory, drop_last=drop_last)
    
    # Created using indices from Train_size to the end.
    test_set = torch.utils.data.Subset(dataset, range(int(TrainSetRatio*NbFile),NbFile))
    test_loader = DataLoader(dataset=test_set, shuffle=shuffle, batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory, drop_last=drop_last)
    
    Dyn = dataset.Dyn
    
    #%% DEFINE PARAM TO BE OPTIMIZED
    
    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)
    
    # Observe that all parameters are being optimized
    optimizer = torch.optim.SGD(params_to_update, lr=learning_rate, momentum=0.9)
    
    print(' ')
    print('TRAINNING : ')
    
    #Launch the training 
    loss_tab_train, loss_tab_test = train(device, model_ft, optimizer, num_epochs, train_loader, test_loader, weight)
  
    print('DONE')
    
    #%% Save Model 
    #save model
    torch.save(model_ft.state_dict(), model_path + os.sep + Version_name + os.sep + 'model_state' + os.sep + Version_name + '_model.pt')
    # save model as script
    model_ft_scripted = torch.jit.script(model_ft) # Export to TorchScript
    model_ft_scripted.save(model_path + os.sep + Version_name + os.sep + 'model_state' + os.sep + Version_name + '_Scripted_model.pt') # Save
    
    np.savez(model_path + os.sep + Version_name + os.sep + 'hyper_parameters' + os.sep + Version_name+ '_hyperparameters.npz', num_epochs=num_epochs, learning_rate=learning_rate, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, num_classes=num_classes, BM_Name=BM_Name, ModelName=ModelName, use_pretrained=use_pretrained, Dyn=Dyn, SplitName=SplitName, TrainSetRatio=TrainSetRatio, LabelsList=LabelsList)
    
    #save metadata as .npz and in a txt file
    metadata_tab = [num_epochs, learning_rate, batch_size, shuffle, num_workers, num_classes, BM_Name, ModelName, use_pretrained, Dyn, SplitName, TrainSetRatio, LabelsList]
    metadata_label = ['num_epochs', 'learning_rate', 'batch_size', 'shuffle', 'num_workers', 'num_classes', 'BM_Name', 'ModelName', 'use_pretrained', 'Dyn', 'SplitName', 'TrainSetRatio', 'LabelsList']

    f= open(model_path + os.sep + Version_name + os.sep + 'hyper_parameters' + os.sep + Version_name+ '_hyperparameters.txt',"w+")
    for i in range(len(metadata_label)):
         f.write(str(metadata_label[i]) + '\t' + str(metadata_tab[i])+'\n')
    f.close()
    #%% PLOT LOSSES
    print(' ')
    print('Train and Test losses over epochs')
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(loss_tab_train, label='Train')
    plt.plot(loss_tab_test, label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (mean)')
    plt.legend()
    plt.grid()
    plt.savefig(model_path + os.sep + Version_name + os.sep + 'train_curves' + os.sep + Version_name + '_LossCurves.png')
