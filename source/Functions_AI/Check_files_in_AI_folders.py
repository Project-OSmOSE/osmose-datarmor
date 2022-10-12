# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 15:45:43 2022

@author: gabri
"""

import os
import pandas as pd
        
def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))


def CheckAvailableAI_tasks_BM_model(dataset_ID):
    with open('path_osmose_dataset.txt') as f:
        path_osmose_dataset = f.readlines()[0]
    base_path = path_osmose_dataset + dataset_ID + os.sep + 'AI' + os.sep
    
    for root, dirs, files in os.walk(base_path):
        level = root.replace(base_path, '').count(os.sep)
        if level <= 2:
            indent = ' ' * 4 * (level)
            print('{}{}/'.format(indent, os.path.basename(root)))
        
def CheckAvailableAnnotation(dataset_ID):
    with open('path_osmose_dataset.txt') as f:
        path_osmose_dataset = f.readlines()[0]
    base_path = path_osmose_dataset + dataset_ID + os.sep + 'final' + os.sep + 'Annotation_Aplose' + os.sep
    list_files(base_path)
    
def CheckAvailable_labels_annotators(dataset_ID, file_annotation):
    with open('path_osmose_dataset.txt') as f:
        path_osmose_dataset = f.readlines()[0]
    base_path = path_osmose_dataset + dataset_ID + os.sep 
    xl_data = pd.read_csv(base_path + 'final' + os.sep + 'Annotation_Aplose/' + file_annotation)
    FullLabelsList = list(dict.fromkeys(xl_data['annotation']))
    FullAnnotatorsList = list(dict.fromkeys(xl_data['annotator']))
    print('Labels Annotated : ',FullLabelsList)
    print('Annotators : ',FullAnnotatorsList)
    
def CheckAvailableAI_DataSplit(dataset_ID, Task_ID, BM_Name, LengthFile, Fs):
    with open('path_osmose_dataset.txt') as f:
        path_osmose_dataset = f.readlines()[0]
        
    folderName_audioFiles = str(LengthFile)+'_'+str(int(Fs))

    base_path = path_osmose_dataset + dataset_ID + os.sep + 'AI' + os.sep + Task_ID + os.sep + BM_Name + os.sep + folderName_audioFiles + os.sep + 'info_datasplit'
        
    for root, dirs, files in os.walk(base_path):
        level = root.replace(base_path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        
def CheckAvailableAI_TrainedNetwork(dataset_ID, Task_ID, BM_Name, LengthFile, Fs):
    with open('path_osmose_dataset.txt') as f:
        path_osmose_dataset = f.readlines()[0]
        
    folderName_audioFiles = str(LengthFile)+'_'+str(int(Fs))

    base_path = path_osmose_dataset + dataset_ID + os.sep + 'AI' + os.sep + Task_ID + os.sep + BM_Name + os.sep + folderName_audioFiles + os.sep + 'models'
        
    for root, dirs, files in os.walk(base_path):
        level = root.replace(base_path, '').count(os.sep)
        if level <= 1:
            indent = ' ' * 4 * (level)
            print('{}{}/'.format(indent, os.path.basename(root)))


