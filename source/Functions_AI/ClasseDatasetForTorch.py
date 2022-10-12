# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 20:59:15 2022

@author: gabri
"""



from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import os

import sys
with open('path_codes.txt') as f:
    codes_path = f.readlines()[0]
sys.path.append(codes_path)

# CLASS DATASET
class ClassDataset(Dataset):
    def __init__(self, root_dir, annotation_file, Dyn=np.array([-15,15]), transform=None):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(annotation_file)
        self.transform = transform
        self.Dyn = Dyn

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        spectro_id = self.annotations.iloc[index, 0][:-4] + '.npz'
        spectro_npz = np.load(os.path.join(self.root_dir, spectro_id))
        spectro = spectro_npz['Sxx']
        y_label = torch.tensor(self.annotations.iloc[index, 1:], dtype=torch.float)
        
        if self.transform is not None:
            spectro = self.transform(spectro, self.Dyn, output='torch')

        return (spectro, y_label)
    
    def __getlabels__(self):
        return self.annotations.columns[1:]
    
    def __getfilename__(self, index):
        return self.annotations.iloc[index, 0]
    
