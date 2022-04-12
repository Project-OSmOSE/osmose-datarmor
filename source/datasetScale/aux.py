#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2017-2018 Project-ODE
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Authors: Alexandre Degurse

"""
Module providing aux functions for jobLauncher
"""

import os, glob
import pandas as pd
import shutil

import soundfile
import numpy as np
from scipy import signal
import pickle
import subprocess


dataset_base_path = "/home/datawork-osmose/dataset/"
datawork_base_path = "/home/datawork-osmose/FeatureEngine/"

def list_datasets(nargout=0):
    
    l_ds = os.listdir(dataset_base_path)
    
    if nargout == 0:
        print("Available datasets:")

        for ds in l_ds:
            print("  - {}".format(ds))

    else:
        return l_ds
            
def read_metaInfo_from_config(config):    
    csvFileArray = pd.read_csv( os.path.join(dataset_base_path,config.dataset_id,"metadata","raw","from_data.csv"), header=0)
    return csvFileArray["sample_rate"][0]    

        
def list_aux(config):
    l_aux = os.listdir(dataset_base_path + config.dataset_id + "/raw/auxiliary")
    l_aux = [aux for aux in l_aux if aux.endswith("csv")]

    print("Available auxiliary files:")

    for aux in l_aux:
        print("  - {}".format(aux))

        
def list_estimation_results(config):
    l_aux = os.listdir(dataset_base_path + config.dataset_id + "/results/DC")
    l_aux = [aux for aux in l_aux if aux.endswith("csv")]

    print("Available estimation results:")

    for aux in l_aux:
        print("  - {}".format(aux))
        
        
def list_feature_params_set(config):
    print("Feature parameters sets for {}:".format(config.dataset_id))

    fps_path = config.dataset_base_path + config.dataset_id + "/results/soundscapes/" 

    if os.path.isdir(os.path.dirname(fps_path)):
        os.chdir(fps_path)
        folder_features = [os.path.join(fps_path, f) for f in os.listdir(fps_path) if f.startswith("features")] 
        folder_features.sort(key=lambda x: os.path.getmtime(x),reverse = True)

#         folder_features_short = [os.path.basename(os.path.normpath(ll))[0:22] for ll in folder_features]
        folder_features = [os.path.basename(os.path.normpath(ll)) for ll in folder_features]
        
        for fps in np.unique(folder_features):
            print("- {}".format( fps ))
            
        return folder_features
            

def delete_jobs(config):
    for path_features_folder in glob.glob(dataset_base_path+config.dataset_id+'/results/soundscapes/features*'):
        print('removing:',path_features_folder)
        shutil.rmtree(path_features_folder)

    for path_features_folder in glob.glob(dataset_base_path+config.dataset_id+'/results/soundscapes/*'):
        print('removing:',path_features_folder)
        os.remove(path_features_folder)
    
    for filename in glob.glob(datawork_base_path+'jobs/configs/'+config.dataset_id+'*'):
        os.remove(filename) 
    for filename in glob.glob(datawork_base_path+'jobs/pbs/'+config.dataset_id+'*'):
        os.remove(filename)     
    

def _tob_bounds_from_toc(center_freq):
    return center_freq * np.power(10, np.array([-0.05, 0.05]))                    
    
def launch(config):
        
    list_sev=[]
    on_nber_exec_per_node = 0
    on_segment_duration = 0
    on_n_nodes = 0
    on_window_size = 0
    on_window_overlap = 0
    
    if type(config.nber_exec_per_node)==list:
        list_sev = config.nber_exec_per_node
        on_nber_exec_per_node = 1
    elif type(config.segment_duration)==list:
        list_sev = config.segment_duration
        on_segment_duration = 1
    elif type(config.n_nodes)==list:
        list_sev = config.n_nodes
        on_n_nodes = 1  
    elif type(config.window_size)==list:
        list_sev = config.window_size
        on_window_size = 1  
    elif type(config.window_overlap)==list:
        list_sev = config.window_overlap
        on_window_overlap = 1  
        
    for ll in list_sev:
        
        if on_nber_exec_per_node:
            config.nber_exec_per_node = ll
        elif on_segment_duration:
            config.segment_duration = ll
        elif on_n_nodes:
            config.n_nodes = ll
        elif on_window_size:
            config.window_size = ll
            config.nfft = ll
        elif on_window_overlap:
            config.window_overlap = ll
                    
        config_js = config.as_json
        with open(config.job_config_file_location, "w") as job_file:
            job_file.write(config_js)

        with open(datawork_base_path + "jobLauncher/template.pbs", "r") as template:
            template_lines = template.readlines()

        with open(config.pbs_file_location, "w") as pbs_file:
            for line in template_lines:
                pbs_file.write(
                    line.replace("JOBNAME", 'compute')\
                        .replace("N_NODES", str(config.n_nodes))\
                        .replace("PATHOUTPUT", config.path_analysisFolder + '/ongoing_pbsFiles/' )\
                        .replace("NUM_EXEC", str(config.nber_exec_per_node *config.n_nodes))\
                        .replace("JAR_LOCATION", config.jar_location)\
                        .replace("JOB_CONFIG_FILE_LOCATION", config.job_config_file_location)
                )

        r = os.system("qsub " + config.pbs_file_location)

    if not list_sev:
        
        config_js = config.as_json

        with open(config.job_config_file_location, "w") as job_file:
            job_file.write(config_js)

        with open(datawork_base_path + "jobLauncher/template.pbs", "r") as template:
            template_lines = template.readlines()

        with open(config.pbs_file_location, "w") as pbs_file:
            for line in template_lines:
                pbs_file.write(
                    line.replace("JOBNAME", 'compute')\
                        .replace("N_NODES", str(config.n_nodes))\
                        .replace("PATHOUTPUT", config.path_analysisFolder + '/ongoing_pbsFiles/log_computeFeaturesDatasetScale.txt' )\
                        .replace("NUM_EXEC", str(config.nber_exec_per_node *config.n_nodes))\
                        .replace("JAR_LOCATION", config.jar_location)\
                        .replace("JOB_CONFIG_FILE_LOCATION", config.job_config_file_location)
                )





