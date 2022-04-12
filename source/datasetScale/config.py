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
Module providing config handler class
"""

import os
import json
import numpy as np
import pandas as pd
import glob

path_osmose_dataset = "/home/datawork-osmose/dataset/"

class Config:
    def __init__(self):
        self.datawork_base_path = "/home/datawork-osmose/FeatureEngine/"
        self.dataset_base_path = "/home/datawork-osmose/dataset/"
        self.jar_location = self.datawork_base_path + "FeatureEngine-assembly-0.1.jar"#"FeatureEngine-assembly-0.1_TOLess2.jar"#"FeatureEngine-assembly-0.1.jar" # replaced by join .jar if config.aux_file not empty
        
        self.path_analysisFolder = None
        self.output_base_directory = None
        self.wav_directory = None
        self.metadata_file = None
        self._aux_file = ""
        self.aux_ts_col_name = None
        self.config_file = None

        self._dataset_id = None
        self.n_files = None
        self.sound_sampling_rate = None
        self.sound_sampling_rate_target = None
        self.sound_n_channels = None
        self.sound_sample_size_in_bits = None        
        self.sound_calibration_factor = 0.0
        
        self.segment_duration = 5.0        
        self.window_size = 2048
        self.window_overlap = 0
        self.nfft = 2048
        self.low_freq_tol = None
        self.high_freq_tol = None
                
        self.n_nodes = 1
        self.nber_exec_per_node = 7        
        
        self.with_auxjoin = 0        
        self.aux_ts_col_name = "timestamp"

    @property
    def dataset_id(self):
        return self._dataset_id

    @dataset_id.setter
    def dataset_id(self, ds_id):
        if not os.path.isdir(self.dataset_base_path + ds_id):
            raise Exception("Dataset doesn't exist")

        self.wav_directory = self.dataset_base_path + ds_id + "/raw/audio/" 

        self.config_file = self.dataset_base_path + ds_id + "/metadata/raw/from_user.json"

        if not os.path.isdir(self.wav_directory):
            raise Exception("Wav files not found")
#         if not os.path.isfile(self.metadata_file):
#             raise Exception("Metadata file not found")

        self._dataset_id = ds_id

        metadata = pd.read_csv(os.path.join(path_osmose_dataset, self._dataset_id, 'raw/metadata.csv'))
        self.sound_n_channels = int(metadata["nchannels"][0])
        self.sound_sample_size_in_bits = int(metadata["sound_sample_size_in_bits"][0])
        self.low_freq_tol = 1.0 #* self.sound_sampling_rate
        self.orig_fileDuration = int(metadata['orig_fileDuration'][0])
        

#         print(type(self.sound_sampling_rate))
#         print(type(self.sound_n_channels))
#         print(type(self.sound_sample_size_in_bits))
#         print(type(self.low_freq_tol ))

#         with open(self.config_file, "r") as config_fd:
#             config_dict = json.load(config_fd)
            
#             print(type(float(config_dict["sample_rate"])))
#             print( type(config_dict["n_channels"]))
#             print( type(config_dict["sample_size_in_bit"]))
#             print( type(0.2 * float(config_dict["sample_rate"])))

#             self.sound_sampling_rate = float(config_dict["sample_rate"])
#             self.sound_n_channels = config_dict["n_channels"]
#             self.sound_sample_size_in_bits = config_dict["sample_size_in_bit"]
#             self.low_freq_tol = 0.2 * self.sound_sampling_rate
#             self.high_freq_tol = 0.4 * self.sound_sampling_rate

        
#         self.metadata_file = self.dataset_base_path + self._dataset_id + "/raw/audio/400.0_timestamp.csv"     
#         self.sound_sampling_rate = 400.0         


    @property
    def aux_file(self):
        return self._aux_file
    
    @aux_file.setter
    def aux_file(self, auxf):
        self._aux_file = self.dataset_base_path + self.dataset_id + "/raw/auxiliary/" + auxf
        
        if 'csv' in auxf:# if config.aux_file not empty then use jar for joining aux var
            self.jar_location = self.datawork_base_path + "FeatureEngine-assembly-join-0.1.jar"
            self.with_auxjoin = 1

        if 'csv' in auxf and not os.path.isfile(self._aux_file):
            raise Exception("Auxilairy file ({}) not found".format(self._aux_file))

    @property
    def as_json(self):
        
        self.n_files = int( len(glob.glob( os.path.join(self.wav_directory, str(int(self.orig_fileDuration)) +'_'+str(int(self.sound_sampling_rate)) , '*wav') )) )

        if self.sound_sampling_rate_target != self.sound_sampling_rate:
            # self.metadata_file = self.dataset_base_path + self._dataset_id + "/raw/audio/"+str(int(self.sound_sampling_rate_target))+"_timestamp.csv"
            self.sound_sampling_rate = float(self.sound_sampling_rate_target)

        self.wav_directory = self.wav_directory + str(int(self.orig_fileDuration)) +'_'+str(int(self.sound_sampling_rate))            
        self.metadata_file = self.wav_directory + "/timestamp.csv"
            
#         self.processing_id = "_".join([str(e) for e in [          float(round(self.segment_duration,4)),int(self.window_size),float(self.window_overlap),self.sound_sampling_rate_target,self.n_nodes,self.nber_exec_per_node,self.with_auxjoin
#         ]])
#        self.processing_id = "_".join([str(e) for e in [self._dataset_id, 'computeDSA', float(round(self.segment_duration,4)),self.sound_sampling_rate ]])
    
        self.processing_id = 'compute'     
        
        self.job_id = self.processing_id        
                
        self.window_overlap = int(self.window_overlap * self.window_size) 
        
        self.output_base_directory = os.path.join(self.path_analysisFolder, 'datasetScale_features', str(int(self.sound_sampling_rate)) )


        # remove features directory if already existing
        if os.path.exists(self.output_base_directory):
            os.system('rm -rf ' + self.output_base_directory)
            
#        self.job_config_file_location = self.datawork_base_path + "jobs/configs/{}.json".format(self.job_id)
        self.job_config_file_location = self.path_analysisFolder + "/ongoing_pbsFiles/{}.json".format(self.job_id)
#         self.pbs_file_location = self.datawork_base_path + "jobs/pbs/{}.pbs".format(self.job_id)

        #self.pbs_file_location = self.dataset_base_path + self._dataset_id + "/results/soundscapes/pbs/{}.pbs".format(self.job_id)
    
        self.pbs_file_location = self.path_analysisFolder + '/ongoing_pbsFiles/pbs_computeFeaturesDatasetScale.pbs'
#         self.pbs_timestampJsonCreation = self.path_analysisFolder + '/ongoing_pbsFiles/pbs_timestampJsonCreation.pbs'

        self.nfft = int(self.nfft)
        self.window_size = int(self.window_size)
        self.segment_duration = int(self.segment_duration)

        # for debug : variables in json.dumps must not be numpy instance
#         dd=dict(vars(self), **{"aux_file": "file://" + self._aux_file})
#         for obj in dd:
#             print( type(dd[obj]) )
#             print( obj )


        return json.dumps( dict(vars(self), **{"aux_file": "file://" + self._aux_file}) )

    def __str__(self):
        return\
        ("Config parameters:\n"
        "  - dataset_id: {}\n"
         "  - sound_calibration_factor: {}\n"
         "  - segment_duration: {}\n"
         "  - window_size: {}\n"
         "  - window_overlap: {}\n"
         "  - nfft: {}\n"
         "  - low_freq_tol: {}\n"
         "  - fs: {}\n"
         "  - target_fs: {}\n"
         "  - timestamp_csv: {}\n"
         "  - high_freq_tol: {}\n").format(
            self.dataset_id,
            self.sound_calibration_factor,
            self.segment_duration,
            self.window_size,
            self.window_overlap,
            self.nfft,
            self.low_freq_tol,
            self.sound_sampling_rate,            
            self.sound_sampling_rate_target,
            self.metadata_file,
            self.high_freq_tol
        )
