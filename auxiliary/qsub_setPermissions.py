import os
import sys
import shutil

path_osmose_home = "/home/datawork-osmose/"
path_osmose_dataset = "/home/datawork-osmose/dataset/"

dataset_ID = sys.argv[1]
path_to_set = sys.argv[2]

# shutil.rmtree(path_to_set + '/ongoing_pbsFiles')
os.remove(path_to_set + '/analysis_fiche.csv')

os.system('chgrp -R gosmose ' + path_to_set )
os.system('chmod -R g+rwx ' + path_to_set )

