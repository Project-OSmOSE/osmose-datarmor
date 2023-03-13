import os
import sys

from os.path import exists
import glob

dataset_ID = sys.argv[1]
path_to_set = sys.argv[2]
folderName_audioFiles = sys.argv[3]

print('folderName_audioFiles',folderName_audioFiles)



print('set permissions to:',os.path.join(path_to_set, 'raw/audio/',folderName_audioFiles))
os.system('chgrp -R gosmose ' + os.path.join(path_to_set, 'raw/audio/',folderName_audioFiles) )
os.system('chmod -R g+rwx ' + os.path.join(path_to_set, 'raw/audio/',folderName_audioFiles) )

print('set permissions to:',os.path.join(path_to_set, 'analysis'))
if os.path.isdir(os.path.join(path_to_set, 'analysis')):
    os.system('chgrp -R gosmose ' + os.path.join(path_to_set, 'analysis') )
    os.system('chmod -R g+rwx ' + os.path.join(path_to_set, 'analysis') )

os.remove( os.path.join( path_to_set, 'analysis', 'ongoing_pbsFiles', 'pbs_setPermissions_0.pbs') )


# shutil.rmtree(path_to_set + '/ongoing_pbsFiles')
# the number of pbs should be nul to be sure that you are at the end of your processing chain!
if (exists(os.path.join(path_to_set, 'analysis','analysis_fiche.csv'))) and (len( glob.glob(os.path.join(path_to_set, 'analysis', 'ongoing_pbsFiles', '*.pbs')))==0):
    os.remove(os.path.join(path_to_set, 'analysis','analysis_fiche.csv'))