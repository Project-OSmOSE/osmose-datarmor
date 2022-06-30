

from module_AImodels import *

import pandas as pd
import shutil


path_osmose_home = "/home/datawork-osmose/"
path_osmose_dataset = "/home/datawork-osmose/dataset/"






def list_folder_audio(dataset_ID,nargout=0):
    
    l_ds = [ss for ss in sorted(os.listdir(os.path.join(path_osmose_dataset,dataset_ID,'raw','audio'))) if '.csv' not in ss ]
    
    if nargout == 0:
        print("Available audio folders:")

        for ds in l_ds:
            print("  - {}".format(ds))

    else:
        return l_ds


def list_datasets(nargout=0):
    
    l_ds = [ss for ss in sorted(os.listdir(path_osmose_dataset)) if '.csv' not in ss ]
    
    if nargout == 0:
        print("Available datasets:")

        for ds in l_ds:
            print("  - {}".format(ds))

    else:
        return l_ds
    

    
def build_image_folder(dataset_ID,task_ID,path_input_features,vector_classlabels_csv):

    path_AI = os.path.join(path_osmose_dataset, dataset_ID,'analysis','AI','task_'+str(task_ID))
    path_AI_dataset = os.path.join(path_osmose_dataset, dataset_ID,'analysis','AI','task_'+str(task_ID),'dataset')

    if not os.path.exists(path_AI):
        os.makedirs(path_AI)
        
    if os.path.exists(path_AI_dataset):
        shutil.rmtree(path_AI_dataset)
    os.makedirs(path_AI_dataset)
    

    list_images=[]
    for root, dirs, files in os.walk(path_input_features):
        for file in files:
            if(file.endswith("_1_0.png")):
                list_images.append(os.path.join(root,file))


    hh = pd.read_csv(vector_classlabels_csv)
    vector_classlabels = hh['label'].values
    
    classlist = np.unique(vector_classlabels)


    for classname in classlist:
        if not os.path.exists(os.path.join(path_AI_dataset,'class_'+str(classname))):
            os.makedirs(os.path.join(path_AI_dataset,'class_'+str(classname)))

    for pp,cl in zip(list_images,vector_classlabels):
        shutil.copy(pp , os.path.join(path_osmose_dataset, dataset_ID,'analysis','AI','task_'+str(task_ID),'dataset','class_'+str(cl)))
            

    
def write_result_file(a_dictionary):

    # write result_file
    file = open(os.path.join(path_AI,'results.txt'), "w")
    str_dictionary = repr(a_dictionary)
    file.write("a_dictionary = " + str_dictionary + "\n")
    file.close()


    

def main(dataset_ID,path_input_features,model_name,task_ID,vector_classlabels,test_percent,training_percent):

    global path_AI
    path_AI = os.path.join(path_osmose_dataset, dataset_ID,'analysis','AI','task_'+str(task_ID))
    
    
    if model_name == 'simpleCNNtensorflow':
        
        build_image_folder(dataset_ID,task_ID,path_input_features,vector_classlabels)
        
        accuracy, metadata_path = simpleCNNtensorflow(dataset_ID,task_ID,test_percent)
        
        a_dictionary = {'model_name':model_name,"path_input_features" : path_input_features, "task_ID" : task_ID,'training_percent':training_percent,'test_percent':test_percent,'accuracy':accuracy,'metadata_path':metadata_path}        
        write_result_file(a_dictionary)
        
        return
    
    