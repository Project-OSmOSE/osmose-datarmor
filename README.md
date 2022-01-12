# Jupyter source scripts for osmose



## General structure of codes

All our processing notebooks are used to launch distributed pbs jobs. Behind each of them we then find the same code organization in three steps : 1. `launcher_<block_name>.py` -> 2. def `launch_qsub` in `launcher_<block_name>.py` -> 3. `<block_name>.py`, described as follows:


1. The launcher python scripts are used to initialize variables (eg load `analParams.pickle`), and split the list of units (eg audio files, time periods) to be processed into different batches ;
2. The method `launch_qsub` present in the `launcher_<block_name>.py` script is called for each batch and launch a pbs job (through the qsub command). The .pbs configuration file of each pbs job is based on a template `template_<block_name>.pbs`, which is edited with specific parameters of the current analysis
3. This .pbs template contains the execution command of the processing block script `<block_name>` (eg `time python3 /home/datawork-osmose/DSA_to_ESA/APLOSEspectroGeneration.py PKFIL INDMIN INDMAX` will execute the python script `APLOSEspectroGeneration.py` with the input parameters PKFIL INDMIN INDMAX from the template to be replaced with the current ones). over batches with size defined by `size_batch` (default value = 200).

Note that calculs are most often distributed over both multiple CPUs (e.g. using the `multiprocessing` python library) and multiple nodes (one pbs job = one batch of `size_batch` files processed on one node).





## Notebooks

1. build_datasets.ipynb : used for the importation and formatting of new datasets

2. fileScaleAnalysis.ipynb : used for the generation of file-scale (or shorter) spectrograms

3. datasetScaleAnalysis.ipynb : used for the generation of long-term spectrograms



## Modules 

### For build_datasets.ipynb

* [launcher_buildDataset.py](buildDataset/launcher_buildDataset.py)



### For fileScaleAnalysis.ipynb

* [launcher_fileScale.py](fileScale/launcher_fileScale.py)



### For datasetScaleAnalysis.ipynb

* [launcher_datasetScale.py](datasetScale/launcher_datasetScale.py)


### Resample

* [qsub_resample.sh](resample/qsub_resample.sh)


### Auxiliary

* [aux_functions.py](auxiliary/aux_functions.py)




