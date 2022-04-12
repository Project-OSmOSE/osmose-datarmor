# Analytics platform of OSmOSE 


Our analytics services are available as user-friendly notebooks (osmose_analytics_v0.0/notebooks/), based on different code sources depending on their application (osmose_analytics_v0.0/source/). 

It is important to note that all our codes have been mostly developed to be run on the infrastructure Datarmor/IFREMER, and most of them are consequently not suited for local stand-alone execution. We are currently working on this point to allow interesting user/developer to review and contribute to our codes (see section \@ref(sec_contribute))).



## Analytical services 

They are accessible through the following notebooks:

1. build_datasets.ipynb : used for the importation and formatting of new datasets

2. fileScaleAnalysis.ipynb : used for the generation of file-scale (or shorter) spectrograms

3. datasetScaleAnalysis.ipynb : used for the generation of long-term spectrograms

See [user_guide.pdf](notebooks/user_guide.pdf )


## Note for developers : how to contribute ? {#sec_contribute}

In the following we list our modules for which we would welcome any contribution. We selected only modules which can be executed locally in a stand-alone way within the environment `env_osmosePlatform` provided above. For each of them, we also provide a list of new functionalities that could be improved. 

1. Start by cloning this folder
2. Only our following scripts can actually be executed locally within this cloned environment


### Sampled dataset

Start by cloning the environment `env_osmosePlatform`


### Module to generate file-scale spectrograms

* [qsub_spectroGenerationFileScale.py](source/fileScale/qsub_spectroGenerationFileScale.py)










