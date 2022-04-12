# Analytics platform of OSmOSE 


Our analytics services are accessible through user-friendly [notebooks](notebooks/). It is important to note that all our codes have been mostly developed to be run on the infrastructure Datarmor/IFREMER, and most of them are consequently not suited for local stand-alone execution. We are currently making some efforts in adapting our codes in this direction so they can be tested and reviewed by external user/developers.


## Notebooks 

1. build_datasets.ipynb : used for the importation and formatting of new datasets

2. fileScaleAnalysis.ipynb : used for the generation of file-scale (or shorter) spectrograms

3. datasetScaleAnalysis.ipynb : used for long-term analysis, can be used to generate soundscape metrics (eg long-term averaged spectrograms, EPD) or retrieve raw welch spectra at different time resolutions.

See [user_guide.pdf](notebooks/user_guide.pdf ) for more details.


## Note for developers : how to contribute ?

1. Start by cloning this folder
2. Download a [sampled dataset](https://drive.google.com/file/d/1hJZAGFlkL1_Cc1lC77LFaiRgM_HTXZc0/view?usp=sharing) (ask for access if required) to be put at the same level as your cloned folder
3. Make contributions using standard git process!

Here is our current [list](https://docs.google.com/document/d/e/2PACX-1vSe6s3FT97Vp3Khqr4NCXtZ9Gr6DKE-RxjbXF8gLxhf7NxkgX76hKXNX4KzMJKyHautm4x__XhMvyj0/pub) of current bugs & new functionalities to be implemented. 


#### Modules adapted for local stand-alone execution

* [qsub_spectroGenerationFileScale.py](source/fileScale/qsub_spectroGenerationFileScale.py)










