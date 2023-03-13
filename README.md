# Analytics platform of OSmOSE 

OSmOSE is an open source project aiming to develop tools to help processing underwater passive acoustic data. Our OSmOSE analytical tools have been primarily developed and made accessible as user-friendly [notebooks](notebooks/) on the infrastructure [Datarmor of IFREMER](https://wwz.ifremer.fr/Recherche/Infrastructures-de-recherche/Infrastructures-numeriques/Pole-de-Calcul-et-de-Donnees-pour-la-Mer). Consequently, it is important to note that most of our codes are not suited for local standalone execution. We are currently making some efforts in adapting our codes in this direction so they can be tested and reviewed by external user/developers.


## List of notebooks 

1. build_datasets.ipynb : used for the importation and formatting of new datasets;

2. fileScaleAnalysis.ipynb : used for the generation of file-scale (or shorter) spectrograms;

3. datasetScaleAnalysis.ipynb : used for long-term analysis (i.e. with timescale at least longer than the audio file duration), including the computation of soundscape metrics (eg long-term averaged spectrograms, EPD) and the retrieval of raw welch spectra at different time resolutions;

4. AI.ipynb : used for machine learning applications.

See [user_guide.pdf](notebooks/user_guide.pdf ) for more details.


## Note for developers : how to contribute ?

1. Start by cloning this folder
2. Download a [sample dataset](https://drive.google.com/file/d/1ZO3_WiaI7j6LZfv8vX_yVeF9ACVZRkio/view?usp=sharing) (ask for access if required) to be put at the same level as your cloned folder
3. Contribute to our open source project on GitHub !

Here is our current [list](https://docs.google.com/document/d/e/2PACX-1vSe6s3FT97Vp3Khqr4NCXtZ9Gr6DKE-RxjbXF8gLxhf7NxkgX76hKXNX4KzMJKyHautm4x__XhMvyj0/pub) of current bugs & new functionalities to be implemented. 

As mentionned in our preambule, only a few of our scripts can be run on a local computer and can then be properly reviewed and augmented by other developers. They are referenced in the next section.


#### Codes adapted for local standalone execution

* The script [qsub_spectroGenerationFileScale.py](source/qsub_spectroGenerationFileScale.py) can be directly used in local to process raw wav data provided eg in ./raw/audio/ . See code comments for details.

* The notebook [AI.ipynb](notebooks/AI.ipynb) can be executed in stand-alone. It is based on the two imported modules [launcher_AI.py ](source/launcher_AI.py) and [module_modelsAI.py](source/module_modelsAI.py). See Use Case 4 of our [user_guide.pdf](notebooks/user_guide.pdf ) for implementation details.
 

#### Modules not adapted for local execution but you can do some stuff anyway!

* [module_soundscape.py](source/module_soundscape.py) : no standalone execution but you can propose new soundscape figures based on code examples and using welch spectra available in the datasets at ./analysis/soundscape/raw_welch/







