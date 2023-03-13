READ ME FILE FOR THE API CODES USED TO DOWNLOAD CDS DATA

I - Getting Started

These codes are used to download data from the CDS (Climate Data Store).
More precisely it can download any hourly single level data since 1950 anywhere in the world.
A list of the variables that can be downloaded is accessible here : https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview
So far, the code has been prepared for surface temperature, u10 and v10 wind speed, total precipitation and wave direction and period.

Adding variables :
Other variables can be added by looking for their correct names in : 
https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form
After checking the desired variables, click on 'Show API request' and copy the variable list and paste it as the data list in the jupyter notebook.

Getting access to the CDS data requires an account. You can create one here :
https://cds.climate.copernicus.eu/user/login
After your account is created, clicking on the same link above gives you access to the UDI and key that you need to enter in the jupyter notebook. You can also find this information by clicking on your name in the upper right corner.

Make sure all the packages are installed, you can install them using conda in a terminal.


II - Using the Jupyter Notebook

This code is meant to be used locally, not on datamor for example. Both the jupyter notebook and the python script need to be in the same directory. 

The area from where to download data is defined by a 'square' whose borders are the cardinal boundaries.

The filename you define is only attributed to the raw data file downloaded from the cds. Additionnal formatted files are also created.
All file will be created in your home directory, in a subdirectory called 'api'.

III - Using the downloaded data

The data you have downloaded is three dimensional (space+time). Therefore, it is stored as a 3D numpy array you can read using np.read().
One array saves the stamps, and each individual variable is saved in its own array.
All arrays have the same dimensions and shapes.
To read the stamps array you need to allow pickle data : stamps = np.load('stamps.npy', allow_pickle=True)
The stamps consists in time, latitude and longitude information. 
For instance, if you've downloaded total precipitation data you will get a tp.npy array and a stamps.npy array.
For chosen i,j,k values corresponding to time, latitude and longitude you will have :
tp[i,j,k] = 3.2
stamps[i,j,k] = (date, latitude, longitude)
Changing i will change the time, changing j will change the latitude and changing k will change the longitude.


