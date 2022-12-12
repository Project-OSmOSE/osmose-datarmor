import cdsapi
from datetime import date, datetime
import os
from netCDF4 import Dataset
import numpy as np
import pandas as pd

def make_cds_file(key, udi):
	os.chdir(os.path.expanduser("~"))
	try :
 	   os.remove('.cdsapirc')
	except FileNotFoundError :
	    pass

	cmd1 = "echo url: https://cds.climate.copernicus.eu/api/v2 >> .cdsapirc"
	cmd2 = "echo key: {}:{} >> .cdsapirc".format(udi, key)
	os.system(cmd1)
	os.system(cmd2)

	try :
	   os.mkdir('api')
	except FileExistsError:
	    pass

	path_to_api = os.path.join(os.path.expanduser("~"), "api/")

	os.chdir(path_to_api)
	os.getcwd()



def return_cdsapi(filename, key, variable, year, month, day, time, area):

  filename = filename + '.nc'

  c = cdsapi.Client()

  r = c.retrieve('reanalysis-era5-single-levels',
            {
              'product_type' : 'reanalysis',
              'variable' : variable,
              'year' : year,
              'month' : month,
              'day' : day,
              'time' : time,
              'area' : area,
              'format' : 'netcdf',
              'grid':[0.25, 0.25],
            },
            filename,
            )
  r.download(filename)
  print('\n[===>--------]\n')


def format_nc(filename):

  downloaded_cds = filename + '.nc'
  fh = Dataset(downloaded_cds, mode='r')

  variables = list(fh.variables.keys())[3:]
  single_levels = np.zeros((len(variables), fh.dimensions.get('time').size, fh.dimensions.get('latitude').size, fh.dimensions.get('longitude').size))

  lon = fh.variables['longitude'][:]
  lat = fh.variables['latitude'][:]
  time_bis = fh.variables['time'][:]
  for i in range(len(variables)):
    single_levels[i] = fh.variables[variables[i]][:]

  time_units = fh.variables['time'].units

  def transf_temps(time) :
    time_str = float(time)/24 + date.toordinal(date(1900,1,1))
    return date.fromordinal(int(time_str))

  time_ = list(time_bis)

  data = pd.DataFrame()
  data['time'] = time_
  data['time'] = data['time'].apply(lambda x : transf_temps(x))
  hours = np.array(time_)%24
  dates = np.array([datetime(elem.year, elem.month, elem.day, hour) for elem, hour in list(zip(data['time'], hours))])

  print('\n[======>-----]\n')

  return dates, lon, lat, single_levels, variables



def save_results(dates, lat, lon, single_levels, variables, filename):
	for i in range(len(variables)):
		np.save(variables[i]+'_'+filename, np.ma.filled(single_levels[i], fill_value=float('nan')), allow_pickle = True)
	stamps = np.zeros((len(dates), len(lat), len(lon), 3), dtype=object)
	for i in range(len(dates)):
		for j in range(len(lat)):
			for k in range(len(lon)):
				stamps[i,j,k] = [dates[i], lat[j], lon[k]]
	np.save('stamps_'+filename, stamps, allow_pickle = True)
	print('\n[============]\nDone')
	return None


def final_creation(df1, filename, key, variable, year, month, day, time, area, type_crea = 'complexe') :
  return_cdsapi(filename, key, variable, year, month, day, time, area)
  dates, lon, lat, single_levels, variables = format_nc(filename)
  test = save_results(dates, lat, lon, single_levels, variables, filename)
  return test

