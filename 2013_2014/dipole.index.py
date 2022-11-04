import numpy as np
from scipy import stats
from scipy import signal
import xarray as xr
import matplotlib.colors as colors
import warnings
import xesmf as xe
import matplotlib.pyplot as plt

pwd = '/work2/Reanalysis/ERA5/ERA5_monthly/pressure/'

'''
# --- z250 --- #
zg_ds = xr.open_dataset('/work2/Reanalysis/ERA5/ERA5_monthly/pressure/ERA5.mon.250mb.hgt.1950-2020.nc')

zg = zg_ds.sel(time=slice("1960-01", "2014-12"))['z']
zg2014 = zg_ds.sel(time=slice("2013-11", "2014-01"))['z']

lat = zg.coords['latitude']
lon = zg.coords['longitude']
time = zg.coords['time']

zg2014_arr = np.array(zg2014.mean(axis=0))
zg_arr = np.array(zg.mean(axis=0))
zonal = np.array(zg.mean(axis=(0,2)))

zg_zonal = zg_arr - zonal[:,None]
anom = zg2014_arr - zg_arr

anomz = anom.mean(axis=1)

plotz = anom - anomz[:, None]
'''


# ---------
f = '/work2/Reanalysis/ERA5/ERA5_monthly/pressure/ERA5.mon.250mb.hgt.1950-2020.nc'

ds = xr.open_dataset(f)
lat = ds.coords['latitude']
lon = ds.coords['longitude']
time = ds.coords['time']

# Year = nov and dec year
years = np.arange(1960, 2014)
ndj = np.zeros((len(years), len(lat), len(lon)))
dpi = np.zeros((len(years)))

a = np.zeros((len(years)))
b = np.zeros((len(years)))

n=0
for idx, year in enumerate(years):
  n+=1
  print(n)
  y1 = str(year)
  y2 = str(year + 1)
  a[idx] = ds.sel(
	longitude=slice(232.5,237.5),
	latitude=slice(52.5, 47.5),
	time=slice(y1+'-11-01', y2+'-01-01')
	)['z'].mean(axis=(0,1,2))
  print('shape of a =', a)
  b[idx] = ds.sel(
	longitude=slice(282.5, 287.5),
	latitude=slice(62.5, 57.5),
	time=slice(y1+'-11-01', y2+'-01-01')
	)['z'].mean(axis=(0,1,2))
  print('shape of b =', b)
  #tds = ds.sel(time = slice(y1+'-11-01', y2+'-01-01'))['z']
  #ndj[idx] = tds


#aanom = a - np.mean(a,axis=0)
#banom = b - np.mean(b,axis=0)

#dipole = aanom - banom

dipole = a - b

print('shape of dipole =', np.shape(dipole))
print(dipole)

index = np.save('1960-2014.dipole.npy', dipole)
exit()
