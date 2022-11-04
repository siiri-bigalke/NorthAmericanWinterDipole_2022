import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal


# Observational T2M data from ERA5
pwd_obs = '/work2/Reanalysis/ERA5/ERA5_monthly/monolevel/ERA5.mon.t2m.1950-2020.nc'
t2m = xr.open_dataset(pwd_obs).sel(time=slice("1960-01-01","2014-12-01"),
	longitude=slice(110,250), latitude=slice(45, -45))#['t2m'] 

nino = xr.open_dataset(pwd_obs).sel(time=slice("1960-01-01","2014-12-01"),
        longitude=slice(190,240), latitude=slice(5, -5))#['t2m']   
# -------------------------------------
# Find SST anomalies from observations
# -------------------------------------
time = t2m.coords['time']
lat = t2m.coords['latitude']
lon = t2m.coords['longitude']
years = (int(len(time)/12))
sst_ndj = np.zeros((years - 1, len(lat), len(lon))) 
t2m_arr = np.array(t2m['t2m'])

# Step 1: Select NDJ months
for year in range(1, years):
  ndj_months = t2m_arr[(year*12-2):(year*12+1), :, :]
  sst_ndj[year-1,:,:] = np.mean(ndj_months, axis=0)

# Step 2: Create climo of NDJ months
climo = np.mean(sst_ndj, axis=(0)) 

# Step 3: Find SST anomalies
ssta = np.zeros((years - 1, len(lat), len(lon)))
for year in range(0, years-1):
  ssta[year,:,:] = sst_ndj[year,:,:] - climo


# Create Nino3.4(Y+1) Index from Observations
# --------------------------------------------
lat = nino.coords['latitude']
lon = nino.coords['longitude']
nino_arr = np.array(nino['t2m'] - 273.15) #convert from K to C
time = nino.coords['time']
years = (int(len(time)/12))
nino4 = np.zeros((years -1, len(lat), len(lon)))

# Select NDJ months
for year in range(1, years):
  n = nino_arr[(year*12-2):(year*12+1), :, :]
  nino4[year-1,:,:] = np.mean(n, axis=0)

nino4_index = signal.detrend(np.mean(nino4, axis=(1,2)))

# Find years where nino3.4 index >=0.5
y = np.arange(1961,2015)
pos = []

for idx, year in enumerate(y):
  if nino4_index[idx] >= 0.5:
    pos.append(idx)

# Composite SSTa where nino3.4 index >= 0.5
# -----------------------------------------
nSST = []

for i, year_idx in enumerate(pos):
  index = pos[i]
  nSST.append(ssta[index])

SST_obs = np.mean(nSST, axis=0)
#s = np.save('SST_obs.np', SST_obs)
exit()






