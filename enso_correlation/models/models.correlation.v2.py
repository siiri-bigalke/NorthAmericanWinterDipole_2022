import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

# LOAD CMIP6 TAS data (all models, all ensembles)
# ------------------------------------------------
tas_all = np.load('tas_pacific_all.npy') # (48, 54, 37, 57)
nino_all = np.load('tas_nino3.4_all.npy') # (48, 54, 5, 21)


ens = [3, 3, 3, 3, 10, 3, 1, 4, 10, 3, 5]
esum = [0, 3, 6, 9, 12, 22, 25, 26, 30, 40, 43, 45]
fname = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR','CESM2', 'CNRM-CM6-1',
        'FGOALS-g3', 'GFDL-ESM4', 'HadGEM3-GC31-LL', 'IPSL-CM6A-LR',
        'MIROC6', 'MRI-ESM2-0']


# Step 1: Create climo of winter for each model
# --------------------------------------------------
sst_climo = []
start = 0
end = ens[0]
da = xr.DataArray(tas_all, dims=["ens", "year", "lat", "lon"])

for i in range(len(fname)):
  climo = da[start:end].mean(dim=['ens','year'])
  sst_climo.append(climo)
  if i == (len(fname)-1):    # so last iteration won't do anything
      break
  else:
      start = end
      end = start + ens[i+1]
print("step 1 complete")

# Step 2: Find SST anomalies for each model
# -----------------------------------------
years = len(tas_all[0,:,0,0])
lat = len(tas_all[0,0,:,0]) # length of lat
lon = len(tas_all[0,0,0,:]) # length of lon
ssta = np.zeros((years, lat, lon))
ssta_master = []
  
lat = len(nino_all[0,0,:,0])
lon = len(nino_all[0,0,0,:])
nino_da = xr.DataArray(nino_all, dims=["ens", "year", "lat", "lon"])
nino34 = np.zeros((years, lat, lon))
nino34_ap = []

start = 0
end = ens[0]

for i in range(len(fname)):
  tas_ens = da[start:end].mean(dim='ens')
  nino_ens = nino_da[start:end].mean(dim='ens')  
#  if i == (len(fname)-1):    # so last iteration won't do anythin
#      break
#  else:
#      start = end
#      end = start + ens[i+1] 
  for year in range(years):
    tyear = tas_ens[year, :, :]
    ssta[year,:,:] = tyear - sst_climo[i]

    nyear = nino_ens[year,:,:]
    nino34[year, :, :] = np.mean(nyear, axis=0)
 
  nino34_ap.append(nino34)
  ssta_master.append(ssta)

nino34_index = signal.detrend(np.mean(nino34_ap, axis=(2,3)))
print('shape of nino34_ap =', np.shape(nino34_ap))
print('shape of nino34_index=', np.shape(nino34_index))
print('step 2 complete')


# Step 3: Composite SSTa where Nino3.4 Index >= 0.5
# -------------------------------------------------------

pos = []
nSST = []
pSST = []
finalC = []

y = np.arange(0,54)

for i, model in enumerate(fname):
  print('current model =', fname[i])
  current_model = nino34_index[i]
  ssta = ssta_master[i]
  for idx, year in enumerate(y):
    if current_model[idx] >=0.5:
      pos.append(idx)
      for i, year_idx in enumerate(pos):
        index = pos[i]
        nSST.append(ssta[index-1])
        SST_model = np.mean(nSST, axis=0)
  finalC.append(SST_model)
print(np.shape(finalC))


exit()
'''









