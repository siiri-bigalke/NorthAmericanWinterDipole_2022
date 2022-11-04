import numpy as np
from scipy import stats
from scipy import signal
import xarray as xr
import matplotlib.colors as colors
import warnings
import xesmf as xe
import matplotlib.pyplot as plt

ens = [3, 3, 3, 3, 10, 3, 1, 4, 10, 3, 5]
fname = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR','CESM2', 'CNRM-CM6-1', 'FGOALS-g3', 'GFDL-ESM4', 'HadGEM3-GC31-LL', 'IPSL-CM6A-LR', 'MIROC6', 'MRI-ESM2-0']

# Define function to append all runs for each model
# -----------------------------------------------------

def tasload_data(model, ensN):
  pwdt = '/work2/CMIP6/CMIP6_monthly/TAS/HIST_ALL/'
  tasloadfiles = []
  for run in range(1, ensN+1):
    if model == 'CNRM-CM6-1':
      member = 'r' + str(run) + 'i1p1f2'

    elif model == 'HadGEM3-GC31-LL':
      member = 'r' + str(run) + 'i1p1f3'

    else:
      member = 'r' + str(run) + 'i1p1f1'
    dataF = pwdt + 'tas_Amon_' + model + '_historical_' + member + '*' +'.nc'
    tasloadfiles.append(dataF)
  return(tasloadfiles)
 
# Loop through TAS data
# -----------------------  

tasmasterArr = []
tasmodelArr = []
n = 0

for idx, f in enumerate(fname):
  for mem in range(ens[idx]):
    n+=1
    print('n = ', n)
    print('current model = ', fname[idx])
    filein = tasload_data(fname[idx],ens[idx])[mem]
    tasds = xr.open_mfdataset(filein)

    # Get dimensions and set up regridding fucntion
    # ---------------------------------------------
    tas = tasds.sel(time=slice("1960-01","2014-12"))['tas']
    ds_out = xr.Dataset({'lat' : (['lat'], np.arange(-90, 90, 1)),
                         'lon' : (['lon'], np.arange(0, 360, 1)), })
    regridder_tas = xe.Regridder(tasds, ds_out, 'bilinear')
    tasds_re = regridder_tas(tas)

    # Select NDJ and subtract climo
    # ------------------------------
    t = tasds_re.sel(lon=slice(110,250), lat=slice(-45, 45))

    lon = t['lon']
    lat = t['lat']

    y1 = np.where(lat == -5)
    y2 = np.where(lat == 5)
    x1 = np.where(lon == 190)
    x2 = np.where(lon == 240)

    #print('lon index values', x1, x2)
    #print('lat index values', y1, y2)

    tas_arr = np.array(t)
    lat = t.coords['lat']
    lon = t.coords['lon']
    time = t.coords['time']

    years = (int(len(time)/12))
    tndj = np.zeros((years - 1, len(lat), len(lon))) # years, lat, lon

    for year in range(1, years):
      print(np.shape(tas_arr[(year * 12 -2):(year * 12 + 1), :, :]))
      tndj[year-1, :, :] = np.mean((tas_arr[(year * 12 -2):(year * 12 + 1), :, :]), axis=0)
#print(np.shape(tndj))
#print(tndj)
#exit()

    climo = np.mean(tndj, axis=0)
    anom = tndj - climo
    tas_detrend = signal.detrend(anom, axis=0)
    tasmodelArr.append(tas_detrend)
    print('shape of tasmodelArr = ', np.shape(tasmodelArr))

  tasmasterArr.append(tasmodelArr)
tasmodelArr = np.save('binary_files/1deg.tas_pacific_all.npy', tasmodelArr)
#tasmasterArr = np.save('tas_pacific_all_model.npy', tasmasterArr)
exit()
