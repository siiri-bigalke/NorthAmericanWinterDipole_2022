import numpy as np
from scipy import stats
from scipy import signal
import xarray as xr
import matplotlib.colors as colors
import warnings
import xesmf as xe 

#nino = np.load('./nino_ghg_model.npy')
#print(np.shape(nino))
#exit()

# ======================
#    Load CMIP6 Data
# ======================

# Define function to append all runs for each model
# -----------------------------------------------------

def zgload_data(model, ensN):
  pwd = '/work2/CMIP6/CMIP6_monthly/ZG/HIST_GHG/' 
  zgloadfiles = []
  for run in range(1,ensN+1):
    if model == 'CNRM-CM6-1': 
      member = 'r' + str(run) + 'i1p1f2'

    elif model == 'HadGEM3-GC31-LL': 
      member = 'r' + str(run) + 'i1p1f3'

    #elif model == 'GISS-E2-1-G':
    #  member = 'r' + str(run+2) + 'i1p1f1'

    else: 
      member = 'r' + str(run) + 'i1p1f1'
    dataF = pwd + 'zg_Amon_' + model + '_hist-GHG_' + member + '*' + '.nc'         
    zgloadfiles.append(dataF)
#    print(dataF)
  return(zgloadfiles)

def tasload_data(model, ensN):
  pwdt = '/work2/CMIP6/CMIP6_monthly/TAS/HIST_GHG/'
  tasloadfiles = []
  for run in range(1, ensN+1):
    if model == 'CNRM-CM6-1': 
      member = 'r' + str(run) + 'i1p1f2'
    
    elif model == 'HadGEM3-GC31-LL':
      member = 'r' + str(run) + 'i1p1f3'    

    else:
      member = 'r' + str(run) + 'i1p1f1'    
    dataF = pwdt + 'tas_Amon_' + model + '_hist-GHG_' + member + '*' +'.nc'
    tasloadfiles.append(dataF)
    #print(dataF)
  return(tasloadfiles)

# -----------------------------------------------------
# START LOOPING THROUGH ENSEMBLE MEMBERS FOR REGRESSION
# -----------------------------------------------------

# For now, not using GISS or CanESM5 models

#ens = [25]
#fname = ['CanESM5']#'GISS-E2-1-G']
#ens = [3, 10, 3, 1, 4, 10, 3, 5, 3]

#fname = ['CESM2', 'CNRM-CM6-1', 'FGOALS-g3', 'GFDL-ESM4', 'HadGEM3-GC31-LL', 'IPSL-CM6A-LR', 'MIROC6', 'MRI-ESM2-0', 'NorESM2-LM']

ens = [3, 3, 3, 3, 10, 3, 1, 4, 10, 3, 5]
fname = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR','CESM2', 'CNRM-CM6-1', 'FGOALS-g3', 'GFDL-ESM4', 'HadGEM3-GC31-LL', 'IPSL-CM6A-LR', 'MIROC6', 'MRI-ESM2-0']

'''
# -----------------------  
# Loop though ZG data
# --------------------
zgmodelArr = []
zgmasterArr = []
n = 0

for idx, f in enumerate(fname):
  for mem in range(ens[idx]): 
    n+=1
    filein = zgload_data(fname[idx],ens[idx])[mem]
    zgds = xr.open_mfdataset(filein).sel(plev = '25000', method='nearest') 
    # Get dimensions and set up regridding fucntion
    # ---------------------------------------------
    zg = zgds.sel(time=slice("1960-12","2014-12"))['zg']
    ds_out = xr.Dataset({'lat' : (['lat'], np.arange(-90, 90, 2.5)),
			 'lon' : (['lon'], np.arange(0, 360, 2.5)), })

    regridder_zg = xe.Regridder(zgds, ds_out, 'bilinear')	
    zgds_re = regridder_zg(zg)       			# zgds_re is a xr dataArray
    #print("current model = ", fname[idx])
    #print("shape of regridded ds =" ,np.shape(zgds_re))
    #print("type of zgds_re = ", type(zgds_re))

    # Get winter average (NDJ) and create climo of z250
    # --------------------------------------------------
    lat = zgds_re.coords['lat']
    lon = zgds_re.coords['lon']
    time = zgds_re.coords['time']

    years = (int(len(time)/12))
    ndj = np.zeros((years - 1 , len(lat), len(lon)))
    zg_arr = np.array(zgds_re)    

    for year in range(1, years):
      ndj_months = zg_arr[(year * 12 -2):(year * 12 + 1), :, :]
      gph = np.nanmean(ndj_months, axis=0)
      zonal = gph.mean(axis=(1))
      ndj[year - 1, :, :] = gph - zonal[:,None]

    zg_ndj = signal.detrend(ndj, axis=0) 
    print("current model = ", fname[idx])
   
    zgmodelArr.append(zg_ndj)
    print('shape of zgmodelArr =', np.shape(zgmodelArr))
  
  zgmasterArr.append(zgmodelArr)
zgmodelArr = np.save('zg_ghg.npy', zgmodelArr)
zgmasterArr = np.save('zg_ghg_model.npy', zgmasterArr)

print('shape of zgmasterArr =', np.shape(zgmasterArr))
'''

# -----------------------  
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
    tas = tasds.sel(time=slice("1960-12","2014-12"))['tas']
    #ds_out = xe.util.grid_global(2.5,2.5)
    ds_out = xr.Dataset({'lat' : (['lat'], np.arange(-90, 90, 2.5)),
                         'lon' : (['lon'], np.arange(0, 360, 2.5)), })

    regridder_tas = xe.Regridder(tasds, ds_out, 'bilinear')	
    tasds_re = regridder_tas(tas)   

    # Select NDJ and subtract climo
    # ------------------------------
    tas_arr = np.array(tasds_re)
    lat = np.arange(0, 72, 1)
    lon = np.arange(0, 144, 1)
    time = tasds_re.coords['time']
      
    years = (int(len(time)/12))
    tndj = np.zeros((years - 1, len(lat), len(lon))) # years, lat, lon
    
    for year in range(1, years):
      tndj[year-1, :, :] = np.nanmean((tas_arr[(year * 12 -2):(year * 12 + 1), :, :]), axis=0)
    
    climo = np.nanmean(tndj, axis = 0)
    anom = tndj - climo
        
    tas_detrend = signal.detrend(anom, axis=0)
    
    tasmodelArr.append(tas_detrend)
    print('shape of tasmodelArr = ', np.shape(tasmodelArr))
               
  tasmasterArr.append(tasmodelArr)   
tasmodelArr = np.save('tas_ghg.npy', tasmodelArr)
tasmasterArr = np.save('tas_ghg_model.npy', tasmasterArr)

'''
# --------------------------  
# Create Nino4(Y+1) Index
# --------------------------
ninomodelArr = []
ninomasterArr = []
n = 0
for idx, f in enumerate(fname):
  for mem in range(ens[idx]):    
    n+=1
    print('n = ', n)
    print('current model = ', fname[idx])
    filein = tasload_data(fname[idx],ens[idx])[mem]
    tasds = xr.open_mfdataset(filein).sel(time=slice("1960-12","2014-12"), lat=slice(-5, 5), lon=slice(160, 210))
    
    nino_arr = np.array(tasds['tas'])    
    lat = tasds.coords['lat']
    lon = tasds.coords['lon']
    time = tasds.coords['time']
    years = (int(len(time)/12))
    sst = np.zeros((years - 1, len(lat), len(lon)))
    #if fname == 'CNRM-CM6-1':
    #  sst = np.zeros((years - 1, 8, 36))
    #else:
    #  sst = np.zeros((years - 1, 10, 41))
    #print('shape of sst =', np.shape(sst))
    
    for year in range(1, years):
      tas_months = nino_arr[(year*12 -2):(year*12+1), :, :]
      sst[year-1, :, :] = np.nanmean(tas_months, axis=(0))

    nino4 = signal.detrend(np.mean(sst, axis=(1,2)))
    ninomodelArr.append(nino4)
    print('shape of ninomodelArr =', np.shape(ninomodelArr))
  ninomasterArr.append(ninomodelArr)
  print('shape of ninomasterArr = ', np.shape(ninomasterArr))

ninomodelArr = np.save('nino_ghg.npy', ninomodelArr)
ninomasterArr = np.save('nino_ghg_model.npy', ninomasterArr) 
'''
exit()
