import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
#from sklearn import preprocessing
#from sklearn.preprocessing import MinMaxScaler
from scipy import signal
import pandas as pd


# Load DPI data
PDI = '/work1/siiri/projects/Dipole/Dipole_index/1850-2100/' 
allDPI = np.load(PDI + 'all/all_1850-2014_dipoleindex.npy') # 48 ens, 164 yeras
natDPI = np.load(PDI + 'nat/nat_1850-2014_dipoleindex.npy') # 48 ens, 164 yeras
ghgDPI = np.load(PDI + 'ghg/ghg_1850-2014_dipoleindex.npy') # 48 ens, 164 years
ssp245 = np.load(PDI + 'ssp245/ssp245_dipoleindex.npy') # 48 ens, 85 years
ssp585 = np.load(PDI + 'ssp585/ssp585_dipoleindex.npy') # 46 ens, 85 years


# Load Nino4(Y+1) data
NINO4 = '/work1/siiri/projects/Dipole/TAS_analysis/'
allNINO = np.load(NINO4 + 'ALL/nino_all.npy') # 48, 164
natNINO = np.load(NINO4 + 'NAT/nino_nat.npy') # 48, 164
ghgNINO = np.load(NINO4 + 'GHG/nino_ghg.npy') # 48, 164

# Define funtion to find sliding correlation of DPI and NINO indices
def sliding(dpi, nino):
  c = []
  corr = np.zeros((48,164))
  for mem in range(0,48):
    pdDPI = pd.Series(dpi[mem, :])
    pdNINO = pd.Series(nino[mem, :])
  
    corr[mem, :] = pdDPI.rolling(30).corr(pdNINO)

  c.append(corr)
  crosscorrelation = (np.asarray(c).mean(axis=0))
  return(crosscorrelation)

all = sliding(allDPI, allNINO)
ghg = sliding(ghgDPI, ghgNINO)
nat = sliding(natDPI, natNINO)

# -------------------------------
# Average ensembles of each model
# -------------------------------

# Create dictionary of models
ens = [3, 3, 3, 3, 10, 3, 1, 4, 10, 3, 5]
esum = [0, 3, 6, 9, 12, 22, 25, 26, 30, 40, 43, 45]
fname = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR','CESM2', 'CNRM-CM6-1', 
	'FGOALS-g3', 'GFDL-ESM4', 'HadGEM3-GC31-LL', 'IPSL-CM6A-LR', 
	'MIROC6', 'MRI-ESM2-0']
#fname = {'ACCESS-CM2': np.array(ghg[0:3,:], 'ACCESS-ESM1-5': np.array(ghg[3:6,:])'BCC-CSM2-MR','CESM2', 'CNRM-CM6-1', 
#	'FGOALS-g3', 'GFDL-ESM4', 'HadGEM3-GC31-LL', 'IPSL-CM6A-LR', 
#	'MIROC6', 'MRI-ESM2-0'}

#ESM1 = ghg[3:6, :]
#ESM2 = ghg[esum[1]:esum[2]]
#print(ESM1 == ESM2)
#exit()
'''
model_dict = {key: value for key, value in zip(fname, ens)}

for key, value in model_dict.items():
  model_dict[key].append('ensembles')

  print(model_dict)

#  for j in range


#ghgDict = dictionary(ghg)

exit()
'''

for idx in range(0, 11):
    model_dict = {}
    key = str(fname[idx])
    print(key)
    model_dict[key] = 1#key[np.shape(forcing[ens[idx],:])]
    

print(model_dict)
#    return(dict)



#ghgDict = dictionary(ghg)
#print(ghgDict)


avg_ghg = ghg.mean(axis=0)

fig, ax = plt.subplots()
x = np.arange(1851, 2015)

for mem in range(0,48):
 ax.plot(x, ghg[mem])

ax.plot(x, avg_ghg, linewidth= 3, color='black')
plt.title('HIST-GHG Sliding correlation with \n 30 year window Dipole Index and NINO4(Y+1)')
plt.show()






exit()

