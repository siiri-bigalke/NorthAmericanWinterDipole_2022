import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pdi
import scipy.stats as stats
from scipy import signal

fname = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR','CESM2', 'CNRM-CM6-1',
        'FGOALS-g3', 'GFDL-ESM4', 'HadGEM3-GC31-LL', 'IPSL-CM6A-LR',
        'MIROC6', 'MRI-ESM2-0']

# Load numpy array of Nino3.4 composite SSTa 
nmodels = np.load('../models/ens.model.composite.npy')
nobs = np.load('../obs/SST_obs.npy')

# Nino3.4(Y+1) arrays
y1models = np.load('../models/ensy1.model.composite.npy')
y1obs = np.load('../obs/SST_obs.Y+1.npy') 

n_obs = nobs.flatten()
y1_obs = y1obs.flatten()

nino34corr = []
ninoy1corr = []

for model in range(len(nmodels)):
  n = nmodels[model,:,:].flatten()
  nino34corr.append(stats.pearsonr(n_obs, n)[0])
  
  y1 = y1models[model,:,:].flatten()
  ninoy1corr.append(stats.pearsonr(y1_obs, y1)[0])

# Plotting
x = np.arange(len(fname))
width = 0.4

fig, ax = plt.subplots(figsize=(11,6))

ax.bar(x-width / 2, nino34corr, width, label="Nino3.4")
ax.bar(x+width / 2, ninoy1corr, width, color='darkgoldenrod', label="Nino3.4(Y+1)")
ax.set_xticks(x)
ax.set_xticklabels(fname, rotation = 45, size=6)
ax.set_ylabel('Correlation Coefficient')
ax.axhline(y=0.4, linestyle = 'dotted', color='red')
ax.legend()
plt.title('CMIP6 Model Skill on ENSO Precursor')

plt.savefig('cmip6_dipole_modelskill.png', dpi=500)

plt.show()

exit()

