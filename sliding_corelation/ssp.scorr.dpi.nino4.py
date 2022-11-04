import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal

# Load DPI data
DPI = '/work1/siiri/projects/Dipole/Dipole_index/1850-2100/'
dpi245 = np.load(DPI + 'ssp245/ssp245_dipole.npy') # 41 ens, 85 years
dpi585 = np.load(DPI + 'ssp585/ssp585_dipole.npy') # 41 ens, 85 years


# Load Nino4(Y+1) data
NINO4 = '/work1/siiri/projects/Dipole/TAS_analysis/'
nino245 = np.load(NINO4 + 'ssp245/nino_ssp245.npy')
nino585 = np.load(NINO4 + 'ssp585/nino_ssp585.npy')


ens = [5, 10, 1, 2, 6, 3, 1, 1, 4, 3, 5]
esum  = [0, 5, 15, 16, 18, 24, 27, 28, 29, 33, 36, 41]

fname = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR','CESM2', 'CNRM-CM6-1',
        'FGOALS-g3', 'GFDL-ESM4', 'HadGEM3-GC31-LL', 'IPSL-CM6A-LR',
        'MIROC6', 'MRI-ESM2-0']

good = ['ACCESS-CM2', 'GFDL-ESM4', 'IPSL-CM6A-LR',
        'MIROC6', 'MRI-ESM2-0']


# ------------------------------------------------------------------
# Define funtion to find sliding correlation of DPI and NINO indices
# ------------------------------------------------------------------

def sliding(dpi, nino):
  c = []
  years = len(dpi[0,:])
  Nens = len(dpi245)
  corr = np.zeros((Nens, years))
  
  for mem in range(0, Nens):
    pdDPI = pd.Series(dpi[mem, :])
    pdNINO = pd.Series(nino[mem, :])

    corr[mem, :] = pdDPI.rolling(30).corr(pdNINO)

  c.append(corr)
  crosscorrelation = (np.asarray(c).mean(axis=0))
  return(crosscorrelation)

ssp245 = sliding(dpi245, nino245)
ssp585 = sliding(dpi585, nino585)

# ---------------------------------------------------------------------
# Define function to find the model average of the sliding correlation
# ---------------------------------------------------------------------

def modelavg(forcing):
  df = pd.DataFrame(forcing)
  df['model_name'] = ""
  start = 0
  end = ens[0]

  for i in range(0, len(fname)):
    model_name = fname[i]
    df.iloc[start:end]['model_name'] = model_name
    if i == (len(fname)-1):    # so last iteration won't do anything
      break
    else:
      start = end
      end = start + ens[i+1]

  avgpd = df.groupby("model_name").mean()
  allavg = avgpd.mean(axis=0)
  return(avgpd, allavg)

ssp245_corr, avgssp245 = modelavg(ssp245)
ssp585_corr, avgssp585 = modelavg(ssp585)

# Create new average dataframe with good performing models
ssp245_good = (ssp245_corr.loc[(good)]).mean(axis=0)
ssp585_good = (ssp585_corr.loc[(good)]).mean(axis=0)

# -----------
# Plotting
# -----------

modelavg = [ssp245_corr, ssp585_corr]
average = [avgssp245, avgssp585]
good = [ssp245_good, ssp585_good]
title = ['SSP5-2.45', 'SSP5-8.5']

colors=['blue','orange','green','red','purple','brown','pink','gray',
	'olive','chocolate','gold']


fig, ax = plt.subplots(2,1, figsize=(9,6))#, constrained_layout = True)
x = np.arange(2015, 2100)
props = dict(facecolor='lightgray', edgecolor='black')

for forcing in range(2):
  for idx in range(11):
    if forcing == 0:
      ax[forcing].plot(x, modelavg[forcing].iloc[idx,:], color=colors[idx],
	alpha=0.5)
    if forcing == 1:
      ax[forcing].plot(x, modelavg[forcing].iloc[idx,:], color=colors[idx],
	label=fname[idx], alpha=0.5)
  
  ax[forcing].text(0.005, 0.95, title[forcing], transform = ax[forcing].transAxes,
        fontsize=10, bbox=props)  
  ax[forcing].set_xlim([2045,2100])
  
  if forcing ==0:  
    ax[forcing].plot(x, good[forcing], linewidth=3, color='red')
    ax[forcing].plot(x, average[forcing], linewidth=3, color='black')
  if forcing ==1:
    ax[forcing].plot(x, average[forcing], linewidth=3, color='black', label='Model mean')
    ax[forcing].plot(x, good[forcing], linewidth=3, color='red', label='Skillful model mean')
 

fig.tight_layout()
fig.legend(loc=7)
fig.subplots_adjust(right=0.75)

#plt.title('Sliding correlation with 30 year window Dipole Index and NINO4(Y+1)')
plt.savefig('./figures/ssp.slidingcorr.dpi.nino.png', dpi=500)
plt.show()
exit()


