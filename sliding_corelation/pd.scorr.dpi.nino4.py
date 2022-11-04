import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal

# Load DPI data
PDI = '/work1/siiri/projects/Dipole/Dipole_index/1850-2100/'
allDPI = np.load(PDI + 'all/all_1850-2014_dipoleindex.npy') # 48 ens, 164 yeras
natDPI = np.load(PDI + 'nat/nat_1850-2014_dipoleindex.npy') # 48 ens, 164 yeras
ghgDPI = np.load(PDI + 'ghg/ghg_1850-2014_dipoleindex.npy') # 48 ens, 164 years

# Load Nino4(Y+1) data
NINO4 = '/work1/siiri/projects/Dipole/TAS_analysis/'
allNINO = np.load(NINO4 + 'ALL/nino_all.npy') # 48, 164
natNINO = np.load(NINO4 + 'NAT/nino_nat.npy') # 48, 164
ghgNINO = np.load(NINO4 + 'GHG/nino_ghg.npy') # 48, 164

ens = [3, 3, 3, 3, 10, 3, 1, 4, 10, 3, 5]


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


# ---------------------------------------------------------------------
# Define function to find the model average of the sliding correlation
# ---------------------------------------------------------------------

def modelavg(forcing):
  df = pd.DataFrame(forcing)
  df['model_name'] = ""

# Get model name, start index and end index
  n_ens = 48
  start = 0
  end = ens[0]

  for i in range(0, len(fname)):
    model_name = fname[i]
    df.iloc[start:end]['model_name'] = model_name
#    print(model_name)
#    print(start, end)
    if i == (len(fname)-1):    # so last iteration won't do anything
      break
    else:
      start = end
      end = start + ens[i+1]

#  print(df)
  avgpd = df.groupby("model_name").mean()
  allavg = avgpd.mean(axis=0)
  return(avgpd, allavg)

all_corr, avgall = modelavg(all)
nat_corr, avgnat = modelavg(nat)
ghg_corr, avgghg = modelavg(ghg)
print(all_corr)


# Create new average pdDF with good performing models
all_good = (all_corr.loc[(good)]).mean(axis=0)
nat_good = (nat_corr.loc[(good)]).mean(axis=0)
ghg_good = (ghg_corr.loc[(good)]).mean(axis=0)

# -----------
# Plotting
# -----------

modelavg = [all_corr, nat_corr, ghg_corr]
average = [avgall, avgnat, avgghg]
good = [all_good, nat_good, ghg_good]
title = ['All', 'Natural', 'GHG']

colors=['blue','orange','green','red','purple','brown','pink','gray','olive','chocolate','gold']

fig, ax = plt.subplots(3,1, figsize=(10, 8))
x = np.arange(1851, 2015)
props = dict(facecolor='lightgray', edgecolor='black')

for forcing in range(3):
  for idx in range(11):
   # if forcing == 0 or 1:  
    ax[forcing].plot(x, modelavg[forcing].iloc[idx,:], alpha=0.5, color=colors[idx])#, label=fname[idx])
    if forcing == 2:
      ax[forcing].plot(x, modelavg[forcing].iloc[idx,:], alpha=0.5, label=fname[idx],
	color=colors[idx])
  
  ax[forcing].text(0.005, 0.95, title[forcing], transform = ax[forcing].transAxes,
        fontsize=10, bbox=props)
  ax[forcing].set_xlim([1900,2015])
  
  if forcing == 0 or 1:  
    ax[forcing].plot(x, good[forcing], linewidth=3, color='red')
    ax[forcing].plot(x, average[forcing], linewidth=3, color='black')
  if forcing == 2:
    ax[forcing].plot(x, average[forcing], linewidth=3, color='black', label='Model mean')
    ax[forcing].plot(x, good[forcing], linewidth=3, color='red', label='Skillful model mean')


fig.tight_layout()
fig.legend(loc=7)
fig.subplots_adjust(right=0.75)


'''
for forcing in range(3):
  ax[forcing].text(0.005, 0.93, title[forcing], transform = ax[forcing].transAxes,
	fontsize=10, bbox=props)
  for idx in range(11):
    ax[forcing].plot(x, modelavg[forcing].iloc[idx,:])
    ax[forcing].plot(x, average[forcing], linewidth=3, color='black')
    ax[forcing].plot(x, good[forcing], linewidth=3, color='red')
    ax[forcing].set_xlim([1900,2015]) 
'''
#plt.title('Sliding correlation with 30 year window Dipole Index and NINO4(Y+1)')
plt.savefig('./figures/hist.slidingcorr.dpi.nin.png', dpi=500)
plt.show()
exit()


