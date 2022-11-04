import xarray as xr
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)




# First, define function to correctly weight months to annual average
# -------------------------------------------------------------------

def weighted_temporal_mean(ds, var):
    month_length = ds.time.dt.days_in_month
    wgts = month_length.groupby("time.year") / month_length.groupby("time.year").sum()
    np.testing.assert_allclose(wgts.groupby("time.year").sum(xr.ALL_DIMS), 1.0)
    obs = ds[var]
    cond = obs.isnull()
    ones = xr.where(cond, 0.0, 1.0)
    obs_sum = (obs * wgts).resample(time="AS").sum(dim="time")
    ones_out = (ones * wgts).resample(time="AS").sum(dim="time")
    return obs_sum / ones_out

# Second, define function to detrend xr arrays
# ---------------------------------------------

def detrend_dim(da, dim, deg=1):
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit




# ------------------
# --- Load data ---
# ------------------

ryear = np.arange(1950,2021)

# GSL elevation from USGS
elev = pd.read_csv('GSL_elevation_1850-2021.csv')
gsl = elev.loc[elev['year'].isin(ryear)]

#print(elev.to_string()), exit()
# SPEI data from NIDIS


# -------------------------------------------
# Find pluvial and drought periods in record
# -------------------------------------------

pluvial = []
drought = []

for i, year in enumerate(ryear):
    if annual_spei[i] >= astd:
        pluvial.append(str(year))
    elif annual_spei[i] <= (astd*-1):
        drought.append(str(year))
    else:
         continue

p = zanom.sel(time=pluvial)
d = zanom.sel(time=drought)

# -------------------
# --- Plotting ------
# -------------------
def plot_background(ax):
  ax.set_extent([160, 300, 0, 80])
  #ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
  #ax.yaxis.set_major_formatter(LatitudeFormatter())
  ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=1.0)
  ax.set_xticks([-120, -60, 0, 60, 120, 180])#,crs=ccrs.PlateCarree())
  ax.set_yticks([20, 40, 60, 80])#, crs=ccrs.PlateCarree())

lat = zanom.latitude
lon = zanom.longitude
X, Y = np.meshgrid(lon, lat)

pcrs = ccrs.PlateCarree()
tcrs = ccrs.PlateCarree()

fig, ax = plt.subplots(2, 1, subplot_kw={'projection':pcrs})

cf1 = ax[0].contourf(X, Y, p.mean('time'), 
	transform = tcrs,
        cmap='RdBu_r')

cf2 = ax[1].contourf(X, Y, d.mean('time'), 
	transform = tcrs,
	cmap='RdBu_r')

for i in range(2):
    #ax[i].add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=1.0)
    plot_background(ax[i])

cbar = fig.colorbar(cf1, ax=ax[0])
cbar = fig.colorbar(cf1, ax=ax[1])


plt.show(), exit()


