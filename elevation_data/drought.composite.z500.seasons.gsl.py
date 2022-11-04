import xarray as xr
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


# ------------------
# --- Load data ---
# ------------------

ryear = np.arange(1895, 2022)
lat1 = 80
lat2 = -20
lon1 = 120
lon2 = 300


# ---- GSL elevation from USGS ----
elev = pd.read_csv('GSL_elevation_1850-2021.csv')
gsl = elev.loc[elev['year'].isin(ryear)]


# ---- 500 hpa from ERA5 ----
pwd ='/work2/Reanalysis/ERA5/ERA5_monthly/pressure/'

f1 = xr.open_dataset(pwd + 'ERA5.mon.500mb.hgt.1950-1978.nc') # preliminary back extension
f2 = xr.open_dataset(pwd + 'ERA5.mon.500mb.hgt.1979-2021.nc')
dsOut = xr.concat([f1, f2], dim = 'time').sel(
                  longitude = slice(lon1, lon2),
                  latitude = slice(lat1, lat2))
 


# ---- SPEI data from NIDIS ----
nyears = len(ryear)

dir = '/work1/siiri/projects/CAS/Observations/nclimgrid_SPEI/binary_files/'
ds = xr.open_dataset(dir + 'gsl.annual.spei.1895-2021.nc')
ds['spei'] = ds['__xarray_dataarray_variable__']
annual_spei = ds.drop(['__xarray_dataarray_variable__'])

astd = np.std(annual_spei.spei)
annual_spei = annual_spei.spei.sel(time = slice('1950', '2021')).to_numpy()



# -------------------------------------------
# Find pluvial and drought periods in record
# -------------------------------------------

pluvial = []
drought = []

ryear = np.arange(1950, 2022)
for i, year in enumerate(ryear):
    print(year)
    print('annual_spei = ', annual_spei[i])
    print('astd = ', astd)
    print('     ')

    if annual_spei[i] >= astd:
        pluvial.append(str(year))
    elif annual_spei[i] <= (astd*-1):
        drought.append(str(year))
    else:
         continue


print('pluvial = ', pluvial)
print('drought = ', drought)



# -------------------------------------------
#  Find anomalies for annual, jja, and jfm
# -------------------------------------------

def anomalies(months):
    ds = dsOut.z.sel(time = dsOut.time.dt.month.isin(months)).resample(
              time = 'AS').mean('time') 

    avg = ds.mean('time')
    zanom = ds - avg

    p = zanom.sel(time = pluvial)
    d = zanom.sel(time = drought)

    return(p, d)

p_annual, d_annual = anomalies([np.arange(1,13)])
p_jja, d_jja = anomalies([6,7,8])
p_jfm, d_jfm = anomalies([1,2,3])

p_all = [p_jfm, p_jja, p_annual]
d_all = [d_jfm, d_jja, d_annual]

# ---------------------------
# -------- Plotting ---------
# ---------------------------

def plot_background(ax):
    ax.add_feature(cfeature.STATES.with_scale('110m'))
    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=1.0)

lat = dsOut.latitude
lon = dsOut.longitude
X, Y = np.meshgrid(lon, lat)

pcrs = ccrs.PlateCarree(central_longitude = 180)#LambertConformal()
tcrs = ccrs.PlateCarree()

fig, ax = plt.subplots(3, 3, figsize = (12, 9), 
                             subplot_kw = {'projection': pcrs},
                             constrained_layout = True)



for idx in range(len(p_all)):
    print('on column =', idx)
    # Row 1: 1950-1979 data
    # ----------------------
    period1 = d_all[idx].sel(time = slice('1950','1979')).mean('time')
    cf1 = ax[0, idx].contourf(X, Y, period1,
   	                       transform = tcrs,
                               cmap = 'RdBu_r', 
                               levels = np.arange(-300, 310, 10), 
			       extend = 'both')

    ax[0, idx].contour(X, Y, period1,  
                        transform = tcrs, 
                        colors = 'black')

    plot_background(ax[0, idx])


    # Row 2: 1980-2021 data
    # ----------------------
    period2 = d_all[idx].sel(time = slice('1980','2021')).mean('time')
    cf2 = ax[1, idx].contourf(X, Y, period2,
   	                       transform = tcrs,
                               cmap = 'RdBu_r', 
                               levels = np.arange(-300, 310, 10),
                               extend = 'both')

    ax[1, idx].contour(X, Y, period2,  
                        transform = tcrs, 
                        colors = 'black')
    plot_background(ax[1, idx])


   # Row 3: Period2 - Period1
   # -------------------------
    diff = period2 - period1
    cf3 = ax[2, idx].contourf(X, Y, diff,
   	                       transform = tcrs,
                               cmap = 'RdBu_r', 
                               levels = np.arange(-300, 310, 10), 
                               extend='both')

    ax[2, idx].contour(X, Y, diff,  
                        transform = tcrs, 
                        colors = 'black')
    plot_background(ax[2, idx])

cbar = fig.colorbar(cf3, ax = ax.ravel().tolist(), 
                         orientation = 'vertical',
                         shrink = 0.75)#'horizontal')
cbar.set_label('gpm')

# Column titles (season)
ax[0,0].set_title('JFM (DROUGHT)', loc = 'left')
ax[0,1].set_title('JJA (DROUGHT)', loc = 'left')
ax[0,2].set_title('Annual (DROUGHT)', loc = 'left')

# Subplot formatting
# --------------------

time_periods = ['1950-1979', '1980-2021', 'Δ']

for col in range(2):
    for row in range(3):
        ax[row, 0].annotate(time_periods[row],
                 (-0.2,0.5),
                 xycoords = 'axes fraction',
                 va = 'center',
                 ha = 'left',
                 rotation = 90,
                 fontsize = 12)

plt.savefig('figs/drought.z500.composites.png', dpi = 500)
#fig.suptitle('Drought Years 500hpa anomaly composites')

plt.show(), exit()


