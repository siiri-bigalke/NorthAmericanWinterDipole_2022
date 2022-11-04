import xarray as xr
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)



# ------------------
# --- Load data ---
# ------------------

ryear = np.arange(1895, 2022)

# ---- GSL elevation from USGS ----
elev = pd.read_csv('GSL_elevation_1850-2021.csv')
gsl = elev.loc[elev['year'].isin(ryear)]

# ---- 500 hpa from ERA5 ----
pwd ='/work2/Reanalysis/ERA5/ERA5_monthly/pressure/'
ds = xr.open_dataset(pwd + 'ERA5.mon.500mb.hgt.1979-2021.nc')
ds = ds.z.sel(time = ds.time.dt.month.isin([6,7,8])).resample(
              time = 'AS').mean('time') # time: 43 years
					

avg = ds.mean('time')
zanom = ds - avg


# ---- SPEI data from NIDIS ----
nyears = len(ryear)

dir = '/work1/siiri/projects/CAS/Observations/nclimgrid_SPEI/binary_files/'
ds = xr.open_dataset(dir + 'gsl.annual.spei.1895-2021.nc')
ds['spei'] = ds['__xarray_dataarray_variable__']
annual_spei = ds.drop(['__xarray_dataarray_variable__'])

astd = np.std(annual_spei.spei)
annual_spei = annual_spei.spei.sel(time = slice('1980', '2021')).to_numpy()


# -------------------------------------------
# Find pluvial and drought periods in record
# -------------------------------------------

pluvial = []
drought = []

ryear = np.arange(1980, 2022)
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


p = zanom.sel(time = pluvial)
d = zanom.sel(time = drought)


# -------------------
# --- Plotting ------
# -------------------
def plot_background(ax):
    ax.set_extent([230, 295, 20, 55])
    ax.add_feature(cfeature.STATES.with_scale('110m'))
    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=1.0)

lat = zanom.latitude
lon = zanom.longitude
X, Y = np.meshgrid(lon, lat)

pcrs = ccrs.LambertConformal()
tcrs = ccrs.PlateCarree()

fig, ax = plt.subplots(1,2, figsize = (8, 3), 
                            subplot_kw = {'projection': pcrs})

cf1 = ax[0].contourf(X, Y, p.mean('time'), 
	transform = tcrs,
        cmap = 'RdBu_r', 
        levels = np.arange(-200, 210, 10))

ax[0].contour(X, Y, p.mean('time'),  transform = tcrs, colors = 'black')

cf2 = ax[1].contourf(X, Y, d.mean('time'), 
	transform = tcrs,
	cmap = 'RdBu_r',
        levels = np.arange(-200, 220, 20))

ax[1].contour(X, Y, d.mean('time'), transform = tcrs, colors = 'black')


for i in range(2):
    plot_background(ax[i])
    
cbar = fig.colorbar(cf2, ax = ax.ravel().tolist(), 
                         shrink = 0.3, 
                         orientation = 'horizontal',
                         pad = 0.1)

cbar.ax.tick_params(labelsize = 8) 
cbar.set_label('gpm')

ax[0].set_title('e) Pluvial years anomalies  n = %s' %len(p),
                 size = 8,  loc = 'left')
ax[1].set_title('f) Drought years anomalies n = %s' %len(d), 
                 size = 8, loc = 'left')

plt.savefig('z250composites.png', dpi = 500)
fig.supylabel('JJA 500hpa gph')

plt.tight_layout()
plt.show(), exit()


