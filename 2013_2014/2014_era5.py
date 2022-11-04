import numpy as np
from scipy import stats
from scipy import signal
import xarray as xr
import matplotlib.colors as colors
import warnings
import xesmf as xe
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath
import matplotlib.path as mpath
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

pwd = '/work2/Reanalysis/ERA5/ERA5_monthly/pressure/'

# --- z250 --- #
zg_ds = xr.open_dataset(pwd +'ERA5.mon.250mb.hgt.1950-2020.nc')

zg = zg_ds.sel(time=slice("1960-01", "2014-12"))['z']
zg2014 = zg_ds.sel(time=slice("2013-11", "2014-01"))['z']

lat = zg.coords['latitude']
lon = zg.coords['longitude']
time = zg.coords['time']

zg2014_arr = np.array(zg2014.mean(axis=0))
zg_arr = np.array(zg.mean(axis=0))
zonal = np.array(zg.mean(axis=(0,2)))

zg_zonal = zg_arr - zonal[:,None]
anom = zg2014_arr - zg_arr

anomz = anom.mean(axis=1)

plotz = anom - anomz[:, None]

# --- SSTs --- #
t2m_ds = xr.open_dataset('/work2/Reanalysis/ERA5/ERA5_monthly/monolevel/ERA5.mon.t2m.1950-2020.nc')

t2m = t2m_ds.sel(time=slice("1960-01", "2014-12"))['t2m']/100
t2m2014 = t2m_ds.sel(time=slice("2013-11", "2014-01"))['t2m']/100

lat = t2m.coords['latitude']
lon = t2m.coords['longitude']
time = t2m.coords['time']

t2m2014_arr = np.array(t2m2014.mean(axis=0))
t2m_arr = np.array(t2m.mean(axis=0))


print('shape of t2m2014_arr = ', np.shape(t2m2014_arr))
print('shape of t2m_arr = ', np.shape(t2m_arr))
tanom = t2m2014_arr - t2m_arr

tanomz = tanom.mean(axis=1)

plot = t2m2014_arr - t2m_arr
t = plot.mean(axis=1)
plott2m = plot - t[:, None]

#plott2m = tanom - tanomz[:, None]
print(np.shape(plott2m))

# --- Plotting --- #
def plot_background(ax):
#  ax.set_extent([-60,60, 0, 80])
  ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
  ax.yaxis.set_major_formatter(LatitudeFormatter())
  ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=1.0)
#  ax.set_xticks([-120, -60, 0, 60, 120, 180])#,crs=ccrs.PlateCarree())
#  ax.set_yticks([20, 40, 60, 80])#, crs=ccrs.PlateCarree())

  return(ax)

pcrs = ccrs.PlateCarree()#(central_longitude=180)
tcrs = ccrs.PlateCarree(central_longitude=180)
tcrs0 = ccrs.PlateCarree()

fig, ax = plt.subplots(1,2, figsize=(10, 4.5), subplot_kw={'projection':pcrs}, sharey=True,
	sharex=True)


X, Y = np.meshgrid(lon, lat)

for col in range(2):
  plot_background(ax[col])

  if col == 0:
    cf = ax[col].contourf(X, Y, plotz, cmap='RdBu_r') 
    ax[col].contour(X,Y, plotz, colors='black')

  if col == 1:
    cf2 = ax[col].contourf(X, Y, plott2m, levels = np.arange(-.08, .09, .01),  cmap='RdBu', extend='both')
    #ax[col].contour(X,Y, plott2m, colors='black')

#cbar1 = fig.colorbar(cf)
#cbar2 = fig.colorbar(cf2)
plt.show()

exit()
