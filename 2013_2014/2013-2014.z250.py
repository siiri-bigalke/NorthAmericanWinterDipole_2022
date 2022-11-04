import numpy as np
import xarray as xr
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

pwd = '/work2/Reanalysis/ERA5/ERA5_monthly/pressure/'

# --- z250 --- #
zg_ds = xr.open_dataset(pwd +'ERA5.mon.250mb.hgt.1950-2020.nc')

zg = zg_ds.sel(time=slice("1960-01", "2014-12"))['z']
zg2014 = zg_ds.sel(time=slice("2013-11", "2014-01"))['z']

lat = zg.coords['latitude']
lon = zg.coords['longitude']
time = zg.coords['time']

#t = zg2014.mean('time')
#zonal = t.mean('longitude')

#anom = t-zonal
#anom.plot()
#plt.show()



#exit()
zg2014_arr = np.array(zg2014.mean(axis=0)) # 2013/2014 NDJ 250hPa average

zg_arr = np.array(zg.mean(axis=0))
zonal = np.array(zg.mean(axis=(0,2)))

zg_zonal = zg_arr - zonal[:,None]
anom = zg2014_arr - zg_arr

anomz = anom.mean(axis=1)

plotz = anom - anomz[:, None]


# --- Plotting --- #
def plot_background(ax):
  ax.set_extent([-180, 1800, 0, 80])
  ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
  ax.yaxis.set_major_formatter(LatitudeFormatter())
  ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=1.0)
  ax.set_xticks([-120, -60, 0, 60, 120, 180])#,crs=ccrs.PlateCarree())
  ax.set_yticks([20, 40, 60, 80])#, crs=ccrs.PlateCarree())

  return(ax)

pcrs = ccrs.PlateCarree(central_longitude=180)
tcrs = ccrs.PlateCarree()

fig, ax =plt.subplots(figsize=(10, 4.5), subplot_kw={'projection':pcrs})
X, Y = np.meshgrid(lon, lat)
plot_background(ax)

cf = ax.contourf(X, Y, plotz, cmap='RdBu_r', levels = 25, transform=tcrs) 
ax.contour(X,Y, plotz, colors='black', levels = 10, transform=tcrs)

cbar1 = fig.colorbar(cf, shrink=0.5)
ax.title.set_text('2013-2014 NDJ Z250 Departure from Climatology (1950-2020)') 

#plt.savefig('2013_2014.z250.png', dpi=500)
plt.show()

exit()
