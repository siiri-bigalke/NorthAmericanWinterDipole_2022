import pandas as pd
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from scipy import stats
import metpy.calc as mpcalc


# !!!!!!! activate metpy conda environment !!!!!!


# ---- Define detrending function ----
def detrend_dim(da, dim, deg=1):
    
    # Calculate anomalies
    clim = da.mean('time')
    anom = da - clim

    # detrend along a single dimension
    p = anom.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(anom[dim], p.polyfit_coefficients)
    detrend = anom - fit
    return(detrend)



# ------------------
# --- Load data ---
# ------------------

ryear = np.arange(1950, 2022)
lat1 = 80
lat2 = -20
lon1 = 120
lon2 = 280


# ---- GSL elevation from USGS ----
elev = pd.read_csv('GSL_elevation_1850-2021.csv')
gsl = elev.loc[elev['year'].isin(ryear)].elev_feet

xtime = pd.date_range(start = '1950', end = '2021', freq = 'AS')
xgsl = xr.DataArray(data = gsl,
                    dims = ['time'],
                    coords = {'time': xtime})

# ---- SSTs from COBE ----
pwd ='/work2/Observation/monthly/sst.cobe.1891-2021.nc'
sst = xr.open_dataset(pwd).sel(time = slice('1950', '2021'),
                               lat = slice(lat2, lat1), 
                               lon = slice(lon1, lon2)).resample(
                                                         time = 'AS').mean('time').sst

# ---- 500 hpa from ERA5 ----
pwd ='/work2/Reanalysis/ERA5/ERA5_monthly/pressure/'
dsOut = xr.open_dataset(pwd + 'ERA5.mon.500mb.hgt.1950-2021.nc')['z']
z500 = dsOut.sel(latitude = slice(lat1, lat2),
                 longitude = slice(lon1, lon2)).resample(time = 'AS').mean('time')


# ---- U, V, & Q from ERA5 ----
lat1 = 80
lat2 = -20
lon1 = 120
lon2 = 280

def era5(var):
    lev = 850
    pwd ='/work2/Reanalysis/ERA5/ERA5_monthly/pressure/'
    ds = xr.open_dataset(pwd + 'ERA5.mon.'+var+'.1000-1.1950-2021.nc').sel(
                         level = lev,
                         longitude = slice(lon1, lon2),
                         latitude = slice(lat1, lat2)).resample(time = 'AS').mean('time')
    return(ds)

qds = era5('q')
vds = era5('vwnd')
uds = era5('uwnd')


# Calculate divergence for each year, then find anomalies
# ---------------------------------------------------------
lats = uds.latitude
lons = uds.longitude
dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats)
full_div = []

for ydx, year in enumerate(ryear):
    print('calculating full year = ', year)
    Q = qds.sel(time = str(year)).mean('time') # averaging to remove time dimension
    U = uds.sel(time = str(year)).mean('time')
    V = vds.sel(time = str(year)).mean('time')

    qu = Q.q * U.u
    qv = Q.q * V.v

    div = mpcalc.divergence(qu, qv, dx = dx, dy = dy) * 1000000
    full_div.append(div)

# Concat and then merge all variables into one dataset for ease of plotting
time = pd.date_range(start = '1950-01-01', periods = 72, freq = 'AS')
divergence = xr.concat(full_div, 'time').assign_coords(time = time).rename('div')

d = divergence.values
#print(type(d)), exit()
dd = detrend_dim(xgsl, 'time', 1)
exit()

# ----------------------------------------------------------------------------
#    Correlate annual GSL elevation data with global Z500, IVT & SST values
# ----------------------------------------------------------------------------

nlags = 7
df = len(ryear)

def lagged_corr(ds, ltx, ldx):

    # First, calculate anomalies and detrend
    gsl = detrend_dim(xgsl,'time', 1)    
    var = detrend_dim(ds, 'time', 1)

    # Second, calculate lagged correations and significance
    corr = []    
    sig = []
    t95 = []
    for idx in range(nlags):
        print(idx)
        c = xr.corr(gsl.shift(time = -idx), var, dim = 'time')
        corr.append(c)

        s = xr.DataArray(data = c.values * np.sqrt((df-2)/(1 - np.square(c.values))),
                         dims = ["lat", "lon"],
                         coords = [c[ltx], c[ldx]])
        #t90 = stats.t.ppf(1-0.05, df-2)
        sig_t95 = stats.t.ppf(1-0.025, df-2)
        sig.append(s)
        t95.append(sig_t95)
 
    return(corr, sig, t95)

z_corr, z_sig, z_t95 = lagged_corr(z500, 'latitude', 'longitude')
sst_corr, sst_sig, sst_t95 = lagged_corr(sst, 'lat', 'lon')
div_corr, div_sig, div_t95 = lagged_corr(divergence, 'latitude', 'longitude')

print('done'), exit()

# --------------------
#       Plotting
# --------------------

pcrs = ccrs.PlateCarree(central_longitude=180)
#pcrs = ccrs.Orthographic(-150, 20)
tcrs = ccrs.PlateCarree()


#fig, ax = plt.subplots(4,2, subplot_kw = {'projection': pcrs}, 
#                          constrained_layout = True)

fig = plt.figure(figsize = (6, 13))
grid = fig.add_gridspec(ncols = 2, nrows = nlags+1,
                        height_ratios = [1,1,1,1,1,1,1,0.1]) # +1 row is for colorbar

min = -0.7
max = 0.7

kwargs = dict(vmin = min,
              vmax = max,
              levels = 21,
              cmap = 'RdBu_r',
              transform = tcrs,
              add_colorbar = False)

skwargs = dict(colors = 'none',
               hatches = ['....', None, '....'],
               extend = 'both',
               add_colorbar = False,
               transform = tcrs)

titles = ['GSL -0', 'GSL -1', 'GSL -2', 'GSL -3',
          'GSL -4', 'GSL -5', 'GSL -6']

for nrow in range(len(z_corr)):

    zax = fig.add_subplot(grid[nrow, 0], projection = pcrs)
    zplot = z_corr[nrow].plot(ax = zax, **kwargs)
    zax.coastlines()
    zax.annotate(titles[nrow],
                 (-0.4,0.5),
                 xycoords = 'axes fraction',
                 va = 'center',
                 ha = 'left',
                 rotation = 90,
                 fontsize = 12)

    z_sig[nrow].plot.contourf(ax = zax, levels = [-1 * z_t95[nrow], z_t95[nrow]], 
 			     **skwargs)          

    sstax = fig.add_subplot(grid[nrow, 1], projection = pcrs)
    sstplot = sst_corr[nrow].plot(ax = sstax, **kwargs)
    sstax.coastlines()


    sst_sig[nrow].plot.contourf(ax = sstax,
                               levels = [-1 * sst_t95[nrow], sst_t95[nrow]],
			       **skwargs)
    
    if nrow == 0:
        zax.set_title('Z500 correlations', loc = 'left')
        sstax.set_title('SST correlations', loc = 'left')

    else:
        continue

cax = plt.subplot(grid[nlags,:])
cbar = plt.colorbar(sstplot, cax = cax, 
                             use_gridspec = True,
                             orientation = 'horizontal')
cbar.set_label('correlation coefficient')


# ---- Subplot formatting ----

for col in range(2):
    for row in range(nlags):
        pax = plt.subplot(grid[row, col])
        pax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=1.0)
        pax.set_title("")

        gl = pax.gridlines(crs = pcrs,
                           draw_labels = True,
                           linewidth = 0.1,
                           color = 'k',
                           alpha = 1,
                           linestyle = '--')

        gl.right_labels = False
        gl.top_labels = False
        gl.bottom_labels = False

        gl.xformatter = LATITUDE_FORMATTER
        gl.yformatter = LONGITUDE_FORMATTER
        gl.xlabel_style = {'size': 8}
        gl.ylabel_style = {'size': 8}

        if col == 1:
            gl.left_labels = False
        else:
            continue


gl = (plt.subplot(grid[6,0])).gridlines(draw_labels = True)
gl.xlabel_style = {'size': 8}
gl.left_labels = False
gl.top_labels = False
           

gl = (plt.subplot(grid[6,1])).gridlines(draw_labels = True)
gl.xlabel_style = {'size': 8}
gl.left_labels = False
gl.top_labels = False

plt.savefig('figs/lagged.corr.z500.gsl.png', dpi=500)
plt.show(), exit()





