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

ryear = np.arange(1950, 2022)
lat1 = 90
lat2 = -20
lon1 = 120
lon2 = 300


# ---- GSL elevation from USGS ----
elev = pd.read_csv('GSL_elevation_1850-2021.csv')
gsl = elev.loc[elev['year'].isin(ryear)].elev_feet

xtime = pd.date_range(start = '1950', end = '2021', freq = 'AS')
xgsl = xr.DataArray(data = gsl,
                    dims = ['time'],
                    coords = {'time': xtime})

# ---- SPEI data from NIDIS ----
nyears = len(ryear)

dir = '/work1/siiri/projects/CAS/Observations/nclimgrid_SPEI/binary_files/'
ds = xr.open_dataset(dir + 'gsl.annual.spei.1895-2021.nc')
ds['spei'] = ds['__xarray_dataarray_variable__']
annual_spei = ds.drop(['__xarray_dataarray_variable__'])

astd = np.std(annual_spei.spei)
annual_spei = annual_spei.spei.sel(time = slice('1950', '2021')).to_numpy()


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

#qds = era5('q')
vds = era5('vwnd')
uds = era5('uwnd')

# ---- Vertically integrated moisture divergence from ERA5 ----
pwd ='/work2/Reanalysis/ERA5/ERA5_monthly/monolevel/'

f1 = xr.open_dataset(pwd + 'ERA5.mon.vimd.1950-1958.nc') # preliminary back extension
f2 = xr.open_dataset(pwd + 'ERA5.mon.vimd.1959-2021.nc')
divds = xr.concat([f1, f2], dim = 'time').sel(
                  longitude = slice(lon1, lon2),
                  latitude = slice(lat1, lat2)).resample(time = 'AS').mean('time')

'''
# Calculate divergence with metpy by multiplying U and V wind by specific humidity (Q)
# -----------------------------------------------------------------------------------

lats = uds.latitude
lons = uds.longitude
dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats)


# Calculate divergence for each year, then find anomalies
quC = []
qvC = []
full_div = []

for ydx, year in enumerate(ryear):
    print('calculating divergence year = ', year)
    Q = qds.sel(time = str(year)).mean('time') # averaging to remove time dimension
    U = uds.sel(time = str(year)).mean('time')
    V = vds.sel(time = str(year)).mean('time')

    qu = Q.q * U.u
    qv = Q.q * V.v

    div = mpcalc.divergence(qu, qv, dx = dx, dy = dy) * 1000000

    quC.append(qu)
    qvC.append(qv)
    full_div.append(div)
'''


# Concat and then merge all variables into one dataset for ease of plotting
time = pd.date_range(start = '1950-01-01', periods = 72, freq = 'AS')
#divergence = xr.concat(full_div, 'time').assign_coords(time = time).rename('div')
#uwind = xr.concat(quC, 'time').assign_coords(time = time).rename('u')
#vwind = xr.concat(qvC, 'time').assign_coords(time = time).rename('v')
#dxr = xr.merge([divds, uwind, vwind])

dxr = xr.merge([divds, uds, vds])

# Calculate anomalies for Q, U, and divergence
average_dxr = dxr.mean('time')
anom = dxr - average_dxr



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
    ds = dsOut.sel(time = dsOut.time.dt.month.isin(months)).resample(
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

pcrs = ccrs.PlateCarree(central_longitude = 180)
tcrs = ccrs.PlateCarree()

fig = plt.figure(figsize = (8, 12), constrained_layout = True)
(sub_top, sub_bot) = fig.subfigures(nrows = 2, ncols = 1)

axes_top = sub_top.subplots(nrows = 3, ncols = 3, 
                            sharex = True, sharey = True, 
                            subplot_kw = {'projection': pcrs})

axes_bot = sub_bot.subplots(nrows = 3, ncols = 3,
                            sharex = True, sharey = True, 
                            subplot_kw = {'projection': pcrs}) 


def plot_subfig(metric, subplot, event):
    for idx in range(len(metric)):
        print('on column =', idx)


        # Row 1: 1950-1979 data
        # ----------------------
        period1 = metric[idx].sel(time = slice('1950','1979')).mean('time')
        cf1 = subplot[0, idx].contourf(X, Y, period1,
   	                       transform = tcrs,
                               cmap = 'RdBu_r', 
                  #             levels = np.arange(-300, 310, 10), 
			       extend = 'both')

        subplot[0, idx].contour(X, Y, period1,  
                        transform = tcrs, 
                        colors = 'black')

        plot_background(subplot[0, idx])


        # Row 2: 1980-2021 data
        # ----------------------
        period2 = metric[idx].sel(time = slice('1980','2021')).mean('time')
        cf2 = subplot[1, idx].contourf(X, Y, period2,
   	                               transform = tcrs,
                                       cmap = 'RdBu_r', 
               #                        levels = np.arange(-300, 310, 10),
                                       extend = 'both')

        subplot[1, idx].contour(X, Y, period2,  
                                transform = tcrs, 
                                colors = 'black')

        plot_background(subplot[1, idx])


       # Row 3: Period2 - Period1
       # -------------------------
        diff = period2 - period1
        cf3 = subplot[2, idx].contourf(X, Y, diff,
   	                               transform = tcrs,
                                       cmap = 'RdBu_r', 
                 #                      levels = np.arange(-300, 310, 10), 
                                       extend = 'both')

        subplot[2, idx].contour(X, Y, diff,  
                                transform = tcrs, 
                                colors = 'black')

        plot_background(subplot[2, idx])



    # Subplot formatting
    # --------------------
    cbar = fig.colorbar(cf3, ax = subplot.ravel().tolist(), 
                         orientation = 'vertical',
                         shrink = 0.5)
    cbar.set_label('kg/m-2')
    
    subplot[0,0].set_title(event+'Winter (JFM)', loc = 'left')
    subplot[0,1].set_title(event+'Summer (JJA)', loc = 'left')
    subplot[0,2].set_title(event+'Annual', loc = 'left')
    
    time_periods = ['1950-1979', '1980-2021', 'Δ']
    for col in range(2):
       for row in range(3):
           subplot[row, 0].annotate(time_periods[row],
                    (-0.2,0.5),
                    xycoords = 'axes fraction',
                    va = 'center',
                    ha = 'left',
                    rotation = 90,
                    fontsize = 12)


plot_subfig(p_all, axes_top, 'Pluvial ')
plot_subfig(d_all, axes_bot, 'Drought ')
fig.suptitle('Moisture Flux Divergence Anomaly Composites', fontsize = 14)

plt.savefig('figs/divergence.drought+pluvial.composites.png', dpi = 500)

plt.show(), exit()


