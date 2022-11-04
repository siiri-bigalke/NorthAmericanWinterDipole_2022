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
lat1 = 90
lat2 = -20
lon1 = 0#120
lon2 = 360#300


# ---- GSL elevation from USGS ----
elev = pd.read_csv('GSL_elevation_1850-2021.csv')
gsl = elev.loc[elev['year'].isin(ryear)]


# ---- SSTs from COBE ----
pwd ='/work2/Observation/monthly/sst.cobe.1891-2021.nc'
dsOut = xr.open_dataset(pwd).sel(time = slice('1950', '2021'),
                               lat = slice(lat2, lat1),
                               lon = slice(lon1, lon2))

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
#    print(year)
#    print('annual_spei = ', annual_spei[i])
#    print('astd = ', astd)
#    print('     ')

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
    ds = dsOut.sst.sel(time = dsOut.time.dt.month.isin(months)).resample(
              time = 'AS').mean('time') 

    avg = ds.mean('time')
    zanom = ds - avg

    p = zanom.sel(time = pluvial)
    d = zanom.sel(time = drought)

    return(p, d)

p_annual, d_annual = anomalies([np.arange(1,13)])
p_jja, d_jja = anomalies([6,7,8])
p_jfm, d_jfm = anomalies([11,12,1,2,3])

p_all = [p_jfm, p_jja, p_annual]
d_all = [d_jfm, d_jja, d_annual]

# ---------------------------
# -------- Plotting ---------
# ---------------------------

def plot_background(ax):
    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=1.0)

lat = dsOut.lat
lon = dsOut.lon
X, Y = np.meshgrid(lon, lat)

pcrs = ccrs.PlateCarree(central_longitude = 180)
tcrs = ccrs.PlateCarree()

fig = plt.figure(figsize = (8, 5.5), constrained_layout = True)
(sub_top, sub_bot) = fig.subfigures(nrows = 2, ncols = 1)

axes_top = sub_top.subplots(nrows = 3, ncols = 3, 
                            sharex = True, sharey = True, 
                            subplot_kw = {'projection': pcrs})

axes_bot = sub_bot.subplots(nrows = 3, ncols = 3,
                            sharex = True, sharey = True, 
                            subplot_kw = {'projection': pcrs}) 


def plot_subfig(metric, subplot, event, mlev):
    for idx in range(len(metric)):
        print('on column =', idx)
        
        lev = mlev
        # Row 1: 1950-1979 data
        # ----------------------
        period1 = metric[idx].sel(time = slice('1950','1979')).mean('time')
        cf1 = subplot[0, idx].contourf(X, Y, period1,
   	                       transform = tcrs,
                               cmap = 'RdBu_r', 
                               levels = lev,
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
                                       levels = lev,
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
                                       levels = lev,
                                       extend = 'both')

        subplot[2, idx].contour(X, Y, diff,  
                                transform = tcrs, 
                                colors = 'black')

        plot_background(subplot[2, idx])

        # Row titles
        # -----------
        time_periods = ['1950-1979', '1980-2021', 'Δ']
        n1 = len(metric[idx].sel(time = slice('1950','1979')))
        n2 = len(metric[idx].sel(time = slice('1980','2021')))

#        kwargs = dict(va = 'center',
#                      ha = 'left',
#                      rotation = 90,
#                      fontsize = 8)

        subplot[0,0].annotate(time_periods[0] + f'\n    n = {n1}', (-0.1,0.5),
                              xycoords = 'axes fraction',
                              va = 'center',
                              ha = 'left',
                              rotation = 90,
                              fontsize = 8)

        subplot[1,0].annotate(time_periods[1] + f'\n    n = {n2}', (-0.1,0.5),
                              xycoords = 'axes fraction', 
                              va = 'center',
                              ha = 'left',
                              rotation = 90,
                              fontsize = 8)

      
        subplot[2,0].annotate(time_periods[2],(-0.1,0.5),
                              xycoords = 'axes fraction', 
                              va = 'center',
                              ha = 'left',
                              rotation = 90,
                              fontsize = 8)


    # Subplot formatting
    # --------------------
    cbar = fig.colorbar(cf3, ax = subplot.ravel().tolist(), 
                         orientation = 'vertical',
                         shrink = 0.5)
    cbar.set_label('Degree C')
    
    subplot[0,0].set_title(event+'Winter (NDJFM)', loc = 'left')
    subplot[0,1].set_title(event+'Summer (JJA)', loc = 'left')
    subplot[0,2].set_title(event+'Annual', loc = 'left')
'''  
    for col in range(2):
       for row in range(3):
           subplot[row, 0].annotate(time_periods[row] + f'\n n = {n}',
                    (-0.2,0.5),
                    xycoords = 'axes fraction',
                    va = 'center',
                    ha = 'left',
                    rotation = 90,
                    fontsize = 12)
'''

plot_subfig(p_all, axes_top, 'Pluvial ', np.arange(-2, 2.2, .2))
plot_subfig(d_all, axes_bot, 'Drought ', np.arange(-1, 1.1, .1))
fig.suptitle('SST Anomaly Composites', fontsize = 14)

plt.savefig('figs/nov-mar.sst.drought+pluvial.composites.png', dpi = 500)

plt.show(), exit()


