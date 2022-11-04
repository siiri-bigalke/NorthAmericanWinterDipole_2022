import pandas as pd
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import metpy.calc as mpcalc


# !!!!!!! activate metpy conda environment !!!!!!


# ------------------
# --- Load data ---
# ------------------

ryear = np.arange(1950, 2022)

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

qds = era5('q')
vds = era5('vwnd')
uds = era5('uwnd')


# --------------------------------------------------------------------------
# --- Composite moisture flux divergence for pluvial and drought years ---
# --------------------------------------------------------------------------


# STEP ONE:
# Identify pluvial and dry years according to SPEI index
# --------------------------------------------------------

pluvial = []
drought = []

ryear = np.arange(1950, 2022)
for i, year in enumerate(ryear):

    if annual_spei[i] >= astd:
        pluvial.append(str(year))
    elif annual_spei[i] <= (astd*-1):
        drought.append(str(year))
    else:
         continue


# STEP TWO:
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
    print('calculating full year = ', year)
    Q = qds.sel(time = str(year)).mean('time') # averaging to remove time dimension
    U = uds.sel(time = str(year)).mean('time')
    V = vds.sel(time = str(year)).mean('time')
        
    qu = Q.q * U.u
    qv = Q.q * V.v

    div = mpcalc.divergence(qu, qv, dx = dx, dy = dy) * 1000000

    quC.append(qu)
    qvC.append(qv)
    full_div.append(div)


# Concat and then merge all variables into one dataset for ease of plotting
time = pd.date_range(start = '1950-01-01', periods = 72, freq = 'AS')
divergence = xr.concat(full_div, 'time').assign_coords(time = time).rename('div')
uwind = xr.concat(quC, 'time').assign_coords(time = time).rename('u')
vwind = xr.concat(qvC, 'time').assign_coords(time = time).rename('v')
dxr = xr.merge([divergence, uwind, vwind])

    
# Calculate anomalies for Q, U, and divergence
average_dxr = dxr.mean('time')
anom = dxr - average_dxr



# STEP THREE:
# Composite anomalies for Q, U, and moisture flux divergence for pluvial and droughts
# -----------------------------------------------------------------------------------

def calcdiv(hydro):    
    quC = []
    qvC = []
    divC = []

    # Select U, V, and DIV values for drought and pluvial years
    for ydx, year in enumerate(hydro):
        print('hydro years = ', year)
        U = anom.u.sel(time = year).mean('time')
        V = anom.v.sel(time = year).mean('time')
        DIV = anom.div.sel(time = year).mean('time') # averaging to remove time dimension
        
        quC.append(U)
        qvC.append(V) 
        divC.append(DIV)


 
    # Concat and then merge all variables into one dataset for ease of plotting
    divergence = xr.concat(divC, 'year').assign_coords(
                            year = pd.to_datetime(hydro)).rename('div')
    uwind = xr.concat(quC, 'year').assign_coords(
                            year = pd.to_datetime(hydro)).rename('u')
    vwind = xr.concat(qvC, 'year').assign_coords(
                            year = pd.to_datetime(hydro)).rename('v')
    dxr = xr.merge([divergence, uwind, vwind])
    return(dxr)


xpluvial = calcdiv(pluvial)
xdrought = calcdiv(drought)


# --------------------------------------------------------------------
# ---- PLOTTING ---- divide periods before and after 1980 + difference
# --------------------------------------------------------------------

pcrs = ccrs.PlateCarree(central_longitude=180)#LambertConformal()
tcrs = ccrs.PlateCarree()


fig, ax = plt.subplots(3,2, figsize = (9, 9),
                            sharex = True,
                            sharey = True, 
			    subplot_kw = {'projection': pcrs}, 
                            constrained_layout = True)

div = [xpluvial, xdrought]
res = 35
scale = None
max = 0.01
min = -0.01

for col in range(2):

    # Period 1 : 1950-1979
    # ----------------------
    period1 = div[col].sel(year = slice('1950','1979')).mean('year')    
    resample1 = period1.isel(longitude = slice(None, None, res), 
                            latitude = slice(None, None, res))
    
    fill1 = period1.div.plot(ax = ax[0, col],  
                             transform = tcrs, 
                             cmap = 'RdBu_r',
                             vmin = min, vmax = max, 
                             add_colorbar = False)

    quiver1 = resample1.plot.quiver(x = 'longitude', y = 'latitude',
                              u = 'u',
                              v = 'v',
                              ax = ax[0,col],
                              #width = 0.005, 
                              #headwidth = 4,
                              #scale_units = 'xy',
                              scale = scale,
                              transform = tcrs) 


    # Period 2: 1980 - 2021
    # -----------------------
    period2 = div[col].sel(year = slice('1980','2021')).mean('year') 
    resample2 = period2.isel(longitude = slice(None, None, res), 
                             latitude = slice(None, None, res))
 
    fill2 = period2.div.plot(ax = ax[1, col],
                             transform = tcrs, 
                             cmap = 'RdBu_r',
                             vmin = min, vmax = max,
                             add_colorbar = False)
 
    quiver2 = resample2.plot.quiver(x = 'longitude', y = 'latitude',
                                    u = 'u',
                                    v = 'v',
                                    ax = ax[1,col],
                                    scale = scale,
                                    transform = tcrs) 


    # Difference (period 2 - period 1)
    # ---------------------------------
    dfdiv = period2 - period1
    dfresample = resample2 - resample1   

 
    fill3 = dfdiv.div.plot(ax = ax[2, col],
                           transform = tcrs,
                           cmap = 'RdBu_r',
                           vmin = min, vmax = max,
                           add_colorbar = False, 
                           add_labels = False)

    quiver3 = dfresample.plot.quiver(x = 'longitude', y = 'latitude',
                                     u = 'u',
                                     v = 'v',
                                     ax = ax[2,col],
                                     scale = scale,
                                     transform = tcrs)

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

        pax = ax[row, col]
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


gl = ax[2,0].gridlines(draw_labels = True)
gl.xlabel_style = {'size': 8}
gl.left_labels = False
gl.top_labels = False

gl = ax[2,1].gridlines(draw_labels = True)
gl.xlabel_style = {'size': 8}
gl.left_labels = False
gl.top_labels = False


# Colorbar Settings
# -------------------
cbar = fig.colorbar(fill3, ax = ax[2, :], orientation = 'horizontal') 


ax[0,0].set_title('Pluvial Years')
ax[0,1].set_title('Drought Years') 

plt.savefig('figs/divergence.composites.png', dpi=500)
plt.show()
exit()
