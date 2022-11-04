import numpy as np
from scipy import stats
from scipy import signal
import xarray as xr
import matplotlib.colors as colors
import warnings
import xesmf as xe 

zg = np.load('zg_ghg.npy') # shape = (34, 164, 72, 144)
tas = np.load('tas_ghg.npy') # shape = (34, 164, 72, 144)
nino = np.load('nino_ghg.npy') # shape = (34, 164)

#print(np.shape(zg))
#print(np.shape(tas))
#print(np.shape(nino))

# ========================
#  Perform Trend Analysis
# ========================

# 72x144x34 = 352,512
'''
#     Z250 and NINO4(Y+1)
# --------------------------
n_mod = np.shape(zg)[0]
n_lat = np.shape(zg)[2]
n_lon = np.shape(zg)[3]

z250_slope_array = np.array([])
z250_rvalue_array = np.array([])

for m in range(0, n_mod):
  for lat in range(0, n_lat):
    print('lat =', lat)
    for lon in range(0, n_lon):
      current_cell = zg[m,:-1, lat, lon]
      X = nino[m, 1:]
      Y = current_cell

      s,i,r,p,e = stats.linregress(X, Y)
      z250_slope_array = np.append(z250_slope_array, s)
      z250_rvalue_array = np.append(z250_rvalue_array, r)
      print('z250_slope_array = ', np.shape(z250_rvalue_array))

z250_slope_array = np.save('ghg.z250.slope.npy', z250_slope_array)
z250_rvalue_array = np.save('ghg.z250.rvalue.npy', z250_rvalue_array)
'''

#     SST and NINO4(Y+1)
# -------------------------
n_mod = np.shape(tas)[0]
n_lat = np.shape(tas)[2]
n_lon = np.shape(tas)[3]

tas_slope_array = np.array([])
tas_rvalue_array = np.array([])

for m in range(0, n_mod):
  for lat in range(0, n_lat):
    print('lat =', lat)
    for lon in range(0, n_lon):
      current_cell = tas[m,:-1, lat, lon]
      X = nino[m, 1:]
      Y = current_cell
             
      s,i,r,p,e = stats.linregress(X, Y)
      tas_slope_array = np.append(tas_slope_array, s)
      tas_rvalue_array = np.append(tas_rvalue_array, r)
      print('tas_slope_array = ', np.shape(tas_rvalue_array))
    
tas_slope_array = np.save('ghg.tas.slope.npy', tas_slope_array)
tas_rvalue_array = np.save('ghg.tas.rvalue.npy', tas_rvalue_array)

exit()

