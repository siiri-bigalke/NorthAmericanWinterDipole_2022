import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


dipole = np.load('dipole.npy')
#dpi2 = np.load('1960-2014.dipole.npy')
dpi2 = np.load('1950-2020.dipole.npy')

#x = np.arange(1960,2014)
x = np.arange(1951,2021)

#plt.plot(x,dipole)
plt.plot(x, dpi2)
plt.show()
