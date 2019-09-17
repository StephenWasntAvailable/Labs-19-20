# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp  
import matplotlib.pyplot as plt

y1= np.array([0.46e-3,0.553e-3,1.5e-3,1.85e-3,2.59e-3,3.11e-3,3.64e-3,4.2e-3,4.68e-3,5.29e-3,6.07e-3,6.76e-3,7.5e-3,8.2e-3,8.85e-3,9.72e-3,10.47e-3,11.41e-3,12.25e-3,13.04e-3,13.86e-3,14.77e-3,15.49e-3,16.23e-3,16.97e-3,17.67e-3,18.5e-3,19.34e-3,20.1e-3,21.0e-3,21.8e-3,23.4e-3,25.6e-3 ])

y = np.zeros(len(y1))
for i in range(0, len(y1)):
    y[i] = y1[i] ** (2/3)

x = np.array([0.53, 1.01, 1.5, 2.0, 2.5, 3, 3.5, 4, 4.5, 5, 5.49, 6.00, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16, 16.3])

plt.plot(x, y, 'bo')

y_err = np.array([])
x_err = np.array([])
#plt.errorbar( x, y, xerr =  x_err, yerr = y_err, fmt = 'o')
#plt.errorbar( x, y, fmt = 'bo', label ='Data')

p = np.polyfit( x, y, 1)
print(p)
slope = p[0]
intercept = p[1]
y_model = slope * x + intercept
plt.plot(x, y_model, 'r', label = 'Fit')


plt.xlabel("")
plt.ylabel("")
plt.title("")
plt.legend(loc = 'upper left')


n = len( x )
s_x = sum( x )
s_y = sum( y )
s_xx = sum( x**2)
s_xy = sum( x*y)
denom = n * s_xx - s_x**2
c = ( s_xx * s_y - s_x * s_xy) / denom
m = ( n * s_xy - s_x * s_y ) /denom

sigma = np.sqrt(sum( ( y -( c + m*x))**2) / (n - 2))
sigma_c = np.sqrt( sigma**2 * s_xx / denom )
sigma_m = np.sqrt( sigma**2 * n /denom )

print('Slope (m):', m, '+-', sigma_m)
print('Intercept (c):', c, '+-', sigma_c)  
