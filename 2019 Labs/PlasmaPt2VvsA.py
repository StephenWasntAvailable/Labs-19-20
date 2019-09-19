#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:43:36 2019

@author: Stephen
"""

# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp  
import matplotlib.pyplot as plt

y1= np.array([0.374e-3,0.646e-3,0.980e-3,1.320e-3,1.840e-3,2.43e-3,2.98e-3,3.53e-3,4.04e-3,4.59e-3,5.19e-3,5.84e-3,6.65e-3,7.38e-3,8.08e-3,8.80e-3,9.63e-3,10.34e-3,11.16e-3,11.93e-3,12.72e-3,13.74e-3,14.53e-3,14.98e-3,16.31e-3,17.03e-3,17.80e-3,18.60e-3,19.28e-3,20.3e-3,20.8e-3,21.8e-3,23.1e-3,27.4e-3,31.6e-3,42.4e-3,52.9e-3])

y = np.zeros(len(y1))
for i in range(0, len(y1)):
    y[i] = y1[i] ** (2/3)

x = np.array([0.033,0.490,1.000,1.490,2.00,2.50,3.00,3.50,4.00,4.50,5.00,5.50,6.00,6.50,7.00,7.50,8.00,8.50,8.97,9.50,10.0,10.50,11.00,11.50,12.00,12.50,13.00,13.50,14.00,14.50,15.00,15.49,16.00,16.50,17.00,17.49,18.00])

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
