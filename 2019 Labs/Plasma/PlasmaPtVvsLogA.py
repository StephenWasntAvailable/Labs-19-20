#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:16:38 2019

@author: Stephen
"""
# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp  
import matplotlib.pyplot as plt
import math

y1 = np.array([21.4e-6,28.7e-6,35.3e-6,49.8e-6,70.0e-6,134.2e-6,147.1e-6,182.2e-6,256e-6,276e-6,325e-6,364e-6,427e-6,553e-6,603e-6,681e-6,843e-6,1111e-6,1265e-6,1364e-6,1495e-6,1683e-6,1784e-6,1943e-6,2.06e-3,2.18e-3,2.53e-3,2.82e-3,3.10e-3,3.34e-3,3.46e-3,3.55e-3,3.62e-3,3.66e-3,3.74e-3])
y = np.zeros(len(y1))
for i in range(0, len(y1)):
    y[i] = math.log(y1[i])


x = np.array([-2.05,-1.73,-1.52,-1.23,-0.99,-0.588,-0.537,-0.417,-0.238,-0.198,-0.121,-0.048,0.040,0.188,0.239,0.323,0.468,0.672,0.798,0.866,0.959,1.089,1.156,1.259,1.334,1.409,1.628,1.790,1.908,2.11,2.26,2.39,2.58,2.79,3.12])
plt.figure()
plt.plot(x, y, 'bx', label = 'Data')
plt.show()

y3 = np.array([0.3e-6,1.7e-6,2.6e-6,3.8e-6,5.3e-6,7.3e-6,9.2e-6,11.4e-6,14.9e-6,17.9e-6,21.4e-6,28.7e-6,35.3e-6,49.8e-6,70.0e-6,134.2e-6,147.1e-6,182.2e-6,256e-6,276e-6,325e-6,364e-6,427e-6,553e-6,603e-6,681e-6,843e-6,1111e-6,1265e-6,1364e-6,1495e-6,1683e-6,1784e-6,1943e-6,2.06e-3,2.18e-3,2.53e-3,2.82e-3,3.10e-3,3.34e-3,3.46e-3,3.55e-3,3.62e-3,3.66e-3,3.74e-3,3.84e-3,3.97e-3,4.18e-3,4.36e-3,4.58e-3,4.52e-3,4.61e-3,4.77e-3,4.83e-3,5.02e-3,5.46e-3,5.86e-3,6.47e-3,7.06e-3,7.71e-3,8.44e-3,9.16e-3])
y2 = np.zeros(len(y3))
for i in range(0, len(y3)):
    y2[i] = math.log(y3[i])

x2 = np.array([-4.49,-4.23,-4.01,-3.77,-3.52,-3.24,-3.01,-2.78,-2.47,-2.26,-2.05,-1.73,-1.52,-1.23,-0.99,-0.588,-0.537,-0.417,-0.238,-0.198,-0.121,-0.048,0.040,0.188,0.239,0.323,0.468,0.672,0.798,0.866,0.959,1.089,1.156,1.259,1.334,1.409,1.628,1.790,1.908,2.11,2.26,2.39,2.58,2.79,3.12,3.56,4.14,5.09,6.06,7.06,7.98,9.07,10.06,11.04,12.12,12.98,13.46,14.07,14.55,15.06,15.51,16.13])

plt.figure()
plt.plot(x2, y2, 'bx', label = 'Data')
plt.xlabel("Voltage (V)")
plt.ylabel("log(I)")
plt.title("Natural log of Current vs. Voltage")
plt.legend(loc = 'upper left')
plt.grid(b = 'true')
plt.show()

y4 = np.array([1.7e-6,2.6e-6,3.8e-6,5.3e-6,7.3e-6,9.2e-6,11.4e-6,14.9e-6,17.9e-6,21.4e-6,28.7e-6,35.3e-6,49.8e-6,70.0e-6,134.2e-6,147.1e-6,182.2e-6,256e-6,276e-6,325e-6,364e-6,427e-6,553e-6,603e-6,681e-6,843e-6,1111e-6,1265e-6,1364e-6,1495e-6,1683e-6,1784e-6,1943e-6,2.06e-3,2.18e-3,2.53e-3,2.82e-3,3.10e-3,3.34e-3])
y5 = np.zeros(len(y4))
for i in range(0, len(y4)):
    y5[i] = math.log(y4[i])
x3 = np.array([-4.23,-4.01,-3.77,-3.52,-3.24,-3.01,-2.78,-2.47,-2.26,-2.05,-1.73,-1.52,-1.23,-0.99,-0.588,-0.537,-0.417,-0.238,-0.198,-0.121,-0.048,0.040,0.188,0.239,0.323,0.468,0.672,0.798,0.866,0.959,1.089,1.156,1.259,1.334,1.409,1.628,1.790,1.908,2.11])

p = np.polyfit( x3, y5, 1)
print(p)
slope = p[0]
intercept = p[1]
y_model = slope * x3 + intercept

plt.figure()
plt.plot(x3, y5, 'bx', label = 'Data')
plt.plot(x3, y_model, 'r', label = 'Fit')
plt.xlabel("Voltage (V)")
plt.ylabel("log(I)")
plt.title("Natural log of Current vs. Voltage Trimmed")
plt.legend(loc = 'upper left')
plt.grid(b = 'true')
plt.show()

n = len( x3 )
s_x = sum( x3 )
s_y = sum( y5 )
s_xx = sum( x3**2)
s_xy = sum( x3*y5)
denom = n * s_xx - s_x**2
c = ( s_xx * s_y - s_x * s_xy) / denom
m = ( n * s_xy - s_x * s_y ) /denom

sigma = np.sqrt(sum( ( y5 -( c + m*x3))**2) / (n - 2))
sigma_c = np.sqrt( sigma**2 * s_xx / denom )
sigma_m = np.sqrt( sigma**2 * n /denom )

print('Slope (m):', m, '+-', sigma_m)
print('Intercept (c):', c, '+-', sigma_c)  



