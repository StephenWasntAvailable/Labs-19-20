# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp  
import matplotlib.pyplot as plt

y1= np.array([])

y = np.array([0,
0,
6.867E-06,
1.2753E-05,
2.0601E-05,
2.943E-05,
4.1202E-05,
5.1012E-05,
6.1803E-05,
7.3575E-05,
8.4366E-05,
9.5157E-05,
0.000104967,
0.000112815,
0.000119682,
0.00012753,
0.000134397
])
x = np.array([0.005184,
0.018496,
0.040804,
0.072361,
0.110889,
0.157609,
0.209764,
0.266256,
0.326041,
0.386884,
0.4489,
0.508369,
0.565504,
0.616225,
0.664225,
0.707281,
0.748225
])

plt.plot(x, y, 'bx', label = 'Data')
#plt.xlim(-1,1)
#plt.ylim(-1, 1)

y_err = np.array([9.81E-07,
9.81E-07,
9.81E-07,
9.81E-07,
9.81E-07,
9.81E-07,
9.81E-07,
9.81E-07,
9.81E-07,
9.81E-07,
9.81E-07,
9.81E-07,
9.81E-07,
9.81E-07,
9.81E-07,
9.81E-07,
9.81E-07
])
x_err = np.array([0.000144,
0.000272,
0.000404,
0.000538,
0.000666,
0.000794,
0.000916,
0.001032,
0.001142,
0.001244,
0.00134,
0.001426,
0.001504,
0.00157,
0.00163,
0.001682,
0.00173
])
plt.errorbar( x, y, xerr =  x_err, yerr = y_err, fmt = '.', label = 'Error')


p = np.polyfit( x, y, 1)
print(p)
slope = p[0]
intercept = p[1]
y_model = slope * x + intercept
plt.plot(x, y_model, 'r', label = 'Fit')


plt.xlabel("Field^2 (T^2)")
plt.ylabel("Force (N)")
plt.title("Vertical Force Experienced by Sample vs. Field Squared", pad = 20)
plt.legend(loc = 'upper left')
plt.grid()
plt.tick_params(axis = 'both', which = 'both', right = True, top = True, labeltop = True, labelright = True)



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
