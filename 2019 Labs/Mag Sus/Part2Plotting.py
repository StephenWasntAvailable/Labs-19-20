# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp  
import matplotlib.pyplot as plt

y1= np.array([])

y = np.array([0,
0,
2.943E-06,
8.829E-06,
1.5696E-05,
2.4525E-05,
3.2373E-05,
4.1202E-05,
5.0031E-05,
5.886E-05,
6.7689E-05,
7.6518E-05,
8.2404E-05,
9.1233E-05,
9.6138E-05,
0.000102024,
0.000105948
])
x = np.array([0.005329,
0.018769,
0.042025,
0.075076,
0.116281,
0.163216,
0.217156,
0.272484,
0.334084,
0.3969,
0.461041,
0.519841,
0.576081,
0.627264,
0.674041,
0.717409,
0.758641
])

plt.plot(x, y, 'bx', label = 'Data')

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
x_err = np.array([0.000146,
0.000274,
0.00041,
0.000548,
0.000682,
0.000808,
0.000932,
0.001044,
0.001156,
0.00126,
0.001358,
0.001442,
0.001518,
0.001584,
0.001642,
0.001694,
0.001742
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
