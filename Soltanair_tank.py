import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from bokeh.io import show
from bokeh.plotting import figure

L = 2.8
n = 300
dx = L/n
dt = 0.2
t_final = 12000

Coef_losses = 0.8045
k_eff = 1.269
u = 0.542
pCp_eff = 1730714
pCp_air = 827.8368
air_rock = 0.35
m_aire = 0.265
Cp_aire = 1044

T_in = 273.5 + 170
T0 = 273.5 + 60
T_ext = 273.5 + 20
T_goal = 273.5 + 150

x = np.linspace(0,L,n+1)
T = np.ones(n+1)*T0
A = np.ones(n+1)
B = np.ones(n+1)
Q_losses = np.ones(n+1)
dTdt = np.ones(n+1)
t = np.arange(0, t_final, dt)

xlen = len(x)
Tlen = len(T)
dTdtlen = len(dTdt)
tlen = len(t)
j= 0

# for j in range(1, len(t)):
while T[-1] < T_goal:
    j = j+1
    # if j<30000:
    T[0] = T_in
    # if 29999 < j <60000:
    #     u = 0
    # if j == 59999:
    #     T = list(reversed(T))
    #     T[0] = 60+273.5
    # if j>59999:
    #     T[0] = 60+273.5

    for i in range(1, n):
        Q_losses[i] = + Coef_losses*(T[i]-T_ext)
        A[i] = k_eff*(T[i+1]-2*T[i]+T[i-1])/dx**2
        B[i] = air_rock*pCp_air*u*(T[i+1]-T[i])/dx
        dTdt[i] = (-B[i]+A[i]+Q_losses[i])/pCp_eff
    # dTdt[0] = ((dx/k_eff*T_in+T[1]/m_aire/Cp_aire) - T[0])/dt
    dTdt[0] = 0
    dTdt[n] = (k_eff*(-T[n]+T[n-1])/dx**2 + Coef_losses*(T[n]-T_ext))/pCp_eff
    #dTdt[n] = 0
    T = T + dTdt*dt
    
    # if j == 83020  or j == 41510:
    #     plt.figure(1)
    #     plt.plot(x, T)
    #     plt.axis([0, L, 273.5, 600])
    #     plt.xlabel('Distance(m)')
    #     plt.ylabel('Temperature(K)')
    #     plt.show()
    #     plt.pause(0.02)
    #     plt.close(1)
plt.figure(1)
plt.plot(x, T)
plt.axis([0, L, 273.5, 600])
plt.xlabel('Distance(m)')
plt.ylabel('Temperature(K)')
plt.show()
plt.pause(0.02)
plt.close(1)

final_time = j*dt
print(T[-1], final_time)

tempfield = pd.DataFrame( T, columns = ['Temperature'])
tempfield.to_csv('Thermocline.csv', index = True)

# # create a new plot (with a title) using figure
#     p = figure(width=400, height=400, title="My Line Plot")

# # add a line renderer
#     p.line(x, T, line_width=2)

#     if j%300 == 0 or j == 1:
#         show(p) # show the results