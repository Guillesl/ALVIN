import numpy as np
import matplotlib.pyplot as plt

from bokeh.io import show
from bokeh.plotting import figure

L = 0.5
n = 100
dx = L/n
dt = 0.1
t_final = 9000

Coef_losses = 1.283
k_eff = 1.375
u = 0.728
pCp_eff = 1757402
pCp_air = 827
air_rock = 0.35
m_aire = 0.2
Cp_aire = 1000

T_in = 273.5 + 170
T0 = 273.5 + 60
T_ext = 273.5 + 20

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

for j in range(1, len(t)):
    # if j<3000:
    #     T[0] = T_in
    # if 2999 < j <6000:
    #     u = 0
    # else:
    #     T = list(reversed(T))
    #     T[0] = T_in

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
    T[0] = T_in
    if j%3000 == 0 or j == 1:
        plt.figure(1)
        plt.plot(x, T)
        plt.axis([0, L, 273.5, 500])
        plt.xlabel('Distance(m)')
        plt.ylabel('Temperature(K)')
        plt.show()
        plt.pause(0.02)
        plt.close(1)
    
# # create a new plot (with a title) using figure
#     p = figure(width=400, height=400, title="My Line Plot")

# # add a line renderer
#     p.line(x, T, line_width=2)

#     if j%300 == 0 or j == 1:
#         show(p) # show the results