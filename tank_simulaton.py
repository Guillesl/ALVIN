import numpy as np
import matplotlib.pyplot as plt

L = 1
n = 100
dx = L/n
dt = 0.01
t_final = 30

Coef_losses = 1.283
k_eff = 1.375
u = 0.728
pCp_eff = 1757402
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
    for i in range(1, n):
        Q_losses[i] = - Coef_losses*(T[i]-T_ext)
        A[i] = k_eff*(T[i+1]-2*T[i]+T[i-1])/dx**2
        B[i] = air_rock*pCp_eff*u*(T[i+1]-T[i])/dx
        dTdt[i] = (-B[i]+A[i]+Q_losses[i])/pCp_eff
    dTdt[0] = ((dx/k_eff*T_in+T[1]/m_aire/Cp_aire) - T[0])/dt
    dTdt[n] = - Coef_losses*(T[n-1]-T_ext)/pCp_eff
    T = T + dTdt*dt

