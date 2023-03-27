import numpy as np
import matplotlib.pyplot as plt

L = 1
n = 1000
dx = L/n
dt = 0.01
t_final = 120

Coef_losses = 1.283
k_eff = 1.375
u = 0.728
pCp_eff = 1757402
air_rock = 0.35

T0 = [170]+[60]*999
T_ext = 20
TL = Coef_losses*(T_ext-T)*dt/pCp_eff + T

x = np.linspace(0,L,n)
T = np.ones(n)*T0
dTdt = np.empty(n)
t = np.arange(0, t_final, dt)