import numpy as np
import matplotlib.pyplot as plt

from bokeh.io import curdoc, show
from bokeh.models import Range1d
from bokeh.plotting import figure
from bokeh.layouts import column

'''Datos del tanque'''

L = 1                   # m
D = 0.95                # m
V = np.pi*(D/2)**2*L    # m3

air_rock = 0.35         
Cp_roca = 1085          # J/kgK
p_roca = 2400           # kg/m3
k_roca = 1.7            # W/mK

Cp_aire = 1016          # J/kgK
p_aire = 0.8148         # kg/m3
k_aire = 0.03511        # W/mK
Q_aire = 0.2            # kg/s

Cp_wall = 500           # J/kgK
p_wall = 8000           # kg/m3
k_wall = 16             # W/mK
thick_wall = 0.003      # m
V_wall = np.pi*L*((D/2+thick_wall)**2 - (D/2)**2)     # m3

''' Pérdidas térmicas '''

thick_aisl = 0.06                   # m
r_aisl = D/2 + thick_aisl + thick_wall      # m
k_aisl = 0.047                      # W/mK                
V_tot = np.pi*r_aisl**2*L           # m3          

T_ext = 273.5 + 20                  # K
p_ext = 1.204                       # kg/m3
visc_dinam_ext = 1.825 * 10**-5     # kg/ms
veloc_ext = 1                       # m/s
k_ext = 0.02514                     # W/mk
Pr_ext = 0.7309
Re_ext = p_ext * veloc_ext * L / visc_dinam_ext
h_ext = 0.664*Re_ext**0.5*Pr_ext**0.5*k_ext/L        # W/m2K
A_ext = 2*np.pi*(D/2+thick_wall+thick_aisl)*L        # m2
R_th = np.log(r_aisl/(D/2 + thick_wall)) / (2*np.pi*k_aisl*L) + 1/(h_ext*2*np.pi*r_aisl*L)     # K/W
h_th = 1/(R_th*2*np.pi*r_aisl*L)                     # W/m2K
Coef_losses = h_th*A_ext/V_tot         # W/m3K

'''Configuración de la simulación'''

n = 400
dx = L/n
dt = 0.15
t_final = 35800     # s
t_standby = 7200    # s
last_min = 0        # s

k_eff = air_rock*k_aire + (1-air_rock)*k_roca + V_wall/V*k_wall     # W/mK
u = Q_aire/(p_aire*air_rock*np.pi*(D/2)**2)                         # m/s

pCp_air = Cp_aire * p_aire      # J/m3K
pCp_rock = Cp_roca * p_roca     # J/m3K
pCp_wall = Cp_wall * p_wall     # J/m3K
pCp_eff = air_rock*pCp_air + (1 - air_rock) * pCp_rock + V_wall/V * pCp_wall     # J/m3K

T_in_charge = 273.5 + 170       # K
T0 = 273.5 + 60                 # K
T_goal_charge = 273.5 + 70     # K
T_in_discharge = 273.5 + 60     # K
T_goal_discharge = 273.5 + 80   # K


'''SIMULACIÓN'''

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
count_standby = 0

def celsius(T):
    T_celsius = [temp - 273.15 for temp in T]
    return T_celsius

def plot(x, T, mode, minute):
    global last_min, last_T
    index = 0
    plt.figure(figsize=(6, 4), layout='constrained')
    if mode == "discharge":
        x_plot = x
        x_last_plot = list(reversed(x))
        plt.plot(x_last_plot, last_T, label = str(last_min) + " min")
    else:
        x_plot = list(reversed(x))
    if mode == "standby":
        plt.plot(x_plot, last_T, label = str(last_min) + " min")
    for i in T:
        minute_plot = round(minute[index], 1)
        plt.plot(x_plot, i, label = str(minute_plot) + " min")
        index += 1
    plt.axis([L, 0, 0, 220])
    plt.xlabel('Height of the tank (m)')
    plt.ylabel('Temperature (ºC)')
    duration = round(minute[-1]-last_min, 2)
    plt.title("Mode: "+ mode + ", duration: " + str(duration) + " min")
    plt.legend(loc = 'lower left')
    plt.show()
    last_T = T[-1]
    last_min = round(minute[-1], 1)

def temp_field (T, mode, s):
    global tlen, u
    my_Tplot = []
    my_timeplot = []
    if mode == "charge":
        vel = u
        T[0] = T_in_charge
    if mode == "standby":
        vel = 0
        tlen = s + int(t_standby/dt)
    if mode == "discharge":
        T = list(reversed(T))
        vel= u
        T[0] = T_in_discharge
        tlen = len(t)
    for j in range(s, tlen):
        for i in range(1, n):
            Q_losses[i] = + Coef_losses*(T[i]-T_ext)
            A[i] = k_eff*(T[i+1]-2*T[i]+T[i-1])/dx**2
            B[i] = air_rock*pCp_air*vel*(T[i+1]-T[i])/dx
            dTdt[i] = (-B[i]+A[i]+Q_losses[i])/pCp_eff
        # dTdt[0] = ((dx/k_eff*T_in+T[1]/m_aire/Cp_aire) - T[0])/dt    NECESARIO PARA QUE HAGA EFECTO LAS PÉRDIDAS TÉRMICAS
        dTdt[0] = 0
        dTdt[n] = (k_eff*(-T[n]+T[n-1])/dx**2 + Coef_losses*(T[n]-T_ext))/pCp_eff
        #dTdt[n] = 0
        T = T + dTdt*dt
        
        if j%16000 == 0:
            my_Tplot.append(celsius(T))
            my_timeplot.append(j*dt/60)

        if mode == "charge" and T[-1] >= T_goal_charge:
            break
        if mode == "discharge" and T[-1] <= T_goal_discharge:
            break
    tim = str(round((j - s)*dt, 2))
    hour = str(round((j - s)*dt/3600, 2))
    my_Tplot.append(celsius(T))
    my_timeplot.append(j*dt/60)
    print ("El tiempo de " + mode + " ha sido de " + tim + " segundos, o de " + hour + " horas")

    return [T, j, my_Tplot, my_timeplot]


[T, s, my_Tplot, my_timeplot] = temp_field(T, "charge", 0)
plot_charge = plot(x, my_Tplot, "charge", my_timeplot)
[T, s, my_Tplot, my_timeplot] = temp_field(T, "standby", s)
plot_standby = plot(x, my_Tplot, "standby", my_timeplot)
[T, j, my_Tplot, my_timeplot] = temp_field(T, "discharge", s)
plot_discharge = plot(x, my_Tplot, "discharge", my_timeplot)
    
'''GRÁFICAS'''

    # p = figure(x_range=(0, L), y_range=(273.5, 500), x_axis_label='Distance(m)', y_axis_label='Temperature(K)')
    # line = p.line(x, T)
    # layout = column(p)
    # show (layout, browser= 'firefox', notebook_handle=True)

    # if j % 300 == 0 or j == 1:
    #     line.data_source.data['y'] = T
    #     time.sleep(10)
    #     curdoc().title = "Temperature Plot"    
