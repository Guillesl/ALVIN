import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from bokeh.io import curdoc, show
from bokeh.models import Range1d
from bokeh.plotting import figure
from bokeh.layouts import column


csv_file_path = 'Piloto_potencia_solar_v2.csv'
column_name = 'Potencia_kW'

data_frame = pd.read_csv(csv_file_path)

selected_column = data_frame[column_name]
potencia_solar = selected_column.tolist()
potencia_solar.pop(0)

'''Lado de agua del campo solar'''
m_agua = 0.2                       # kg/s
Cp_agua = 4300                      # J/kgK
T_w_out_dis = 105 + 273.15          # K
T_w_in_dis = 95 + 273.15           # K
# DT_agua_aire = 5                    # K

'''Datos del tanque'''

L = 1                   # m
D = 0.95                # m
V = np.pi*(D/2)**2*L    # m3

air_rock = 0.35         
Cp_roca = 1085          # J/kgK
p_roca = 2400           # kg/m3
k_roca = 0.96           # W/mK

Cp_aire = 1016          # J/kgK
p_aire = 0.8148         # kg/m3
k_aire = 0.03511        # W/mK
m_aire = 0.15           # kg/s

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

T_ext = 273.15 + 20                  # K
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

n = round(250*L)
dx = L/n
dt = 1              # Use dt multiples of 60, ex: 0.25, 0.5, 1, 2, 5, 10, not 7
potencia_solar = [item for item in potencia_solar for _ in range(int(60/dt))] # The file timestep is 60 seconds
t_final = 135800     # s
t_standby = 14400    # s
last_min = 0        # s
control = "caudal_air"  # Puede ser "caudal aire" si queremos estar siempre en la T_in_design

k_eff = air_rock*k_aire + (1-air_rock)*k_roca + V_wall/V*k_wall     # W/mK
u = m_aire/(p_aire*air_rock*np.pi*(D/2)**2)                         # m/s

pCp_air = Cp_aire * p_aire      # J/m3K
pCp_rock = Cp_roca * p_roca     # J/m3K
pCp_wall = Cp_wall * p_wall     # J/m3K
pCp_eff = air_rock*pCp_air + (1 - air_rock) * pCp_rock + V_wall/V * pCp_wall     # J/m3K

T0 = 273.15 + 60                 # K
# T_in_charge = 273.15 + 170       # K
# T_in_discharge = 273.15 + 60     # K
T_in_design= 170 + 273.15        # K
T_goal_charge = 273.15 + 70     # K
T_goal_discharge = T_w_out_dis + 10   # K



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

# Ploting energy accumulated, lost, air flow and T_in
def plot_temp_flow(T_iniciales, m_aire_dia, Stored_energy_acc, Storage_power_acc, Lost_power_acc, Lost_energy_acc):
    time_charge_minutes = [idx * dt / 60 for idx in range(len(Stored_energy_acc))]
    plt.plot(time_charge_minutes, T_iniciales)
    plt.xlabel('Minute of the day')
    plt.ylabel('Initial Temperature (ºC)')
    plt.title(f'Inlet temperature to tank, fixed air flow {m_aire} kg/s')
    plt.grid(True)
    plt.show()

    plt.plot(time_charge_minutes, m_aire_dia)
    plt.xlabel('Minute of the day')
    plt.ylabel('Air flow kg/s')
    plt.title('Air flow for inlet temp 170ºC')
    plt.grid(True)
    plt.show()

    plt.plot(time_charge_minutes, Lost_energy_acc, label='Lost Energy')
    plt.plot(time_charge_minutes, Stored_energy_acc, label='Stored Energy')
    plt.xlabel('Minute of the day')
    plt.ylabel('Energy (kWh)')
    plt.title('Lost and Stored Energy')
    plt.legend()
    plt.grid(True)
    plt.text(0, 22, 'Stored energy: ' + str(round(Stored_energy_acc[-1], 2)) + ' kWh', fontsize=10)
    plt.text(0, 20, 'Lost energy: ' + str(round(Lost_energy_acc[-1], 2)) + ' kWh', fontsize=10)
    plt.show()

    plt.plot(time_charge_minutes, Lost_power_acc, label='Lost power')
    plt.plot(time_charge_minutes, Storage_power_acc, label='Storage power')
    plt.xlabel('Minute of the day')
    plt.ylabel('Power (kW)')
    plt.title('Lost and Stored Energy')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_disch(Discharged_power_acc):
    time_discharge_minutes = [idx * dt / 60 for idx in range(len(Discharged_power_acc))]
    plt.plot(time_discharge_minutes, Discharged_power_acc)
    plt.xlabel('Minute of the day')
    plt.ylabel('Power (kW)')
    plt.title('Discharge power at fixed m_aire')
    plt.grid(True)
    # plt.show()

def temp_field (T, mode, s, m_aire):
    global tlen, u
    Lost_energy = 0
    Lost_power = 0
    Stored_energy = 0
    Storage_power = 0
    Discharged_power = 0
    Discharged_energy = 0
    T_iniciales = []
    m_aire_dia = []
    my_Tplot = []
    my_timeplot = []
    Lost_power_acc = []
    Lost_energy_acc = []
    Storage_power_acc = []
    Stored_energy_acc = []
    Discharged_power_acc = []
    Discharged_energy_acc = []

    if mode == "charge":
        vel = u
        # T[0] = T_in_charge
    if mode == "standby":
        vel = 0
        tlen = s + int(t_standby/dt)
    if mode == "discharge":
        T = list(reversed(T))
        vel= u
        # T[0] = T_in_discharge
        tlen = len(t)
    for j in range(s, tlen):

        if mode == "charge":

            m_aire_control = potencia_solar[j]*1000/((T_in_design - T[-1])*Cp_aire)
            if control == "caudal_aire":
                T[0] = T_in_design                                  # K
                m_aire = m_aire_control                             # kg/s     
                vel = m_aire/(p_aire*air_rock*np.pi*(D/2)**2)         # m/s
            else:
                T[0] = potencia_solar[j]*1000/(m_aire*Cp_aire)+T[-1]
                if T[0] >= T_in_design:
                    T[0] = T_in_design

            Lost_power = m_aire*Cp_aire*(T[0]-T_in_design)/1000    # kW
            Lost_energy += Lost_power*dt/(3600/dt)                 # kWh
            Storage_power = potencia_solar[j]-Lost_power
            Stored_energy += Storage_power*dt/(3600/dt)
            
            T_iniciales.append(T[0]-273.15)
            m_aire_dia.append(m_aire_control)
            Lost_power_acc.append(Lost_power)
            Lost_energy_acc.append(Lost_energy)
            Storage_power_acc.append(Storage_power)
            Stored_energy_acc.append(Stored_energy)

        if mode == "discharge":
            if control == "caudal_aire":
                T[0] = T_w_in_dis
                m_aire_control = m_agua*Cp_agua*(T_w_out_dis - T_w_in_dis)/(Cp_aire*(T[-1]-T[0]))
                m_aire = m_aire_control                             # kg/s
                
                vel = m_aire/(p_aire*air_rock*np.pi*(D/2)**2)         # m/s
                # print(m_aire, vel)

            Discharged_power = m_aire*Cp_aire*(T[-1]-T[0])/1000    # kW
            Discharged_power_acc.append(Discharged_power)
            Discharged_energy += Discharged_power*dt/(3600/dt)
            Discharged_energy_acc.append(Discharged_energy)

        for i in range(1, n):
            Q_losses[i] = - Coef_losses*(T[i]-T_ext)
            A[i] = k_eff*(T[i+1]-2*T[i]+T[i-1])/dx**2
            B[i] = air_rock*pCp_air*vel*(T[i+1]-T[i])/dx
            dTdt[i] = (-B[i]+A[i]+Q_losses[i])/pCp_eff
        dTdt[0] = 0 #(-Coef_losses*(T[0]-T_ext))/pCp_eff            # DEBERÍA LLEVAR K_eff ?               
        dTdt[n] = (k_eff*(-T[n]+T[n-1])/dx**2 - Coef_losses*(T[n]-T_ext))/pCp_eff
        T = T + dTdt*dt
        
        if j%4000 == 0:
            my_Tplot.append(celsius(T))
            my_timeplot.append(j*dt/60)

        if mode == "charge" and T[-1] >= T_goal_charge:
            break
        if mode == "discharge" and (T[-1] <= T_goal_discharge or m_aire > 0.25):
            break
    tim = str(round((j - s)*dt, 2))
    hour = str(round((j - s)*dt/3600, 2))
    my_Tplot.append(celsius(T))
    my_timeplot.append(j*dt/60)
    print ("El tiempo de " + mode + " ha sido de " + tim + " segundos, o de " + hour + " horas")

    return [T, j, my_Tplot, my_timeplot, T_iniciales, m_aire_dia, Stored_energy_acc, Storage_power_acc, Lost_power_acc, Lost_energy_acc, Discharged_power_acc, Discharged_energy_acc]


[T, s, my_Tplot, my_timeplot, T_iniciales, m_aire_dia, Stored_energy_acc, Storage_power_acc, Lost_power_acc, Lost_energy_acc, Discharged_power_acc, Discharged_energy_acc] = temp_field(T, "charge", 0, m_aire)
plot_charge = plot(x, my_Tplot, "charge", my_timeplot)
plot_temp_flow_charge = plot_temp_flow(T_iniciales, m_aire_dia, Stored_energy_acc, Storage_power_acc, Lost_power_acc, Lost_energy_acc)
[T, s, my_Tplot, my_timeplot, T_iniciales, m_aire_dia, Stored_energy_acc, Storage_power_acc, Lost_power_acc, Lost_energy_acc, Discharged_power_acc, Discharged_energy_acc] = temp_field(T, "standby", s, m_aire)
plot_standby = plot(x, my_Tplot, "standby", my_timeplot)
[T, s, my_Tplot, my_timeplot, T_iniciales, m_aire_dia, Stored_energy_acc, Storage_power_acc, Lost_power_acc, Lost_energy_acc, Discharged_power_acc, Discharged_energy_acc] = temp_field(T, "discharge", s, m_aire)
plot_discharge = plot(x, my_Tplot, "discharge", my_timeplot)
plot_temp_flow_discharge = plot_disch(Discharged_power_acc)
print (Discharged_energy_acc[-1])
    
