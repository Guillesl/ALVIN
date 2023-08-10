import pandas as pd
import matplotlib.pyplot as plt

csv_file_path = 'Piloto_potencia_solar.csv'
column_name = 'Potencia_kW'

data_frame = pd.read_csv(csv_file_path)

selected_column = data_frame[column_name]
potencia_solar = selected_column.tolist()
potencia_solar.pop(0)

T_out_tank = 60
Q_aire = 0.25
Cp_aire = 1008
T_in_design= 170

T_iniciales = []
Q_aire_dia = []

for i in potencia_solar:
    T_in = i*1000/(Q_aire*Cp_aire)+T_out_tank
    Q_aire_control = i*1000/((T_in_design - T_out_tank)*Cp_aire)
    T_iniciales.append(T_in)
    Q_aire_dia.append(Q_aire_control)

# Plot the calculated initial temperatures
plt.plot(T_iniciales)
plt.xlabel('Minute of the day')
plt.ylabel('Initial Temperature (ºC)')
plt.title('Inlet temperature to tank, fixed air flow 0.25 kg/s')
plt.grid(True)
plt.show()

plt.plot(Q_aire_dia)
plt.xlabel('Minute of the day')
plt.ylabel('Initial Temperature (K)')
plt.title('Air flow for inlet temp 170ºC')
plt.grid(True)
plt.show()