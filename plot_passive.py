import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from matplotlib.lines import Line2D

rms_window_size = 200 # Define the window size for calculating RMS values
degree = 3 # Polynomial fitting degree

with h5py.File('MLogs/10_LFT_34gg_2kHz_UpsideDown_26-July-2023_21-17-32/data.mat', 'r') as file:
    data = {}
    for key, value in file.items():
        if len(value.shape) == 1:
            data[key] = value[:]
        else:
            for i in range(value.shape[1]):
                data[f"{key}_{i+1}"] = value[:, i]

df = pd.DataFrame(data)
time = df.iloc[0, :]
cmd_voltage = df.iloc[1, :]
displacement = df.iloc[2, :]
ss_voltage = df.iloc[3, :]

# Factor for laser displacement and reference
displacement = displacement * 4 + 30
displacement = displacement - np.mean(displacement.iloc[:50])

num_samples = len(time)
num_rms = num_samples // rms_window_size
adjusted_time = np.zeros(num_rms)
adj_displacement = np.zeros(num_rms)
rms_voltage_values = np.zeros(num_rms)

# Calculate RMS adjusted voltage, time and displacement values where each data point is an average of rms_window_size data points from the raw data
for i in range(num_rms):
    start_index = i * rms_window_size
    end_index = (i + 1) * rms_window_size - 1   
    rms_voltage_values[i] = np.sqrt(np.mean(ss_voltage.iloc[start_index:end_index] ** 2))
    adjusted_time[i] = np.mean(time.iloc[start_index:end_index])
    adj_displacement[i] = np.mean(displacement.iloc[start_index:end_index])

# convert rms_voltage_values to pandas Series for polyfit
rms_voltage_values = pd.Series(rms_voltage_values)

# get estimated displacement via polynomial fitting
coefficients = np.polyfit(rms_voltage_values, adj_displacement, degree)
est_displ = np.polyval(coefficients, rms_voltage_values)

# offset in y direction
offset = np.min(adj_displacement)
adj_displacement -= offset
est_displ -= offset

# cut unnecessary data points on left and right side
adj_displacement = adj_displacement[(adjusted_time >= 4.4) & (adjusted_time < 16)]
est_displ = est_displ[(adjusted_time >= 4.4) & (adjusted_time < 16)]
adjusted_time = adjusted_time[(adjusted_time >= 4.4) & (adjusted_time < 16)]
adjusted_time -= 4.4

plt.rc('text', usetex=True)
plt.rc('font', family='serif', weight='bold')
plt.rcParams['axes.labelsize'] = 25   
plt.rcParams['legend.fontsize'] = 8  
plt.rcParams['xtick.labelsize'] = 20 
plt.rcParams['ytick.labelsize'] = 20  
plt.rcParams['grid.linewidth'] = 1.5
plt.rcParams['axes.linewidth'] = 1.5

line_width = 2.5
fig, axs = plt.subplots(1, 1, figsize=(16, 9)) 

axs.plot(adjusted_time, adj_displacement, linewidth=line_width, color='r')
axs.plot(adjusted_time, est_displ, linewidth=line_width, color='#1f77b4')
axs.set_xlabel(r'Time (s)', weight='bold') 
axs.set_ylabel(r'Displacement (mm)') 
axs.grid(True) 

legend_elements = [
    Line2D([0], [0], color='#1f77b4', lw=2, label='Estim. Displacement (voltage-method)'),
    Line2D([0], [0], color='r', lw=2, label='Ground Truth Displacement'),
]

fig.legend(handles=legend_elements, loc='upper center', handlelength=2,ncol=7, bbox_to_anchor=(0.5, 1.01), fontsize=18)

fig.subplots_adjust(
    top=0.925,
    bottom=0.615,
    left=0.28,
    right=0.72,
    hspace=0.2,
    wspace=0.105
)

plt.savefig('Final-Plot-9.pdf')
plt.show()