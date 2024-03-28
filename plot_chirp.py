import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from matplotlib.lines import Line2D
from scipy.signal import correlate
from scipy import signal
from scipy.optimize import curve_fit

rms_window_size = 200 # Define the window size for calculating RMS values
degree = 3 # Polynomial fitting degree

with h5py.File('MLogs/10_Vogt_34gg_2kHz_TestPCB_26-July-2023_15-36-24/data.mat', 'r') as file:
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

#for the polynomial fit, which gives us the estimated displacement, we ignore the step signal
adj_displacement_for_fit = adj_displacement[(adjusted_time >= 3.45) & (adjusted_time < 24)]
rms_voltage_values_for_fit = rms_voltage_values[(adjusted_time >= 3.45) & (adjusted_time < 24)]

# for plotting the data though we include also the step signal
adj_displacement = adj_displacement[(adjusted_time >= 1.6) & (adjusted_time < 24)]
rms_voltage_values = rms_voltage_values[(adjusted_time >= 1.6) & (adjusted_time < 24)]
adjusted_time = adjusted_time[(adjusted_time >= 1.6) & (adjusted_time < 24)]
adjusted_time -= 1.6

# get estimated displacement via polynomial fitting
coefficients = np.polyfit(rms_voltage_values_for_fit, adj_displacement_for_fit, degree)
est_displ = np.polyval(coefficients, rms_voltage_values)

# offset in y direction
offset = np.min(adj_displacement)
adj_displacement -= offset
est_displ -= offset

# define the time intervals with one particular frequency by hand
interval_05hz = (adjusted_time > 1.5) & (adjusted_time < 11.5)
interval_1hz = (adjusted_time > 11.7) & (adjusted_time < 16.7)
interval_2hz = (adjusted_time > 16.8) & (adjusted_time < 19.3)
interval_3hz = (adjusted_time > 19.45) & (adjusted_time < 20.785)
interval_5hz = (adjusted_time > 21.075) & (adjusted_time < 21.875)

# define new estimated displacement, actual displacement and time arrays for each interval
est_displ_05hz= est_displ[interval_05hz]
est_displ_1hz = est_displ[interval_1hz]
est_displ_2hz = est_displ[interval_2hz]
est_displ_3hz = est_displ[interval_3hz]
est_displ_5hz = est_displ[interval_5hz]
adj_displ_05hz = adj_displacement[interval_05hz]
adj_displ_1hz = adj_displacement[interval_1hz]
adj_displ_2hz = adj_displacement[interval_2hz]
adj_displ_3hz = adj_displacement[interval_3hz]
adj_displ_5hz = adj_displacement[interval_5hz]
adj_time_05hz= adjusted_time[interval_05hz]
adj_time_1hz = adjusted_time[interval_1hz]
adj_time_2hz = adjusted_time[interval_2hz]
adj_time_3hz = adjusted_time[interval_3hz]
adj_time_5hz = adjusted_time[interval_5hz]

# make lists out of this arrays
adj_displ_list = [adj_displ_05hz, adj_displ_1hz, adj_displ_2hz, adj_displ_3hz, adj_displ_5hz]
est_displ_list = [est_displ_05hz, est_displ_1hz, est_displ_2hz, est_displ_3hz, est_displ_5hz]
adj_time_list = [adj_time_05hz, adj_time_1hz, adj_time_2hz, adj_time_3hz, adj_time_5hz]

# We got the nrmse and phaselag values from a MATLAB script which used a fit also including the step signal
nrmse_list_MATLAB = [0.030351, 0.026979, 0.022837, 0.026343, 0.0533]
nrmse_list_MATLAB_impedance = [0.032851, 0.027403, 0.024022,  0.027462, 0.049285]
phaselag_list_MATLAB = [-3.6387, -2.7955, -1.2233, 0.30736, 2.5207]
phaselag_list_MATLAB_impedance = [-3.6224, -2.822, -1.1574, 0.40644, 2.1635]
frequencies = [0.5, 1, 2, 3, 5]

plt.rc('text', usetex=True)
plt.rc('font', family='serif', weight='bold')
plt.rcParams['axes.labelsize'] = 25   
plt.rcParams['legend.fontsize'] = 16 
plt.rcParams['xtick.labelsize'] = 20 
plt.rcParams['ytick.labelsize'] = 20 
plt.rcParams['grid.linewidth'] = 1.5
plt.rcParams['axes.linewidth'] = 1.5

p2_color = 'c'
p1_color = '#1f77b4'
line_width = 2
fig, axs = plt.subplots(1, 2, figsize=(16, 9)) 

axs[0].plot(frequencies, nrmse_list_MATLAB, color=p1_color, linewidth=line_width, label='NRMSE')
axs[0].plot(frequencies, nrmse_list_MATLAB, color=p1_color, linestyle='None', marker='x', markersize=8, markeredgewidth=3, zorder=5)
axs[0].plot(frequencies, nrmse_list_MATLAB_impedance, color=p2_color, linewidth=line_width, label='NRMSE')
axs[0].plot(frequencies, nrmse_list_MATLAB_impedance, color=p2_color, linestyle='None', marker='x', markersize=8, markeredgewidth=3, zorder=5)
axs[0].grid(True, which='both')  
axs[0].set_xlabel(r'Frequency (Hz)', weight='bold')  
axs[0].set_ylabel(r'NRMSE')
axs[0].set_ylim(0, 0.06) 
axs[0].set_xticks(frequencies)

axs[1].plot(frequencies, phaselag_list_MATLAB, color=p1_color, linewidth=line_width, label='Phase Lag (Degrees)')
axs[1].plot(frequencies, phaselag_list_MATLAB, color=p1_color, linestyle='None', marker='x', markersize=8, markeredgewidth=3, zorder=5)
axs[1].plot(frequencies, phaselag_list_MATLAB_impedance, color=p2_color, linewidth=line_width, label='Phase Lag (Degrees)')
axs[1].plot(frequencies, phaselag_list_MATLAB_impedance, color=p2_color, linestyle='None', marker='x', markersize=8, markeredgewidth=3, zorder=5)
axs[1].set_ylim(-4, 4)  
axs[1].grid(True, which='both')
axs[1].set_xlabel(r'Frequency (Hz)', weight='bold')
axs[1].set_ylabel(r'Phase Lag (Deg)')
axs[1].set_xticks(frequencies)

import matplotlib.ticker as ticker
def custom_formatter(x, pos):
        if x == 0.5:
            return f'{x:.1f}'  
        else:
            return f'{x:.0f}' 
axs[0].xaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))
axs[1].xaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))

legend_elements = [
    Line2D([0], [0], color=p1_color, lw=2, label='voltage-method'),
    Line2D([0], [0], color=p2_color, lw=2, label='impedance-method'),
]

fig.legend(handles=legend_elements, loc='upper center', handlelength=2,ncol=7, bbox_to_anchor=(0.5, 1.01), fontsize=22)

fig.subplots_adjust(
    top=0.925,
    bottom=0.575,
    left=0.25,
    right=0.76,
    hspace=0.36,
    wspace=0.32
)

plt.savefig('Final-Plot-7.pdf')
plt.show()