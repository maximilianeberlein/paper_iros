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

plt.rc('text', usetex=True)
plt.rc('font', family='serif', weight='bold')
plt.rcParams['axes.labelsize'] = 25   # Set x and y labels fontsize
plt.rcParams['legend.fontsize'] = 8  # Set legend fontsize
plt.rcParams['xtick.labelsize'] = 20  # Set x tick labels fontsize
plt.rcParams['ytick.labelsize'] = 20  # Set y tick labels fontsize
plt.rcParams['grid.linewidth'] = 1.5
plt.rcParams['axes.linewidth'] = 1.5

line_width = 2.5
fig, axs = plt.subplots(1, 1, figsize=(16, 9)) 

axs.plot(adjusted_time, adj_displacement, linewidth=line_width, color='r')
axs.plot(adjusted_time, est_displ, linewidth=line_width, color='#1f77b4')
axs.set_xlabel(r'Time (s)', weight='bold') 
axs.set_ylabel(r'Displacement (mm)') 
axs.set_xlim(0, 22.4)
axs.grid(True) 

legend_elements = [
    Line2D([0], [0], color='#1f77b4', lw=2, label='Estimated Displacement (voltage-method)'),
    Line2D([0], [0], color='r', lw=2, label='Ground Truth Displacement'),
]

fig.legend(handles=legend_elements, loc='upper center', handlelength=2,ncol=7, bbox_to_anchor=(0.5, 1.01), fontsize=18)

fig.subplots_adjust(
    top=0.925,
    bottom=0.625,
    left=0.075,
    right=0.98,
    hspace=0.36,
    wspace=0.27
)

plt.savefig('Final-Plot-6.pdf')
plt.show()