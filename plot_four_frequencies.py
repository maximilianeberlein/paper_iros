import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from matplotlib.lines import Line2D
from scipy import signal
import matplotlib.ticker as ticker

rms_window_size = 200 # Define the window size for calculating RMS values
degree = 3 # Polynomial fitting degree

file_paths = [
    'MLogs/10_Sine_34gg_2kHz_TestPCB_26-July-2023_16-52-48/data.mat',
    'MLogs/10_Sine_34gg_2kHz_TestPCB_26-July-2023_16-51-12/data.mat',
]

plt.rc('text', usetex=True)
plt.rc('font', family='serif', weight='bold')
plt.rcParams['axes.labelsize'] = 25  
plt.rcParams['legend.fontsize'] = 16  
plt.rcParams['xtick.labelsize'] = 20 
plt.rcParams['ytick.labelsize'] = 20 
plt.rcParams['grid.linewidth'] = 1.5
plt.rcParams['axes.linewidth'] = 1.5

line_width = 2.5
fig, axs = plt.subplots(2, 1, figsize=(16, 9)) 

# iterate over both data.mat files and evaluate/plot them
for j, file_path in enumerate(file_paths):
        
    with h5py.File(file_path, 'r') as file:
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
    ss_current = df.iloc[4, :]

    # Factor for laser displacement and reference
    displacement = displacement * 4 + 30
    displacement = displacement - np.mean(displacement.iloc[:50])

    num_samples = len(time)
    num_rms = num_samples // rms_window_size
    adjusted_time = np.zeros(num_rms)
    adj_displacement = np.zeros(num_rms)
    rms_voltage_values = np.zeros(num_rms)
    rms_current_values = np.zeros(num_rms)


    # Calculate RMS adjusted voltage, time and displacement values where each data point is an average of rms_window_size data points from the raw data
    for i in range(num_rms):
        start_index = i * rms_window_size
        end_index = (i + 1) * rms_window_size - 1 
        rms_voltage_values[i] = np.sqrt(np.mean(ss_voltage.iloc[start_index:end_index] ** 2))
        # rms_current_values[i] = np.sqrt(np.mean(ss_current.iloc[start_index:end_index] ** 2)) # this line should be added for impedance-method
        adjusted_time[i] = np.mean(time.iloc[start_index:end_index])
        adj_displacement[i] = np.mean(displacement.iloc[start_index:end_index])

    # convert rms_voltage_values to pandas Series for polyfit
    rms_voltage_values = pd.Series(rms_voltage_values, dtype=float)

    # get estimated displacement via polynomial fitting
    coefficients = np.polyfit(rms_voltage_values, adj_displacement, degree)
    est_displ = np.polyval(coefficients, rms_voltage_values)

    # offset in y direction
    offset = np.min(adj_displacement)
    adj_displacement -= offset
    est_displ -= offset

    # calculate the RMSE and NRMSE values
    rmse = np.sqrt(np.mean((adj_displacement - est_displ) ** 2))
    range_gt_displ = np.max(adj_displacement) - np.min(adj_displacement)
    nrmse = rmse / range_gt_displ
    print("Root Mean Square Error (RMSE):", rmse)
    print("Normalized Root Mean Square Error (NRMSE):", nrmse)

    axs[j].plot(adjusted_time, adj_displacement, linewidth=line_width, color='r')
    axs[j].plot(adjusted_time, est_displ, linewidth=line_width, color='#1f77b4')

    if j==1:
        axs[1].set_xlabel(r'Time (s)', weight='bold') 
        axs[1].set_ylabel(r'Displacement (mm)') 
        axs[1].grid(True)
    if j==0:
        axs[0].set_xlabel(r'Time (s)', weight='bold') 
        axs[0].set_ylabel(r'Displacement (mm)')  
        axs[0].grid(True)  


legend_elements = [
    Line2D([0], [0], color='#1f77b4', lw=2, label='Estimated Displacement (voltage-method)'),
    Line2D([0], [0], color='r', lw=2, label='Ground Truth Displacement'),
]

fig.legend(handles=legend_elements, loc='upper center', handlelength=2,ncol=7, bbox_to_anchor=(0.5, 1.01), fontsize=18)

fig.subplots_adjust(
    top=0.930,
    bottom=0.225,
    left=0.27,
    right=0.78,
    hspace=0.3,
    wspace=0.105
)

plt.savefig('Final-Plot-8.pdf')
plt.show()