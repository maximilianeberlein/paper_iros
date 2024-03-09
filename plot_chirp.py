import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from matplotlib.lines import Line2D
from scipy.signal import correlate
from scipy import signal
from scipy.optimize import curve_fit

def sine_func(x, A, phi, offset, frequency):
    return A * np.sin(2 * np.pi * frequency * x + phi) + offset

# Define the window size for calculating RMS values and moving average filter
rms_window_size = 200
moving_average_window_size = 1 # Adjust the window size for the moving average filter

# 10_Sine_34gg_2kHz_1M-1M_26-July-2023_12-18-37 für Fig 9 und 10

# 10_Vogt_34gg_2kHz_TestPCB_26-July-2023_15-36-24 Fig. 15
# 10_LFT_34gg_2kHz_UpsideDown_26-July-2023_21-17-32 Fig. 16

with h5py.File('MLogs/10_Vogt_34gg_2kHz_TestPCB_26-July-2023_15-36-24/data.mat', 'r') as file:
    # Extract the data
    data = {}
    for key, value in file.items():
        # Check if the value is 1-dimensional
        if len(value.shape) == 1:
            data[key] = value[:]
        else:
            # If not 1-dimensional, convert to a list of 1-dimensional arrays
            for i in range(value.shape[1]):
                data[f"{key}_{i+1}"] = value[:, i]

# Convert to DataFrame
df = pd.DataFrame(data)

# Extract the columns from the data
time = df.iloc[0, :]
cmd_voltage = df.iloc[1, :]
displacement = df.iloc[2, :]
ss_voltage = df.iloc[3, :]

# Factor for laser displacement and reference
displacement = displacement * 4 + 30
displacement = displacement - np.mean(displacement.iloc[:50])

# Calculate the number of samples and the number of RMS calculations
num_samples = len(time)
num_rms = num_samples // rms_window_size

# Initialize arrays for RMS values and adjusted time and error
adjusted_time = np.zeros(num_rms)
adj_displacement = np.zeros(num_rms)
rms_voltage_values = np.zeros(num_rms)

#impedance = np.zeros(num_rms)

#displacement = displacement.rolling(window=moving_average_window_size, min_periods=1).mean()

# Calculate RMS values and adjusted time
for i in range(num_rms):
    start_index = i * rms_window_size
    end_index = (i + 1) * rms_window_size - 1   

    # Calculate RMS values
    rms_voltage_values[i] = np.sqrt(np.mean(ss_voltage.iloc[start_index:end_index] ** 2))

    # Adjusted time is set to the middle time value of the window
    adjusted_time[i] = np.mean(time.iloc[start_index:end_index])
    adj_displacement[i] = np.mean(displacement.iloc[start_index:end_index])

degree = 3
rms_voltage_values = pd.Series(rms_voltage_values)

if rms_voltage_values.isna().any():
        print(f"NaN values found in window ALDER")

rms_voltage_values = rms_voltage_values.rolling(window=moving_average_window_size, min_periods=1).mean()

if rms_voltage_values.isna().any():
        print(f"NaN values found in window ")

adj_displacement_for_fit = adj_displacement[(adjusted_time >= 3.45) & (adjusted_time < 24)]
rms_voltage_values_for_fit = rms_voltage_values[(adjusted_time >= 3.45) & (adjusted_time < 24)]

adj_displacement = adj_displacement[(adjusted_time >= 1.6) & (adjusted_time < 24)]
rms_voltage_values = rms_voltage_values[(adjusted_time >= 1.6) & (adjusted_time < 24)]
adjusted_time = adjusted_time[(adjusted_time >= 1.6) & (adjusted_time < 24)]
adjusted_time -= 1.6

coefficients_1 = np.polyfit(rms_voltage_values_for_fit, adj_displacement_for_fit, degree)
print(coefficients_1)
### if VOGT File change coeffs:
#coefficients_1 = [63.78575, -236.9001, 278.6847, -101.7]
###
est_displ = np.polyval(coefficients_1, rms_voltage_values)
est_displ_for_fit = np.polyval(coefficients_1, rms_voltage_values_for_fit)


offset = np.min(adj_displacement)
adj_displacement -= offset
est_displ -= offset
adj_displacement_for_fit -= offset
est_displ_for_fit -= offset


interval_05hz = (adjusted_time > 1.5) & (adjusted_time < 11.5)
interval_1hz = (adjusted_time > 11.7) & (adjusted_time < 16.7)
interval_2hz = (adjusted_time > 16.8) & (adjusted_time < 19.3)
interval_3hz = (adjusted_time > 19.45) & (adjusted_time < 20.785)
interval_5hz = (adjusted_time > 21.075) & (adjusted_time < 21.875)


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

print(adj_displ_2hz[:100])

# rms_voltage_values_05hz = rms_voltage_values[(adjusted_time >= 1.82) & (adjusted_time < 11.6)]
# rms_voltage_values_1hz = rms_voltage_values[(adjusted_time >= 11.6) & (adjusted_time < 16.7)]
# rms_voltage_values_2hz = rms_voltage_values[(adjusted_time >= 16.7) & (adjusted_time < 19.3)]
# rms_voltage_values_3hz = rms_voltage_values[(adjusted_time >= 19.3) & (adjusted_time < 20.97)]
# rms_voltage_values_5hz = rms_voltage_values[(adjusted_time >= 20.97) & (adjusted_time < 22.0)]

adj_displ_list = [adj_displ_05hz, adj_displ_1hz, adj_displ_2hz, adj_displ_3hz, adj_displ_5hz]

# Lists of estimated displacements
est_displ_list = [est_displ_05hz, est_displ_1hz, est_displ_2hz, est_displ_3hz, est_displ_5hz]

adj_time_list = [adj_time_05hz, adj_time_1hz, adj_time_2hz, adj_time_3hz, adj_time_5hz]

# Initialize lists to store RMSE and NRMSE values
rmse_list = []
nrmse_list = []

frequencies = [0.5, 1, 2, 3, 5]
nrmse_list_MATLAB = [0.030351, 0.026979, 0.022837, 0.026343, 0.0533]
phaselag_list = [-3.6387, -2.7955, -1.2233, 0.30736, 2.5207]
phaselag_list_only_waves_for_fit = [-3.6549, -2.8168, -1.2638, 0.25153, 2.4641]


# Calculate RMSE and NRMSE for each pair of adjusted and estimated displacements
for i, (adj_displ, est_displl, adj_time) in enumerate(zip(adj_displ_list, est_displ_list, adj_time_list)):    
    rmse = np.sqrt(np.mean((adj_displ - est_displl) ** 2))
    range_gt_displ = np.max(adj_displ) - np.min(adj_displ)
    print("range",range_gt_displ)
    nrmse = rmse / range_gt_displ
    print("rmse",rmse)
    print("nrmse",nrmse)
    rmse_list.append(rmse)
    nrmse_list.append(nrmse)

    amplitude_guess = (np.max(adj_displ) - np.min(adj_displ)) / 2
    phi_guess = 0  # Initial phase guess
    offset_guess = np.mean(adj_displ)

    # Perform curve fitting
    # popt_adj, _ = curve_fit(sine_func, adj_time, adj_displ, p0=[amplitude_guess, phi_guess, offset_guess, frequencies[i]])
    # A_adj, phi_adj, offset_adj, _ = popt_adj
    # print(A_adj, phi_adj, offset_adj)

    # popt_est, _ = curve_fit(sine_func, adj_time, est_displl, p0=[amplitude_guess, phi_guess, offset_guess, frequencies[i]])
    # A_est, phi_est, offset_est, _ = popt_est
    # print(A_est, phi_est, offset_est)

    # phase_shift = phi_est - phi_adj
    # phase_lag_degrees = np.degrees(phase_shift)
    # print("Phase shift:", phase_lag_degrees)


    # adj_time -= adj_time[0]
    # adj_displ -= adj_displ.mean()
    # adj_displ /= adj_displ.std()
    # est_displl -= est_displl.mean()
    # est_displl /= est_displl.std()

#     adj_displ = np.roll(adj_displ, 10)
#     nsamples = len(adj_displ)

#     cross_corr = np.correlate(adj_displ, est_displl, mode='full')
#     dt = np.linspace(-adj_time[-1], adj_time[-1], (2*nsamples)-1)
#     t_shift = dt[cross_corr.argmax()]
#     print("t_shift",t_shift)

#     phase_lag_radians = 2 * np.pi * t_shift * frequencies[i]
#     phase_lag_degrees = np.degrees(phase_lag_radians)
#     print('phase_lag_degrees', phase_lag_degrees)
# # 

    # lags = signal.correlation_lags(adj_displ.size, est_displl.size, mode="full")
    # lag = lags[np.argmax(cross_corr)]
    # print("LAG", lag)
    # # Find the index of the maximum value in the cross-correlation array
    # nsamples = len(adj_displ)
    # max_corr_index = np.argmax(cross_corr)
    # a = max_corr_index - len(adj_displ) + 1
    # print(cross_corr[:50])
    # print("len", len(cross_corr))
    # print("corr_index", max_corr_index)
    # print("timee", a)

    # dt = np.arange(1-nsamples, nsamples)

    # delay_steps = dt[max_corr_index]
    # print("delay",delay_steps)

    # phase_lag_radians = 2 * np.pi * delay_steps * 0.0005 * frequencies[i]
    # phase_lag_degrees = np.degrees(phase_lag_radians)

   # phaselag_list.append(phase_lag_degrees)


print("RMSE list:", rmse_list)
print("NRMSE list:", nrmse_list)


A = 6 
plt.rc('text', usetex=True)
plt.rc('font', family='serif', weight='bold')

t2_color = 'y'
t1_color = '#2ca02c'
p2_color = 'c'
p1_color = '#1f77b4'

plt.rcParams['axes.labelsize'] = 25   # Set x and y labels fontsize
plt.rcParams['legend.fontsize'] = 8  # Set legend fontsize
plt.rcParams['xtick.labelsize'] = 20  # Set x tick labels fontsize
plt.rcParams['ytick.labelsize'] = 20  # Set y tick labels fontsize
plt.rcParams['grid.linewidth'] = 1.5
plt.rcParams['axes.linewidth'] = 1.5

line_width = 2.5


fig = plt.figure()
fig.set_size_inches(16, 9)
gs = fig.add_gridspec(2,2)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])

ax1.plot(adjusted_time, adj_displacement, linewidth=line_width, color='r')
ax1.plot(adjusted_time, est_displ, linewidth=line_width, color=p1_color)
ax1.set_xlabel(r'Time (s)', weight='bold')  # X-axis label with increased font size and bold
ax1.set_ylabel(r'Displacement (mm)')  # Y-axis label with increased font size and bold
ax1.grid(True)  # Add grid with dashed lines
ax1.set_xlim(0, 22.4)  # Setting lower limit to -0.1

nrmse_list_including = np.concatenate((nrmse_list, [0.0638,0.1467,0.2615]))
frequencies_including = np.concatenate((frequencies, [8, 10, 20]))
ax2.plot(frequencies, nrmse_list_MATLAB, color='green', linewidth=line_width, label='NRMSE')
ax2.plot(frequencies, nrmse_list_MATLAB, color='green', linestyle='None', marker='x', markersize=8, markeredgewidth=3, zorder=5)
ax2.grid(True, which='both')  # Set the grid lines to be aligned with both major ticks
ax2.set_xlabel(r'Frequency (Hz)', weight='bold')  # X-axis label with increased font size and bold
ax2.set_ylabel(r'NRMSE')
ax2.set_ylim(0, 0.06)  # Setting lower limit to -0.1
ax2.set_xticks(frequencies)
#ax2.set_yticks(nrmse_list)

#ax3.plot(frequencies, phaselag_list_only_waves_for_fit, color='darkcyan', linewidth=line_width, label='Phase Lag (Only Waves) (Degrees)')
#ax3.plot(frequencies, phaselag_list_only_waves_for_fit, color='darkcyan', linestyle='None', marker='x', markersize=8, markeredgewidth=3, zorder=5)
ax3.plot(frequencies, phaselag_list, color='darkcyan', linewidth=line_width, label='Phase Lag (Degrees)')
ax3.plot(frequencies, phaselag_list, color='darkcyan', linestyle='None', marker='x', markersize=8, markeredgewidth=3, zorder=5)
ax3.set_ylim(-4, 4)  # Setting lower limit to -0.1
ax3.grid(True, which='both')
ax3.set_xlabel(r'Frequency (Hz)', weight='bold')  # X-axis label with increased font size and bold
ax3.set_ylabel(r'Phase Lag (Deg)')
ax3.set_xticks(frequencies)
#ax3.set_yticks(phaselag_list)



legend_elements = [
    Line2D([0], [0], color=p1_color, lw=2, label='Estimated Displacement (voltage-method)'),
    Line2D([0], [0], color='r', lw=2, label='Ground Truth Displacement'),
]

fig.legend(handles=legend_elements, loc='upper center', handlelength=2,ncol=7, bbox_to_anchor=(0.5, 1.01), fontsize=18)

fig.subplots_adjust(
    top=0.925,
    bottom=0.225,
    left=0.075,
    right=0.98,
    hspace=0.36,
    wspace=0.27
)

plt.savefig('FINAL-FIG-3.pdf')

# Show the plot
plt.show()