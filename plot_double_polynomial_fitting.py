import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from matplotlib.lines import Line2D
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


# Define the window size for calculating RMS values and moving average filter
rms_window_size = 200
moving_average_window_size = 1  # Adjust the window size for the moving average filter

with h5py.File('MLogs/10_Sine_34gg_2kHz_TestPCB_26-July-2023_16-49-14/data.mat', 'r') as file:
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
time = df.iloc[0, 10000:-20000]
cmd_voltage = df.iloc[1, 10000:-20000]
displacement = df.iloc[2, 10000:-20000]
ss_voltage = df.iloc[3, 10000:-20000]

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

displacement = displacement.rolling(window=moving_average_window_size, min_periods=1).mean()

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


diff_rms_voltage = rms_voltage_values - rms_voltage_values.shift(2)
print(diff_rms_voltage)
#diff_rms_voltage = rms_voltage_values.diff()
diff_rms_voltage[0] = diff_rms_voltage[2]
diff_rms_voltage[1] = diff_rms_voltage[0]
#diff_rms_voltage[2] = diff_rms_voltage[1]
#diff_rms_voltage[3] = diff_rms_voltage[2]


positive_derivative = rms_voltage_values[diff_rms_voltage >= 0]
negative_derivative = rms_voltage_values[diff_rms_voltage < 0]

print(len(positive_derivative))
print(len(negative_derivative))

pos_rms_voltage_values = rms_voltage_values[positive_derivative.index]
neg_rms_voltage_values = rms_voltage_values[negative_derivative.index]
pos_adj_displacement = adj_displacement[positive_derivative.index]
neg_adj_displacement = adj_displacement[negative_derivative.index]
pos_adjusted_time = adjusted_time[positive_derivative.index]
neg_adjusted_time = adjusted_time[negative_derivative.index]

# X = pos_rms_voltage_values.values.reshape(-1, 1)
# y = pos_adj_displacement
# model = make_pipeline(PolynomialFeatures(degree=3), RANSACRegressor())
# model.fit(X, y)
# coefficients = model.named_steps['ransacregressor'].estimator_.coef_# Getting the coefficients of the polynomial
# X_fit = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
# y_fit = model.predict(X_fit)
# est_displ_pos_predict = model.predict(X)
# estimated_displacement_pos_predict = np.full_like(rms_voltage_values, np.nan)
# estimated_displacement_pos_predict[positive_derivative.index] = est_displ_pos_predict

coefficients = np.polyfit(rms_voltage_values, adj_displacement, degree)
est_displ = np.polyval(coefficients, rms_voltage_values)

coefficients_pos = np.polyfit(pos_rms_voltage_values, pos_adj_displacement, degree)
est_displ_pos = np.polyval(coefficients_pos, pos_rms_voltage_values)

coefficients_neg = np.polyfit(neg_rms_voltage_values, neg_adj_displacement, degree)
est_displ_neg = np.polyval(coefficients_neg, neg_rms_voltage_values)

estimated_displacement_pos = np.full_like(rms_voltage_values, np.nan)
estimated_displacement_pos[positive_derivative.index] = est_displ_pos
estimated_displacement_neg = np.full_like(rms_voltage_values, np.nan)
estimated_displacement_neg[negative_derivative.index] = est_displ_neg



A = 6 
plt.rc('text', usetex=True)
plt.rc('font', family='serif', weight='bold')

t2_color = 'y'
t1_color = '#2ca02c'
p2_color = 'c'
p1_color = '#1f77b4'

plt.rcParams['axes.labelsize'] = 25   # Set x and y labels fontsize
plt.rcParams['legend.fontsize'] = 16  # Set legend fontsize
plt.rcParams['xtick.labelsize'] = 20  # Set x tick labels fontsize
plt.rcParams['ytick.labelsize'] = 20  # Set y tick labels fontsize
plt.rcParams['grid.linewidth'] = 1.5
plt.rcParams['axes.linewidth'] = 1.5

line_width = 2.5

fig, axs = plt.subplots(2, 2, figsize=(16, 9))  # 1 row, 2 columns

sorted_indices = np.argsort(rms_voltage_values)
sorted_voltage_values = rms_voltage_values[sorted_indices]
sorted_displ = est_displ[sorted_indices]

pos_rms_voltage_values = pos_rms_voltage_values.values if isinstance(pos_rms_voltage_values, pd.Series) else pos_rms_voltage_values
sorted_indices_0 = np.argsort(pos_rms_voltage_values)
sorted_voltage_values_pos = pos_rms_voltage_values[sorted_indices_0]
sorted_displ_pos = est_displ_pos[sorted_indices_0]
#sorted_displ_pos_predict = est_displ_pos_predict[sorted_indices]

neg_rms_voltage_values = neg_rms_voltage_values.values if isinstance(neg_rms_voltage_values, pd.Series) else neg_rms_voltage_values
sorted_indices_1 = np.argsort(neg_rms_voltage_values)
sorted_voltage_values_neg = neg_rms_voltage_values[sorted_indices_1]
sorted_displ_neg = est_displ_neg[sorted_indices_1]

offset = np.min(adj_displacement)
adj_displacement -= offset
pos_adj_displacement -= offset
neg_adj_displacement -= offset
est_displ -= offset
estimated_displacement_neg -= offset
estimated_displacement_pos -= offset
sorted_displ -= offset
sorted_displ_pos -= offset
sorted_displ_neg -= offset



axs[0][0].plot(rms_voltage_values, adj_displacement, 'mediumturquoise', linestyle='None', marker='x', markersize=8, markeredgewidth=1, label='Measured Data Points')
axs[0][0].plot(sorted_voltage_values, sorted_displ, 'darkcyan', linewidth=line_width, label='Polynomial Fit: Coefficients: ')
axs[0][0].set_ylabel(r'Displacement (mm)')  # Y-axis label with increased font size and bold
axs[0][0].set_xlabel(r'Voltage (V)', weight='bold')  # X-axis label with increased font size and bold
axs[0][0].grid(True)  # Add grid with dashed lines
axs[0][0].set_ylim(-0.08, None)  # Setting lower limit to -0.1

axs[0][1].plot(pos_rms_voltage_values, pos_adj_displacement, color='mediumpurple', linestyle='None', marker='x', markersize=8, markeredgewidth=1, label='Measured Data Points')
axs[0][1].plot(neg_rms_voltage_values, neg_adj_displacement, 'springgreen', linestyle='None', marker='x', markersize=8, markeredgewidth=1, label='Measured Data Points')
axs[0][1].plot(sorted_voltage_values_pos, sorted_displ_pos, color='darkslateblue', linewidth=line_width, label='Polynomial Fit: Coefficients: ')
axs[0][1].plot(sorted_voltage_values_neg, sorted_displ_neg, 'limegreen', linewidth=line_width, label='Polynomial Fit: Coefficients: ')
axs[0][1].set_xlabel(r'Voltage (V)', weight='bold')  # X-axis label with increased font size and bold
axs[0][1].grid(True)  # Add grid with dashed lines
axs[0][1].set_ylim(-0.08, None)  # Setting lower limit to -0.1

estimated_displacement = np.where(diff_rms_voltage >= 0, estimated_displacement_pos, estimated_displacement_neg)
#estimated_displacement_predict = np.where(diff_rms_voltage >= 0.01, estimated_displacement_pos_predict, estimated_displacement_neg)

rmse = np.sqrt(np.mean((adj_displacement - est_displ) ** 2))
range_gt_displ = np.max(adj_displacement) - np.min(adj_displacement)
nrmse = rmse / range_gt_displ

rmse_doublefit = np.sqrt(np.mean((adj_displacement - estimated_displacement) ** 2))
range_gt_displ = np.max(adj_displacement) - np.min(adj_displacement)
nrmse_doublefit = rmse_doublefit / range_gt_displ

axs[1][0].plot(adjusted_time, adj_displacement, linewidth=line_width, color='r')
axs[1][0].plot(adjusted_time, est_displ, linewidth=line_width, color=p1_color)
axs[1][0].set_ylabel(r'Displacement (mm)')  # Y-axis label with increased font size and bold
axs[1][0].set_xlabel(r'Time (s)', weight='bold')  # X-axis label with increased font size and bold
axs[1][0].grid(True)  # Add grid with dashed lines
axs[1][0].set_ylim(-0.075, 0.48)  # Setting lower limit to -0.1
axs[1][0].set_title(f'NRMSE: {round(nrmse, 4)}',fontsize=25)

axs[1][1].plot(adjusted_time, adj_displacement, linewidth=line_width, color='r')
axs[1][1].plot(adjusted_time, estimated_displacement, linewidth=line_width, color=p1_color)
axs[1][1].set_xlabel(r'Time (s)', weight='bold')  # X-axis label with increased font size and bold
axs[1][1].grid(True)  # Add grid with dashed lines
axs[1][1].set_ylim(-0.075, 0.48)  # Setting lower limit to -0.1
axs[1][1].set_title(f'NRMSE: {round(nrmse_doublefit, 4)}',fontsize=25)

legend_elements = [
    Line2D([0], [0], color='darkcyan', lw=2, label='Cubic Polynomial Fit'),
    Line2D([0], [0], color='mediumturquoise', marker='x', linestyle='None', markersize=10, markeredgewidth=2, label='Measured Data Points'),
    Line2D([0], [0], color='limegreen', lw=2, label='Cub. Pol. Fit (neg. deriv.)'),
    Line2D([0], [0], color='springgreen', marker='x', linestyle='None', markersize=10, markeredgewidth=2, label='Data Points (neg. deriv.)'),
    Line2D([0], [0], color='darkslateblue', lw=2, label='Cub. Pol. Fit (pos. deriv.)'),
    Line2D([0], [0], color='mediumpurple', marker='x', linestyle='None', markersize=10, markeredgewidth=2, label='Data Points (pos. deriv.)'),
    Line2D([0], [0], color=p1_color, lw=2, label='Est. Displacement (voltage-method)'),
    Line2D([0], [0], color='r', lw=2, label='Ground Truth Displacement')
    
]


fig.legend(handles=legend_elements, loc='upper center', handlelength=2,ncol=4, bbox_to_anchor=(0.5, 1.01), fontsize=18)

fig.subplots_adjust(
    top=0.89,
    bottom=0.18,
    left=0.075,
    right=0.98,
    hspace=0.46,
    wspace=0.105
)

plt.savefig('FINAL-FIG-5.pdf')

# Show the plot
plt.show()