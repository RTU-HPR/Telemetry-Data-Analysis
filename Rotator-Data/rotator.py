import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np

file_name = "ROTATOR_DATA_LAUNCH.txt"
data_folder = "Rotator-Data/Data/"
map_folder = "Rotator-Data/Results/Maps/"
plot_folder = "Rotator-Data/Results/Plots/"

rssi_list = []
snr_list = []
dist_list = []

with open(f"{data_folder}{file_name}") as f:
  for line in f.readlines():
    if "RADIO COMMAND" in line and "MSG: rtu" in line:
      parts = line.split("|")
      rssi = parts[1][parts[1].find(":")+2:-1]
      snr = parts[2][parts[2].find(":")+2:parts[2].find("F")-1]
      rssi = int(float(rssi))
      snr = float(snr)
      rssi_list.append(rssi)
      snr_list.append(snr)
    if "CALC Straight line to ballon:" in line:
      distance = int(float(line.split(":")[1][1:]))
      dist_list.append(distance)
      
rssi_list = rssi_list[3:]
snr_list = snr_list[3:]

rssi_snr_plot_name = "rssi_snr.png"

# Plot for RSSI
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(dist_list, rssi_list, label='Received packet', color='red')

# Best-fit line for RSSI
slope_rssi, intercept_rssi, _, _, _ = linregress(dist_list, rssi_list)
fit_line_rssi = slope_rssi * np.array(dist_list) + intercept_rssi
plt.plot(dist_list, fit_line_rssi, label=f'RSSI Fit Line: y = {slope_rssi:.2f}x + {intercept_rssi:.2f}', linestyle='--', color='red')

plt.xlabel('Distance (meters)')
plt.ylabel('RSSI')
plt.legend()
plt.grid()

# Plot for SNR
plt.subplot(1, 2, 2)
plt.scatter(dist_list, snr_list, label='Received packet', color='green')

# Best-fit line for SNR
slope_snr, intercept_snr, _, _, _ = linregress(dist_list, snr_list)
fit_line_snr = slope_snr * np.array(dist_list) + intercept_snr
plt.plot(dist_list, fit_line_snr, label=f'SNR Fit Line: y = {slope_snr:.2f}x + {intercept_snr:.2f}', linestyle='--', color='green')

plt.xlabel('Distance (meters)')
plt.ylabel('SNR')
plt.legend()
plt.grid()

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()
plt.savefig(f"{plot_folder}{rssi_snr_plot_name}", dpi=500)
