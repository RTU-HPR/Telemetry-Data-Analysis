import pandas as pd
import matplotlib.pyplot as plt
import os
import folium
import numpy as np
import datetime

pd.options.mode.chained_assignment = None

file_name = "PAYLOAD_FC_TELEMETRY.csv"
data_folder = "Payload-Data/Data/"
map_folder = "Payload-Data/Results/Maps/"
plot_folder = "Payload-Data/Results/Plots/"

# Just for reference
header = [
    "gps_epoch_time",
    "gps_lat",
    "gps_lng",
    "gps_height",
    "gps_speed",
    "gps_time_since_last",
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "r1_dist",
    "r2_dist",
    "r3_dist",
    "r1_time_since",
    "r2_time_since",
    "r3_time_since",
    "r_pos_lat",
    "r_pos_lng",
    "r_pos_alt",
    "r_pos_time_since",
    "inner_baro_pressure",
    "outer_baro_pressure",
    "avg_inner_temp",
    "avg_outer_temp",
    "heater_power",
    "time_on",
    "avg_battery_voltage",
    "outer_baro_altitude",
    "outer_baro_altitude_speed",
    "gps_heading",
    "gps_pdop",
    "gps_satellites",
    "raw_inner_temp",
    "raw_outer_temp_thermistor",
    "raw_inner_temp_baro",
    "raw_outer_temp_baro",
    "raw_batt_voltage",
    "raw_heater_current",
    "avg_heater_current",
    "p_term",
    "i_term",
    "d_term",
    "target_temp",
    "r1_time",
    "r1_rssi",
    "r1_snr",
    "r1_f_error",
    "r2_time",
    "r2_rssi",
    "r2_snr",
    "r2_f_error",
    "r3_time",
    "r3_rssi",
    "r3_snr",
    "r3_f_error",
    "last_frequency",
]

# Read data
df = None
if os.path.exists(f"{data_folder}{file_name}"):
  # Clean up the original file
  df = pd.read_csv(f"{data_folder}{file_name}")

# STATISTICS
print("STATISTICS")

# Create a new column with time on converted to seconds
df["time_on_seconds"] = df["time_on"] / 1000

# Time of flight
time_on = df["time_on"].iloc[-1] / 1000  # ms to s
start_time = df["gps_epoch_time"].iloc[0]
end_time = df["gps_epoch_time"].iloc[-1]

launch_time = df[(df['gps_speed'] > 3) & (df['gps_height'] > 150)]['gps_epoch_time'].iloc[0]
landing_time = df[(df['gps_speed'] > 3) & (df['gps_height'] > 150)]['gps_epoch_time'].iloc[-1]
launch_time_millis = df[(df['gps_speed'] > 3) & (df['gps_height'] > 150)]['time_on_seconds'].iloc[0]
landing_time_millis = df[(df['gps_speed'] > 3) & (df['gps_height'] > 150)]['time_on_seconds'].iloc[-1]
time_of_flight = landing_time - launch_time

max_height_index = df['gps_height'].idxmax()
max_height_epoch = df.loc[max_height_index, 'gps_epoch_time']
max_height_time_on = df.loc[max_height_index, 'time_on_seconds']

print("Time of flight")
print(f"Turned ON: {str(datetime.datetime.fromtimestamp(start_time))}")
print(f"Turned OFF: {str(datetime.datetime.fromtimestamp(end_time))}")
print(f"Total time ON: {time_on:.2f} seconds | {int(time_on//60)} minutes and {time_on%60:.0f} seconds | {str(datetime.timedelta(seconds=round(time_on)))} hours")
print(f"Launch time: {str(datetime.datetime.fromtimestamp(launch_time))} or {launch_time_millis:.0f} seconds after turning on")
print(f"Landing time: {str(datetime.datetime.fromtimestamp(landing_time))} or {landing_time_millis:.0f} seconds after turning on")
print(f"Time of flight: {str(datetime.timedelta(seconds=round(time_of_flight)))} hours")
print(f"Top of ascent time: {str(datetime.datetime.fromtimestamp(max_height_epoch))} or {max_height_time_on:.0f} seconds after turning on")
print()

# Time between packets
print("Time between packets")
df['time_diff'] = df['time_on'].diff()

average_time_diff = df['time_diff'].mean()
max_time_diff = df['time_diff'].max()
min_time_diff = df['time_diff'].min()
median_time_diff = df['time_diff'].median()

threshold = 4 * average_time_diff
extreme_values = df[df['time_diff'] > threshold]
num_extreme_values = len(extreme_values)
extreme_info = extreme_values.copy()
extreme_info['row_index'] = extreme_info.index
extreme_info = extreme_info[['row_index', 'time_diff', 'time_on']]
median_time_diff_extreme = extreme_values['time_diff'].median()

print(f"Average time between packets: {average_time_diff:.2f} milliseconds")
print(f"Max time between packets: {max_time_diff:.2f} milliseconds")
print(f"Min time between packets: {min_time_diff:.2f} milliseconds")
print(f"Median time between packets: {median_time_diff:.2f} milliseconds")
print("Extreme values")
print(f"Number of Extreme Values: {num_extreme_values}")
print(extreme_info.to_string(index=False, formatters={'time_diff': '{:.0f}'.format}))
print(f"Median Time Difference of Extreme Values: {median_time_diff_extreme} milliseconds")
print()

# Make separate data frames for each stage of flight
before_df = df[df['gps_epoch_time'] < launch_time]
flight_df = df[(df['gps_epoch_time'] > launch_time) & (df['gps_epoch_time'] < landing_time)]
after_df = df[(df['gps_epoch_time'] > landing_time)]

# Create a new column that contains the time in seconds since the balloon was launched
flight_df['flight_time'] = flight_df['time_on'] - flight_df['time_on'].iloc[0]
flight_df['flight_time'] /= 1000

# Useful statistics
# Altitude
print("Altitude")
max_gps_altitude = df["gps_height"].max()
max_baro_altitude = df["outer_baro_altitude"].max()
print(f"Max GPS altitude: {max_gps_altitude:.2f} meters | {max_gps_altitude/1000:.2f} kilometers")
print(f"Max barometer altitude: {max_baro_altitude:.2f} meters | {max_baro_altitude/1000:.2f} kilometers")
print()

# Speed
print("Speed")
# 0 to 5 km
speed_df = flight_df.loc[:flight_df[flight_df['gps_height'] >= 5000].index.min() - 1]
distance = speed_df["gps_height"].iloc[-1] - speed_df["gps_height"].iloc[0]
time = speed_df["time_on_seconds"].iloc[-1] - speed_df["time_on_seconds"].iloc[0]
speed = distance/time
print(f"GPS average speed from {speed_df["gps_height"].iloc[0]:.0f} meters to {speed_df["gps_height"].iloc[-1]:.0f} meters is {speed:.2f} m/s")

# 5 to 10 km
speed_df = flight_df.loc[flight_df[flight_df['gps_height'] >= 5000].index.min() - 1:flight_df[flight_df['gps_height'] >= 10000].index.min() - 1]
distance = speed_df["gps_height"].iloc[-1] - speed_df["gps_height"].iloc[0]
time = speed_df["time_on_seconds"].iloc[-1] - speed_df["time_on_seconds"].iloc[0]
speed = distance/time
print(f"GPS average speed from {speed_df["gps_height"].iloc[0]:.0f} meters to {speed_df["gps_height"].iloc[-1]:.0f} meters is {speed:.2f} m/s")

# 10 to 15 km
speed_df = flight_df.loc[flight_df[flight_df['gps_height'] >= 10000].index.min() - 1:flight_df[flight_df['gps_height'] >= 15000].index.min() - 1]
distance = speed_df["gps_height"].iloc[-1] - speed_df["gps_height"].iloc[0]
time = speed_df["time_on_seconds"].iloc[-1] - speed_df["time_on_seconds"].iloc[0]
speed = distance/time
print(f"GPS average speed from {speed_df["gps_height"].iloc[0]:.0f} meters to {speed_df["gps_height"].iloc[-1]:.0f} meters is {speed:.2f} m/s")

# 15 to 23 km
speed_df = flight_df.loc[flight_df[flight_df['gps_height'] >= 15000].index.min() - 1:flight_df['gps_height'].idxmax()-1]
distance = speed_df["gps_height"].iloc[-1] - speed_df["gps_height"].iloc[0]
time = speed_df["time_on_seconds"].iloc[-1] - speed_df["time_on_seconds"].iloc[0]
speed = distance/time
print(f"GPS average speed from {speed_df["gps_height"].iloc[0]:.0f} meters to {speed_df["gps_height"].iloc[-1]:.0f} meters is {speed:.2f} m/s")

# 23 to 15 km
speed_df = flight_df.loc[flight_df['gps_height'].idxmax()-1:flight_df[flight_df['gps_height'] >= 15000].index.max() - 1]
distance = speed_df["gps_height"].iloc[-1] - speed_df["gps_height"].iloc[0]
time = speed_df["time_on_seconds"].iloc[-1] - speed_df["time_on_seconds"].iloc[0]
speed = distance/time
print(f"GPS average speed from {speed_df["gps_height"].iloc[0]:.0f} meters to {speed_df["gps_height"].iloc[-1]:.0f} meters is {speed:.2f} m/s")

# 15 to 10 km
speed_df = flight_df.loc[flight_df[flight_df['gps_height'] >= 15000].index.max() - 1:flight_df[flight_df['gps_height'] >= 10000].index.max() - 1]
distance = speed_df["gps_height"].iloc[-1] - speed_df["gps_height"].iloc[0]
time = speed_df["time_on_seconds"].iloc[-1] - speed_df["time_on_seconds"].iloc[0]
speed = distance/time
print(f"GPS average speed from {speed_df["gps_height"].iloc[0]:.0f} meters to {speed_df["gps_height"].iloc[-1]:.0f} meters is {speed:.2f} m/s")

# 10 to 5 km
speed_df = flight_df.loc[flight_df[flight_df['gps_height'] >= 10000].index.max() - 1:flight_df[flight_df['gps_height'] >= 5000].index.max() - 1]
distance = speed_df["gps_height"].iloc[-1] - speed_df["gps_height"].iloc[0]
time = speed_df["time_on_seconds"].iloc[-1] - speed_df["time_on_seconds"].iloc[0]
speed = distance/time
print(f"GPS average speed from {speed_df["gps_height"].iloc[0]:.0f} meters to {speed_df["gps_height"].iloc[-1]:.0f} meters is {speed:.2f} m/s")

# 5 to 1 km
speed_df = flight_df.loc[flight_df[flight_df['gps_height'] >= 5000].index.max() - 1:flight_df[flight_df['gps_height'] >= 1000].index.max() - 1]
distance = speed_df["gps_height"].iloc[-1] - speed_df["gps_height"].iloc[0]
time = speed_df["time_on_seconds"].iloc[-1] - speed_df["time_on_seconds"].iloc[0]
speed = distance/time
print(f"GPS average speed from {speed_df["gps_height"].iloc[0]:.0f} meters to {speed_df["gps_height"].iloc[-1]:.0f} meters is {speed:.2f} m/s")

# 1 to 300 m
speed_df = flight_df.loc[flight_df[flight_df['gps_height'] >= 1000].index.max() - 1:flight_df[flight_df['gps_height'] >= 300].index.max() - 1]
distance = speed_df["gps_height"].iloc[-1] - speed_df["gps_height"].iloc[0]
time = speed_df["time_on_seconds"].iloc[-1] - speed_df["time_on_seconds"].iloc[0]
speed = distance/time
print(f"GPS average speed from {speed_df["gps_height"].iloc[0]:.0f} meters to {speed_df["gps_height"].iloc[-1]:.0f} meters is {speed:.2f} m/s")

# 300 m to 0 m
speed_df = flight_df.loc[flight_df[flight_df['gps_height'] >= 300].index.max() - 1:]
distance = speed_df["gps_height"].iloc[-1] - speed_df["gps_height"].iloc[0]
time = speed_df["time_on_seconds"].iloc[-1] - speed_df["time_on_seconds"].iloc[0]
speed = distance/time
print(f"GPS average speed from {speed_df["gps_height"].iloc[0]:.0f} meters to {speed_df["gps_height"].iloc[-1]:.0f} meters is {speed:.2f} m/s")

# GPS
map_name = "trajectory.html"
if not os.path.exists(f"{map_folder}{map_name}"):
  m = folium.Map([df["gps_lat"].iloc[0], df["gps_lng"].iloc[0]], zoom_start=11)
  coordinates = []
  for i in range(len(df["gps_lat"])):
    coordinates.append([df["gps_lat"].iloc[i], df["gps_lng"].iloc[i]])
  
  folium.PolyLine(coordinates).add_to(m)
  m.save(f"{map_folder}{map_name}")
  print("Trajectory plotted")
else:
  print("Trajectory already plotted")
print()

# PLOTS
print("PLOTS")

# Time between packets
time_between_packets_plot_name = "time_between_packets_plot.png"
if not os.path.exists(f"{plot_folder}{time_between_packets_plot_name}"):
  fig, ax = plt.subplots()
  plt.gca().xaxis.set_major_locator(plt.MaxNLocator(12))
  df.plot(x="time_on_seconds", y="time_diff", kind="scatter", ax=ax, label="Time between packets")
  plt.xlabel("Time on (seconds)")
  plt.ylabel("Time between packets (milliseconds)")
  plt.legend()
  plt.grid()
  plt.savefig(f"{plot_folder}{time_between_packets_plot_name}", dpi=500)
  print("Time between packets plot created")
else:
  print("Time between packets plot already created")
  
# Time between extreme packets
time_between_packets_extreme_plot_name = "time_between_packets_extreme_plot.png"
if not os.path.exists(f"{plot_folder}{time_between_packets_extreme_plot_name}"):
  fig, ax = plt.subplots()
  plt.gca().xaxis.set_major_locator(plt.MaxNLocator(12))
  extreme_values.plot(x="time_on_seconds", y="time_diff", kind="scatter", ax=ax, label="Time between packets")
  plt.xlabel("Time on (seconds)")
  plt.ylabel("Time between packets (milliseconds)")
  plt.legend()
  plt.grid()
  plt.savefig(f"{plot_folder}{time_between_packets_extreme_plot_name}", dpi=500)
  print("Time between extreme packets plot created")
else:
  print("Time between extreme packets plot already created")

# GPS altitude plot
gps_alt_plot_name = "gps_altitude.png"
if not os.path.exists(f"{plot_folder}{gps_alt_plot_name}"):
  fig, ax = plt.subplots()
  plt.gca().xaxis.set_major_locator(plt.MaxNLocator(12))
  flight_df.plot(x="time_on_seconds", y="gps_height", kind="line", ax=ax, label="Altitude")
  plt.xlabel("Time on (seconds)")
  plt.ylabel("Altitude (m)")
  plt.legend()
  plt.grid()
  plt.savefig(f"{plot_folder}{gps_alt_plot_name}", dpi=500)
  print("GPS Altitude plot created")
else:
  print("GPS Altitude plot already created")
  
# Temperature plots
temp_plot_name = "all_temperature_plot.png"
if not os.path.exists(f"{plot_folder}{temp_plot_name}"):
  fig, ax = plt.subplots()
  plt.gca().xaxis.set_major_locator(plt.MaxNLocator(12))
  flight_df.plot(x="time_on_seconds", y="avg_inner_temp", kind="line", ax=ax, label="Inner temperature")
  flight_df.plot(x="time_on_seconds", y="avg_outer_temp", kind="line", ax=ax, label="Outer temperature")
  flight_df.plot(x="time_on_seconds", y="raw_inner_temp_baro", kind="line", ax=ax, label="Inner barometer temperature")
  plt.xlabel("Time on (seconds)")
  plt.ylabel("Temperature (°C)")
  plt.legend()
  plt.grid()
  plt.savefig(f"{plot_folder}{temp_plot_name}", dpi=500)
  print("All temperature plot created")
else:
  print("All temperature plot already created")

# Baro temp vs inner temp
baro_vs_probe_temp_plot_name = "baro_vs_probe_temperature_plot.png"
if not os.path.exists(f"{plot_folder}{baro_vs_probe_temp_plot_name}"):
  fig, ax = plt.subplots()
  plt.gca().xaxis.set_major_locator(plt.MaxNLocator(12))
  flight_df.plot(x="time_on_seconds", y="avg_inner_temp", kind="line", ax=ax, label="Inner temperature")
  flight_df.plot(x="time_on_seconds", y="raw_inner_temp_baro", kind="line", ax=ax,label="Barometer temperature")
  flight_df.plot(x="time_on_seconds", y="target_temp", kind="line", ax=ax,label="Target temperature")
  plt.xlabel("Time on (seconds)")
  plt.ylabel("Temperature (°C)")
  plt.legend()
  plt.grid()
  plt.savefig(f"{plot_folder}{baro_vs_probe_temp_plot_name}", dpi=500)
  print("Baro vs Inner temperature probe temperature plot created")
else:
  print("Baro vs Inner temperature probe temperature plot already created")

# Baro temp vs inner temp full duration
baro_vs_probe_temp_plot_full_name = "baro_vs_probe_temperature_plot_full.png"
if not os.path.exists(f"{plot_folder}{baro_vs_probe_temp_plot_full_name}"):
  fig, ax = plt.subplots()
  plt.gca().xaxis.set_major_locator(plt.MaxNLocator(12))
  df.plot(x="time_on_seconds", y="avg_inner_temp", kind="line", ax=ax, label="Inner temperature")
  df.plot(x="time_on_seconds", y="raw_inner_temp_baro", kind="line", ax=ax,label="Barometer temperature")
  df.plot(x="time_on_seconds", y="target_temp", kind="line", ax=ax,label="Target temperature")
  plt.xlabel("Time on (seconds)")
  plt.ylabel("Temperature (°C)")
  plt.legend()
  plt.grid()
  plt.savefig(f"{plot_folder}{baro_vs_probe_temp_plot_full_name}", dpi=500)
  print("Baro vs Inner temperature probe full duration temperature plot created")
else:
  print("Baro vs Inner temperature probe full duration temperature plot already created")

# Temperature vs altitude duration
temp_vs_altitude_plot_name = "temp_vs_altitude_plot.png"
if not os.path.exists(f"{plot_folder}{temp_vs_altitude_plot_name}"):
  # Create graph
  fig, ax1 = plt.subplots()
  # Set labels
  ax1.set_xlabel("Time on [s]")
  ax1.set_ylabel("Altitude [m]")
  # Plot altitude data
  ax1.plot(flight_df["time_on_seconds"], flight_df["gps_height"], color="b", label="Altitude")
  ax1.tick_params(axis="y")
  # Add other y-axis
  ax2 = ax1.twinx()
  # Set labels
  ax2.set_ylabel("Temperature [°C]")
  # Plot temperature data
  ax2.plot(flight_df["time_on_seconds"], flight_df["avg_inner_temp"], color="r", label="Inner temperature")
  ax2.plot(flight_df["time_on_seconds"], flight_df["avg_outer_temp"], color="orange", label="Outer temperature")
  ax2.tick_params(axis="y")
  # Show plot
  ax1.grid(True)
  ax1.legend(loc="center right")
  ax2.legend()
  plt.savefig(f"{plot_folder}{temp_vs_altitude_plot_name}", dpi=500)
  print("Temperature vs altitude plot created")
else:
  print("Temperature vs altitude plot already created")

# Pressure
all_barometer_plot_name = "all_barometer_plot.png"
if not os.path.exists(f"{plot_folder}{all_barometer_plot_name}"):
  fig, ax = plt.subplots()
  plt.gca().xaxis.set_major_locator(plt.MaxNLocator(12))
  flight_df.plot(x="time_on_seconds", y="inner_baro_pressure", kind="line", ax=ax, label="Inner pressure")
  flight_df.plot(x="time_on_seconds", y="outer_baro_pressure", kind="line", ax=ax, label="Outer pressure")
  plt.xlabel("Time on (seconds)")
  plt.ylabel("Pressure (Pa)")
  plt.legend()
  plt.grid()
  plt.savefig(f"{plot_folder}{all_barometer_plot_name}", dpi=500)
  print("All Barometer plot created")
else:
  print("All Barometer plot already created")

# PID
pid_plot_name = "pid_plot.png"
if not os.path.exists(f"{plot_folder}{pid_plot_name}"):
  fig, ax = plt.subplots()
  plt.gca().xaxis.set_major_locator(plt.MaxNLocator(12))
  flight_df.plot(x="time_on_seconds", y="p_term", kind="line", ax=ax, label="P")
  flight_df.plot(x="time_on_seconds", y="i_term", kind="line", ax=ax, label="I")
  flight_df.plot(x="time_on_seconds", y="d_term", kind="line", ax=ax, label="D")
  flight_df.plot(x="time_on_seconds", y="heater_power", kind="line", ax=ax, label="Heater power")
  plt.ylim(bottom=-200)
  plt.xlabel("Time on (seconds)")
  plt.ylabel("Values")
  plt.legend()
  plt.grid()
  plt.savefig(f"{plot_folder}{pid_plot_name}", dpi=500)
  print("PID plot created")
else:
  print("PID plot already created") 

# PID
pid_plot_full_name = "pid_plot.png"
if not os.path.exists(f"{plot_folder}{pid_plot_full_name}"):
  fig, ax = plt.subplots()
  plt.gca().xaxis.set_major_locator(plt.MaxNLocator(12))
  df.plot(x="time_on_seconds", y="p_term", kind="line", ax=ax, label="P")
  df.plot(x="time_on_seconds", y="i_term", kind="line", ax=ax, label="I")
  df.plot(x="time_on_seconds", y="d_term", kind="line", ax=ax, label="D")
  df.plot(x="time_on_seconds", y="heater_power", kind="line", ax=ax, label="Heater power")
  plt.xlabel("Time on (seconds)")
  plt.ylabel("Values")
  plt.legend()
  plt.grid()
  plt.savefig(f"{plot_folder}{pid_plot_full_name}", dpi=500)
  print("PID plot full duration created")
else:
  print("PID plot full duration already created")

# Heater power
heater_power_plot_name = "heater_pwm_plot.png"
if not os.path.exists(f"{plot_folder}{heater_power_plot_name}"):
  # Create graph
  fig, ax1 = plt.subplots()
  ax1.set_xlabel("Time on [s]")
  ax1.set_ylabel("PWM signal")
  ax1.plot(df["time_on_seconds"], df["heater_power"], color="b", label="Heater pwm")
  ax1.tick_params(axis="y")

  df["heating_power"] = df["avg_battery_voltage"] * df["avg_heater_current"]

  ax2 = ax1.twinx()
  ax2.set_ylabel("Heating power [W]")
  ax2.plot(df["time_on_seconds"], df["heating_power"], color="r", label="Heating power")
  ax2.tick_params(axis="y")
  ax1.grid(True)
  ax1.legend()
  ax2.legend(loc="center right")
  plt.savefig(f"{plot_folder}{heater_power_plot_name}", dpi=500)
  print("Heater power plot created")
else:
  print("Heater power plot already created")

# Battery voltage
battery_voltage_plot_name = "battery_voltage_plot.png"
if not os.path.exists(f"{plot_folder}{battery_voltage_plot_name}"):
  fig, ax = plt.subplots()
  plt.gca().xaxis.set_major_locator(plt.MaxNLocator(12))
  df.plot(x="time_on_seconds", y="avg_battery_voltage", kind="line", ax=ax, label="Battery voltage")
  plt.xlabel("Time on (seconds)")
  plt.ylabel("Voltage (V)")
  plt.legend()
  plt.grid()
  plt.savefig(f"{plot_folder}{battery_voltage_plot_name}", dpi=500)
  print("Battery voltage plot created")
else:
  print("Battery voltage plot already created")
  
# Acceleration
acceleration_plot_name = "acceleration_plot.png"
if not os.path.exists(f"{plot_folder}{acceleration_plot_name}"):
  fig, ax = plt.subplots()
  plt.gca().xaxis.set_major_locator(plt.MaxNLocator(12))
  flight_df['smoothed_acc_x'] = flight_df['acc_x'].rolling(window=60, min_periods=1).mean()
  flight_df['smoothed_acc_y'] = flight_df['acc_y'].rolling(window=60, min_periods=1).mean()
  flight_df['smoothed_acc_z'] = flight_df['acc_z'].rolling(window=60, min_periods=1).mean()
  flight_df['total_acceleration'] = (flight_df['acc_x']**2 + flight_df['acc_y']**2 + flight_df['acc_z']**2)**0.5
  flight_df['smoothed_total_acceleration'] = flight_df['total_acceleration'].rolling(window=60, min_periods=1).mean()
  flight_df.plot(x="time_on_seconds", y="smoothed_acc_x", kind="line", ax=ax, label="Acc x")
  flight_df.plot(x="time_on_seconds", y="smoothed_acc_y", kind="line", ax=ax, label="Acc y")
  flight_df.plot(x="time_on_seconds", y="smoothed_acc_z", kind="line", ax=ax, label="Acc z")
  flight_df.plot(x="time_on_seconds", y="smoothed_total_acceleration", kind="line", ax=ax, label="Total acceleration")
  plt.xlabel("Time on (seconds)")
  plt.ylabel("Acceleration (m/s^2)")
  plt.legend()
  plt.grid()
  plt.savefig(f"{plot_folder}{acceleration_plot_name}", dpi=500)
  print("Acceleration plot created")
else:
  print("Acceleration plot already created")
  
# Gyro
gyro_plot_name = "gyro_plot.png"
if not os.path.exists(f"{plot_folder}{gyro_plot_name}"):
  fig, ax = plt.subplots()
  plt.gca().xaxis.set_major_locator(plt.MaxNLocator(12))
  flight_df['smoothed_gyro_x'] = flight_df['gyro_x'].rolling(window=60, min_periods=1).mean()
  flight_df['smoothed_gyro_y'] = flight_df['gyro_y'].rolling(window=60, min_periods=1).mean()
  flight_df['smoothed_gyro_z'] = flight_df['gyro_z'].rolling(window=60, min_periods=1).mean()
  flight_df.plot(x="time_on_seconds", y="smoothed_gyro_x", kind="line", ax=ax, label="Gyro x")
  flight_df.plot(x="time_on_seconds", y="smoothed_gyro_y", kind="line", ax=ax, label="Gyro y")
  flight_df.plot(x="time_on_seconds", y="smoothed_gyro_z", kind="line", ax=ax, label="Gyro z")
  plt.xlabel("Time on (seconds)")
  plt.ylabel("Rotation (rad/s)")
  plt.legend()
  plt.grid()
  plt.savefig(f"{plot_folder}{gyro_plot_name}", dpi=500)
  print("Gyro plot created")
else:
  print("Gyro plot already created")
  
# Ranging
ranging_plot_name = "ranging_plot.png"
if not os.path.exists(f"{plot_folder}{ranging_plot_name}"):
  fig, ax = plt.subplots()
  plt.gca().xaxis.set_major_locator(plt.MaxNLocator(12))
  df[df["time_on_seconds"] < 5500].plot(x="time_on_seconds", y="r1_dist", kind="line", ax=ax, label="Ranging 1")
  df[df["time_on_seconds"] < 5500].plot(x="time_on_seconds", y="r2_dist", kind="line", ax=ax, label="Ranging 2")
  plt.xlabel("Time on (seconds)")
  plt.ylabel("Ranging distance (meters)")
  plt.legend()
  plt.grid()
  plt.savefig(f"{plot_folder}{ranging_plot_name}", dpi=500)
  print("Ranging distance plot created")
else:
  print("Ranging distance plot already created")
  
# Ranging closeup
ranging_closeup_plot_name = "ranging_closeup_plot.png"
if not os.path.exists(f"{plot_folder}{ranging_closeup_plot_name}"):
  fig, ax = plt.subplots()
  plt.gca().xaxis.set_major_locator(plt.MaxNLocator(12))
  df[(df["time_on_seconds"] < 5400) & (df["time_on_seconds"] > 4730)].plot(x="time_on_seconds", y="r2_dist", kind="line", ax=ax, label="Ranging 2 distance")
  df[(df["time_on_seconds"] < 5400) & (df["time_on_seconds"] > 4730)].plot(x="time_on_seconds", y="gps_height", kind="line", ax=ax, label="GPS Altitude")
  plt.xlabel("Time on (seconds)")
  plt.ylabel("Altitude/Ranging distance (meters)")
  plt.legend()
  plt.grid()
  plt.savefig(f"{plot_folder}{ranging_closeup_plot_name}", dpi=500)
  print("Ranging closeup distance plot created")
else:
  print("Ranging closeup distance plot already created")
  