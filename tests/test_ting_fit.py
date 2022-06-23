# Import libraries we will need
import time
import datetime

import numpy as np

# Get file reader from library
from jpkreader import load_jpk_file
import matplotlib.pyplot as plt

# Get data analysis tools
from pyafmrheo.utils.force_curves import *
from pyafmrheo.hertz_fit import HertzFit
from pyafmrheo.ting_fit import TingFit

# Define global variables
file_path = '/Users/javierlopez/Documents/Marseille/05-11-2021/E/map-data-2021.11.05-15.41.45.699.jpk-force-map'
# Shapes available: Paraboloid, Pyramidal
indenter_shape = "paraboloid"
tip_parameter = 75 * 1e-9 # meters
# tip_parameter = 35 # degrees
poisson_ratio = 0.5
# If None it will use the deflection sensitivity from the file
deflection_sensitivity = None # m/V
# If None it will use the spring constant from the file
spring_constant = None # N/m
contact_offset= 1

file = load_jpk_file(file_path)

file_id = file.file_id
file_type = file.file_type
file_metadata = file.file_metadata
file_data = file.data
piezo_image = file.piezo_image
quality_map = file.quality_map

closed_loop = file_metadata['z_closed_loop_status']
file_deflection_sensitivity = file_metadata['original_deflection_sensitivity'] #nm/V
file_spring_constant = file_metadata['original_spring_constant'] #N/m
height_channel = file_metadata['height_channel_key']

if not deflection_sensitivity: deflection_sensitivity = file_deflection_sensitivity / 1e9 #m/V
if not spring_constant: spring_constant = file_spring_constant

curve_idx = 0
force_curve = file_data[curve_idx]
extend_segments = force_curve.extend_segments
pause_segments = force_curve.pause_segments
modulation_segments = force_curve.modulation_segments
retract_segments = force_curve.retract_segments
force_curve_segments = [*extend_segments, *pause_segments, *modulation_segments, *retract_segments]
force_curve_segments = sorted(force_curve_segments, key=lambda x: int(x[0]))

t0 = 0
for seg_id, segment in force_curve_segments:
    height = segment.segment_formated_data[height_channel]
    time = segment.segment_formated_data["time"] + t0
    # plt.plot(time, height)
    t0 = time[-1]

# plt.xlabel("Time [s]")
# plt.ylabel("Height [Meters]")
# plt.grid()
# plt.show()

t0 = 0
for seg_id, segment in force_curve_segments:
    deflection = segment.segment_formated_data["vDeflection"]
    time = segment.segment_formated_data["time"] + t0
    # plt.plot(time, deflection)
    t0 = time[-1]

# plt.xlabel("Time [s]")
# plt.ylabel("Deflection [Volts]")
# plt.grid()
# plt.show()

for seg_id, segment in force_curve_segments:
    height = segment.segment_formated_data[height_channel]
    deflection = segment.segment_formated_data["vDeflection"]
    # plt.plot(height, deflection)

# plt.xlabel("Piezo Height [Meters]")
# plt.ylabel("vDeflection [Volts]")
# plt.grid()
# plt.show()

_, approach_segment = force_curve.extend_segments[0]
approach_segment_metadata = approach_segment.segment_metadata

app_deflection, app_height, app_time =\
    preprocess_segment(approach_segment, height_channel, deflection_sensitivity)

_, retract_segment = force_curve.retract_segments[-1]
retract_segment_metadata = retract_segment.segment_metadata
ret_deflection, ret_height, ret_time =\
    preprocess_segment(retract_segment, height_channel, deflection_sensitivity)

# Shift height
xzero = ret_height[-1] # Maximum height
app_height = xzero - app_height
ret_height = xzero - ret_height

# Find initial PoC, you may get some warnings.
poc = get_poc_RoV_method(app_height, app_deflection, win_size=100)
print(*poc)

# Hertz fit
poc[1] = 0

app_indentation, app_force =\
    get_force_vs_indentation_curve(app_height, app_deflection, poc, spring_constant)
ret_indentation, ret_force =\
    get_force_vs_indentation_curve(ret_height, ret_deflection, poc, spring_constant)

# Initial parameters
# delta0, f0, slope, E0
p0 = [0, 0, 1, 1000]

hertz_result = HertzFit(app_indentation, app_force, indenter_shape, tip_parameter, p0, poisson_ratio)

hertz_E = hertz_result.best_values['E0']
hertz_d0 = hertz_result.best_values['delta0']

# print(hertz_result.fit_report())

# Plot Hertz Fit results
# plt.plot(app_indentation, app_force)
# plt.plot(app_indentation, hertz_result.best_fit)
# plt.xlabel('Indentation [m]')
# plt.ylabel('Force [N]')
# plt.grid()
# plt.show()

# Get force vs indentation for all segments
poc[0] += hertz_d0
poc[1] = 0

# Get indentation and force
app_indentation, app_force = get_force_vs_indentation_curve(app_height, app_deflection, poc, spring_constant)
ret_indentation, ret_force = get_force_vs_indentation_curve(ret_height, ret_deflection, poc, spring_constant)

indentation = np.r_[app_indentation, ret_indentation]
force = np.r_[app_force, ret_force]
t0 = app_time[-1]
time = np.r_[app_time, ret_time + t0]

fit_mask = indentation > (-1 * contact_offset)
    
ind_fit = indentation[fit_mask] 
force_fit = force[fit_mask]
time_fit = time[fit_mask]
time_fit = time_fit - time_fit[0]

# Parameters for fit
dT = time_fit[1] - time_fit[0]
dTp = 1
E0 = hertz_E
fluidity_exponent = 0.15
smoothing_window = 5
tmax = time_fit[force_fit.argmax()]
v0t = (app_height[-1] - app_height[0]) / (app_time[-1] - app_time[0])
v0r = -1 * (ret_height[-1] - ret_height[0]) / (ret_time[-1] - ret_time[0])
print(v0t, v0r)
v0t = np.polyfit(app_time, app_indentation, 1)[0]
v0r = -1 * np.polyfit(ret_time, ret_indentation, 1)[0]
print(v0t, v0r)
slope = 0
tc = tmax/2
F0 = 0
d0 = 0
t0 = 1

p0_ting_num = [t0, E0, tc, fluidity_exponent, F0]

# resultTingNum = TingFit(
#     force_fit, ind_fit, time_fit, indenter_shape, tip_parameter, p0_ting_num, 'numerical', poisson_ratio, 0, smoothing_window
# )

# print(resultTingNum.fit_report())

p0_ting_anal = [t0, E0, tc, fluidity_exponent, F0]

first_time = datetime.datetime.now()

resultTingAnal = TingFit(
     force_fit, ind_fit, time_fit, indenter_shape, tip_parameter, p0_ting_anal, 'analytical', poisson_ratio, 0, smoothing_window
)

later_time = datetime.datetime.now()

difference = later_time - first_time

print(difference)

print(resultTingAnal.fit_report())

figure, axis = plt.subplots(1, 2)

axis[0].plot(ind_fit, force_fit)
# axis[0].plot(ind_fit, resultTingNum.best_fit, '--', label='Ting Numercial best fit')
axis[0].plot(ind_fit, resultTingAnal.best_fit, '--', label='Ting Analytical best fit')
axis[0].set_xlabel("Indentation [m]")
axis[0].set_ylabel("Force[N]")

axis[1].plot(time_fit, force_fit)
# axis[1].plot(time_fit, resultTingNum.best_fit, '--', label='Ting Numercial best fit')
axis[1].plot(time_fit, resultTingAnal.best_fit, '--', label='Ting Analytical best fit')
axis[1].set_xlabel("Time [s]")
axis[1].set_ylabel("Force[N]")

plt.legend()
plt.grid()
plt.show()