# Import libraries we will need
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend

# Get file reader from library
from pyafmreader import loadfile

# Get data analysis tools
from pyafmrheo.utils.force_curves import *
from pyafmrheo.models.hertz import HertzModel
from pyafmrheo.models.ting import TingModel

# Define global variables
file_path = '/Users/javierlopez/Documents/Datasets/05-11-2021/E/map-data-2021.11.05-15.41.45.699.jpk-force-map'
# Shapes available: paraboloid, pyramid
indenter_shape = "paraboloid"
tip_parameter = 30 * 1e-9 # meters
# Poisson ratio
poisson_ratio = 0.5
# Max non contact region
maxnoncontact = 2.5 * 1e-6
# Window to find cp
windowforCP = 70 * 1e-9
# Smooth window
smooth_w = 1
# t0 scaling factor
t0 = 1
# Viscous drag for PFQNM
vdrag = 0.77*1e-6
# If None it will use the deflection sensitivity from the file
deflection_sensitivity = None # m/V
# If None it will use the spring constant from the file
spring_constant = None # N/m
# General plotting params
plt.rcParams["figure.figsize"] = (10,5)

file = loadfile(file_path)

filemetadata = file.filemetadata
print(filemetadata['file_type'])

closed_loop = filemetadata['z_closed_loop']
file_deflection_sensitivity = filemetadata['defl_sens_nmbyV'] #nm/V
file_spring_constant = filemetadata['spring_const_Nbym'] #N/m
height_channel = filemetadata['height_channel_key']

if not deflection_sensitivity: deflection_sensitivity = file_deflection_sensitivity / 1e9 #m/V
if not spring_constant: spring_constant = file_spring_constant

curve_idx = 0
force_curve = file.getcurve(curve_idx)
extend_segments = force_curve.extend_segments
pause_segments = force_curve.pause_segments
modulation_segments = force_curve.modulation_segments
retract_segments = force_curve.retract_segments
force_curve_segments = force_curve.get_segments()

t0 = 0
for seg_id, segment in force_curve_segments:
    height = segment.segment_formated_data[height_channel]
    time = segment.segment_formated_data["time"] + t0
    plt.plot(time, height)
    t0 = time[-1]

plt.xlabel("Time [s]")
plt.ylabel("Height [Meters]")
plt.grid()
# plt.show()

t0 = 0
for seg_id, segment in force_curve_segments:
    deflection = segment.segment_formated_data["vDeflection"]
    time = segment.segment_formated_data["time"] + t0
    plt.plot(time, deflection)
    t0 = time[-1]

plt.xlabel("Time [s]")
plt.ylabel("Deflection [Volts]")
plt.grid()
# plt.show()

for seg_id, segment in force_curve_segments:
    height = segment.segment_formated_data[height_channel]
    deflection = segment.segment_formated_data["vDeflection"]
    plt.plot(height, deflection)

plt.xlabel("Piezo Height [Meters]")
plt.ylabel("vDeflection [Volts]")
plt.grid()
# plt.show()

# Get approach, first extend segment
first_exted_seg_id, first_ext_seg = extend_segments[0]
first_ext_seg.preprocess_segment(deflection_sensitivity, height_channel)

# Get retract, last retract segment
last_ret_seg_id, last_ret_seg = retract_segments[-1]
last_ret_seg.preprocess_segment(deflection_sensitivity, height_channel)

# Shift height
xzero = last_ret_seg.zheight[-1] # Maximum height
first_ext_seg.zheight = xzero - first_ext_seg.zheight
last_ret_seg.zheight = xzero - last_ret_seg.zheight

app_height = first_ext_seg.zheight
app_deflection = first_ext_seg.vdeflection
ret_height = last_ret_seg.zheight
ret_deflection = last_ret_seg.vdeflection

# Find initial PoC, you may get some warnings.
poc = get_poc_RoV_method(app_height, app_deflection, 350e-9)
print(*poc)

plt.plot(app_height, app_deflection)
plt.plot(ret_height, ret_deflection)
plt.axvline(x=poc[0], color='r', linestyle='--')
plt.axhline(y=poc[1], color='r', linestyle='--')
plt.grid()
# plt.show()

height = np.r_[app_height, ret_height]
deflection = np.r_[app_deflection, ret_deflection]
idx = len(app_height)
corr_defl = correct_tilt(height, deflection, poc[0], 1e-6, 10e-9)
plt.axvline(x=poc[0]-1e-6, color='r', linestyle='--')
plt.axvline(x=poc[0]-10e-9, color='g', linestyle='--')
plt.plot(height[:idx], corr_defl[:idx])
plt.plot(height[idx:], corr_defl[idx:])
plt.grid()
# plt.show()

app_deflection = corr_defl[:idx]
ret_deflection = corr_defl[idx:]

# Hertz fit
poc[1] = 0
first_ext_seg.get_force_vs_indentation(poc, spring_constant)
app_indentation, app_force = first_ext_seg.indentation, first_ext_seg.force

hertzmodel = HertzModel(indenter_shape, tip_parameter)
hertzmodel.fit(app_indentation, app_force)

hertzmodel.fit_report()

hertz_E0 = hertzmodel.E0
hertz_d0 = hertzmodel.delta0
hertz_f0 = hertzmodel.f0

# Plot Hertz Fit results
plt.plot(app_indentation-hertz_d0, app_force, label="Experimental Data")
plt.plot(app_indentation-hertz_d0, hertzmodel.eval(app_indentation)-hertz_f0, label="Model Fit")
plt.xlabel('Indentation [m]')
plt.ylabel('Force [N]')
plt.legend()
plt.grid()
# plt.show()

# Plot residuals
plt.plot(app_indentation-hertz_d0, hertzmodel.get_residuals(app_indentation, app_force), "bo")
plt.xlabel('Indentation [m]')
plt.ylabel('Residuals [N]')
plt.grid()
# plt.show()

# Get force vs indentation for all segments
poc[0] += hertz_d0

# Get indentation and force
first_ext_seg.get_force_vs_indentation(poc, spring_constant)
app_indentation, app_force, app_time = first_ext_seg.indentation, first_ext_seg.force, first_ext_seg.time
last_ret_seg.get_force_vs_indentation(poc, spring_constant)
ret_indentation, ret_force, ret_time = last_ret_seg.indentation, last_ret_seg.force, last_ret_seg.time

idx_tc = (np.abs(app_indentation - 0)).argmin()
t0 = app_time[-1]
indentation = np.r_[app_indentation, ret_indentation]
time = np.r_[app_time, ret_time + t0]
force = np.r_[app_force, ret_force]
fit_mask = indentation > (-1 * maxnoncontact)
tc = time[idx_tc]
ind_fit = indentation[fit_mask]
force_fit = force[fit_mask]
force_fit = force_fit - force_fit[0]
time_fit = time[fit_mask]
tc_fit = tc-time_fit[0]
time_fit = time_fit - time_fit[0] - tc_fit
tc_fit = 0
tm = time_fit[np.argmax(force_fit)]

v0t = np.abs(first_ext_seg.zheight.min() - first_ext_seg.zheight.max())/first_ext_seg.segment_metadata['duration']
v0r = np.abs(last_ret_seg.zheight.min() - last_ret_seg.zheight.max())/first_ext_seg.segment_metadata['duration']

downfactor= len(time_fit) // 300
print(f'Downfactor --> {downfactor}')
print(f'tm --> {tm}')

idxDown = list(range(0, len(time_fit), downfactor))
idx_tm = np.argmax(force_fit[idxDown])
f0idx = np.where(time_fit==0)[0]
F0_init=force_fit[f0idx]-vdrag*v0t

tc_max = tc_fit+downfactor/(1/(time_fit[1]-time_fit[0]))*10
tc_min = tc_fit-downfactor/(1/(time_fit[1]-time_fit[0]))*10
f0_max = F0_init+100e-12
f0_min = F0_init-100e-12

if hertz_E0 > 10e3:
    betaE_init = 0.05
else:
    betaE_init = 0.25

plt.plot(time_fit[idxDown], force_fit[idxDown], 'o')
plt.axvline(x=time_fit[idxDown][idx_tm], color='m', linestyle='--', label='tm')
plt.axvline(x=tc_min, color='r', linestyle='--', label='tc_min')
plt.axvline(x=tc_max, color='r', linestyle='--', label='tc_min')
plt.axvline(x=tc_fit, color='g', linestyle='--', label='tc')
plt.axhline(y=F0_init, color='k', linestyle='--', label='f0')
plt.axhline(y=f0_max, color='b', linestyle='--', label='f0_max')
plt.axhline(y=f0_min, color='b', linestyle='--', label='f0_min')
plt.legend()
# plt.show()

model_ting_analytical = TingModel(indenter_shape, tip_parameter, 'analytical')
# E0
model_ting_analytical.E0_init = hertz_E0
model_ting_analytical.E0_min = hertz_E0/1000
model_ting_analytical.E0_max = np.inf
# tc
model_ting_analytical.tc_init = tc_fit
model_ting_analytical.tc_min = tc_min
model_ting_analytical.tc_max = tc_max
# betaE
model_ting_analytical.betaE_init = betaE_init
model_ting_analytical.betaE_min = 0.01
model_ting_analytical.betaE_max = 0.49
# F0
model_ting_analytical.F0_init = F0_init
model_ting_analytical.F0_min = f0_min
model_ting_analytical.F0_max = f0_max

# Do fit
model_ting_analytical.fit(
    time_fit, force_fit, ind_fit,
    t0, idx_tm, smooth_w, v0t, v0r
)