from jpkreader import load_jpk_file
import matplotlib.pyplot as plt

from pyafmrheo.utils.force_curves import *
from pyafmrheo.models.hertz import HertzModel

# Define global variables
file_path = '/Users/javierlopez/Documents/Marseille/05-11-2021/Rheo/map-data-2021.11.05-17.37.44.432.jpk-force-map'
# Shapes available: Paraboloid, Pyramidal
indenter_shape = "pyramid"
tip_radius = 5000 * 1e-9 # meters
tip_angle = 35 # degrees
poisson_ratio = 0.5
# If None it will use the deflection sensitivity from the file
deflection_sensitivity = None # m/V
# If None it will use the spring constant from the file
spring_constant = None # N/m

file = load_jpk_file(file_path)

ile_id = file.file_id
file_type = file.file_type
file_metadata = file.file_metadata
file_data = file.data
piezo_image = file.piezo_image
quality_map = file.quality_map

closed_loop = file_metadata['z_closed_loop_status']
file_deflection_sensitivity = file_metadata['original_deflection_sensitivity'] #nm/V
file_spring_constant = file_metadata['original_spring_constant'] #N/m
height_channel = file_metadata['height_channel_key']

if not deflection_sensitivity: deflection_sensitivity = file_deflection_sensitivity / 1e9
if not spring_constant: spring_constant = file_spring_constant

curve_idx = 0
force_curve = file_data[curve_idx]
extend_segments = force_curve.extend_segments
pause_segments = force_curve.pause_segments
modulation_segments = force_curve.modulation_segments
retract_segments = force_curve.retract_segments
force_curve_segments = [*extend_segments, *pause_segments, *modulation_segments, *retract_segments]
force_curve_segments = sorted(force_curve_segments, key=lambda x: int(x[0]))

app_deflection, app_height, app_time = preprocess_segment(extend_segments[0][1], height_channel, deflection_sensitivity)
ret_deflection, ret_height, ret_time = preprocess_segment(retract_segments[-1][1], height_channel, deflection_sensitivity)

xzero = ret_height[-1] # Maximum height
app_height = xzero - app_height

poc = get_poc_RoV_method(app_height, app_deflection)

poc[1] = 0
indentation, force = get_force_vs_indentation_curve(app_height, app_deflection, poc, spring_constant)

hertz = HertzModel(indenter_shape, tip_angle)
hertz.fit(indentation, force)

plt.plot(indentation, force)
plt.show()