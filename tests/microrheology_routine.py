import os
from numpy.core.fromnumeric import var
import regex as re
import matplotlib.pyplot as plt
import numpy as np
from lmfit import Parameters, minimize
from lmfit.printfuncs import report_fit

from jpkreader import load_jpk_file
from src.models.hertz import hertz_model
from src.models.single_frequency_microrheology import ComputeComplexModulus
from src.utils.force_curves import get_poc_RoV_method

# Constants
modelProbe = "pyramid"
halfAngle = 35
poisson_ratio = 0.5
folder_path = "/Users/javierlopez/Desktop/NoppaternsNNIH3T3/MR/Cell1Nucleus/"
keyword = r"(Cell)([0-9])(Nucleus)"
pattern = re.compile(keyword)

def residual(pars, x, model, data=None):
    simulated = model(x, **pars)
    if data is None:
        return simulated
    return simulated - data

def preprocess_segment(segment, file_metadata):
    # For JPK files the deflection sensitivity is in nm/volts.
    # Divide by 1e9 to transform into m/volts
    # volt * (nm / volt) * (m / 1e9 nm)
    deflection_sensitivity = file_metadata["original_deflection_sensitivity"] # nm / volts
    scaled_deflection_sensitivity = deflection_sensitivity / 1e9 # TO DO: Do not hard code? Is it always in nm?

    spring_constant = file_metadata["original_spring_constant"]

    print(spring_constant)

    height_channel_key = file_metadata["height_channel_key"]

    # Get experimental parameters from metadata
    vDelfection_v = segment.segment_formated_data["vDeflection"]
    if segment.segment_metadata["baseline_measured"]:
        vDelfection_v = vDelfection_v - segment.segment_metadata["baseline"]
    else:
        y0 = vDelfection_v[0]
        vDelfection_v = vDelfection_v - y0
    vDelfection_m = vDelfection_v * scaled_deflection_sensitivity
    zheight = segment.segment_formated_data[height_channel_key]
    time = segment.segment_formated_data["time"]
    tip_position = zheight - vDelfection_m
    force = vDelfection_m * spring_constant

    return vDelfection_m, zheight, time, tip_position, force

for root, dirs, files in os.walk(folder_path):
    for file_name in files:
        file_path = os.path.join(root, file_name)
        cell_id = pattern.findall(file_path)
        if cell_id:
            # Load file
            loaded_file = load_jpk_file(file_path)
            file_data = loaded_file.data
            file_metadata = loaded_file.file_metadata
            for curve in file_data:
                curve_index = curve.curve_index
                file_id = curve.file_id
                print(file_id)
                extend_segments = curve.extend_segments
                modulation_segments = curve.modulation_segments
                # Get first extend segment to compute PoC
                approach_segment_id, approach_segment = extend_segments[0]
                approach_metadata = approach_segment.segment_metadata
                # Preprocess data
                app_data = preprocess_segment(approach_segment, file_metadata)
                app_deflection, app_zheight, app_time, app_tip_pos, app_force = app_data
                # Find initial PoC
                poc = get_poc_RoV_method(app_zheight, app_deflection, 20)
                app_indentation = -1 * (app_zheight - app_deflection - poc[0])
                # Hertz Fit to find PoC
                pass

            # Hertz Fit to find PoC
            # Find Working Indentation
            # Compute G'and G'' for each segment / Frequency
            # Save data to dataframe
            # Export dataframe to excel
            # For every frequency compute the geomean and then do the double power law fit
            break