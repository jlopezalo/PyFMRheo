# Import libraries we will need
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend

# Get file reader from library
from jpkreader import load_jpk_file

# Get data analysis tools
from pyafmrheo.utils.force_curves import *
from pyafmrheo.utils.signal_processing import *
from pyafmrheo.hertz_fit import HertzFit
from pyafmrheo.models.rheology import ComputeBh


def doViscousDrag(file_path, deflection_sensitivity=None, spring_constant=None):

    file_results = pd.DataFrame(columns=["file_id", "curve_idx", "distance","frequency", "Bh", "Hd", "inVols"])

    file = load_jpk_file(file_path)

    file_id = file.file_id
    file_metadata = file.file_metadata
    file_data = file.data

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

    # Get distance from the sample, needed to compute Bh
    _, first_ret_seg = retract_segments[0]
    distance_from_sample = -1 * first_ret_seg.segment_metadata['ramp_size'] # Negative
    print(f"Distance from sample {distance_from_sample * 1e-9} m")

    # Get approach, first extend segment
    _, first_ext_seg = extend_segments[0]
    _, app_height, _ =\
        preprocess_segment(first_ext_seg, height_channel, deflection_sensitivity)

    # Get retract, last retract segment
    _, last_ret_seg = retract_segments[-1]
    _, ret_height, _ =\
        preprocess_segment(last_ret_seg, height_channel, deflection_sensitivity)

    if pause_segments:
        # Get first pause segment
        _, first_pause_seg = pause_segments[0]
        _, pau_height, _ =\
            preprocess_segment(first_pause_seg, height_channel, deflection_sensitivity)

    if modulation_segments:
        modulation_data = {}
        for seg_id, seg_data in modulation_segments:
            segment_data = {}
            mod_deflection, mod_height, mod_time =\
                preprocess_segment(seg_data, height_channel, deflection_sensitivity)
            segment_data['height'] = mod_height
            segment_data['deflection'] = mod_deflection
            segment_data['time'] = mod_time
            modulation_data[seg_id] = segment_data

    # Shift height
    xzero = ret_height[-1] # Maximum height
    app_height = xzero - app_height
    ret_height = xzero - ret_height
    if pause_segments:
        pau_height = xzero - pau_height
    if modulation_segments:
        for segment_data in modulation_data.values():
            segment_data['height'] = xzero - segment_data['height']

    if modulation_segments:
        for segment_data in modulation_data.values():
            mod_indentation, mod_force =\
                get_force_vs_indentation_curve(segment_data['height'], segment_data['deflection'], [0,0], spring_constant)
            segment_data['indentation'] = mod_indentation
            segment_data['force'] = mod_force

    results = []

    for seg_id, seg_data in modulation_segments:
        frequency = seg_data.segment_metadata["frequency"]
        data = modulation_data[seg_id]
        time = data['time']
        indentation = data['indentation']
        force = data['force']
        deltat = time[1] - time[0]
        fs = 1 / deltat
        ntra_in, ntra_out, _ =\
            detrend_rolling_average(frequency, indentation, force, time, 'indentation', 'force', [])
        Bh, Hd, gamma2 = ComputeBh(ntra_out, ntra_in, fs, frequency)
        results.append((frequency, Bh, Hd, gamma2))

    results = sorted(results, key=lambda x: int(x[0]))
    frequencies_results = [x[0] for x in results]
    Bh_results = [x[1] for x in results]
    Hd_results = [x[2] for x in results]

    file_results["frequency"] = frequencies_results
    file_results["Bh"] = Bh_results
    file_results["Hd"] = Hd_results
    file_results["distance"] = distance_from_sample * 1e-9
    file_results["file_id"] = file_id
    file_results["inVols"] = deflection_sensitivity
    file_results["curve_idx"] = curve_idx
    
    return file_results

if __name__ == "__main__":
    file_path = '/Users/javierlopez/Documents/Marseille/29102021/Viscous Drag Correction - H20/Pos1/force-save-2021.10.29-16.43.34.600.jpk-force'
    result = doViscousDrag(file_path)
    print(result)