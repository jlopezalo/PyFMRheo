# Import libraries we will need
from _typeshed import NoneType
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend

# Get file reader from library
from jpkreader import load_jpk_file

# Get data analysis tools
from pyafmrheo.utils.force_curves import *
from pyafmrheo.utils.signal_processing import *
from pyafmrheo.models.rheology import ComputeBh


def doViscousDragMultiFreq(file_path, deflection_sensitivity=None, spring_constant=None, piezoCharData=None):

    file_results = []

    file = load_jpk_file(file_path)

    file_id = file.file_id
    file_metadata = file.file_metadata
    file_data = file.data

    file_deflection_sensitivity = file_metadata['original_deflection_sensitivity'] #nm/V
    file_spring_constant = file_metadata['original_spring_constant'] #N/m
    height_channel = file_metadata['height_channel_key']

    if not deflection_sensitivity: deflection_sensitivity = file_deflection_sensitivity / 1e9 #m/V
    if not spring_constant: spring_constant = file_spring_constant

    for curve_idx in range(len(file_data)):
        curve_results = pd.DataFrame(
            columns=["file_id", "curve_idx", "distance","frequency", "Bh", "Hd", "inVols", "k"]
        )
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

        results = []
        poc = [0, 0] # Assume d0 as 0, since we are not in contact.
        fi = 0  # Default fi, in degrees
        amp_quotient = 1 # Default amplitude quotient

        for seg_id, seg_data in modulation_segments:
            frequency = seg_data.segment_metadata["frequency"]
            data = modulation_data[seg_id]
            time = data['time']
            zheight = data['height']
            deflection = data['deflection']
            deltat = time[1] - time[0]
            fs = 1 / deltat
            if piezoCharData is not None:
                piezoChar = piezoCharData.loc[piezoCharData['frequency'] == frequency]
                if len(piezoChar) == 0:
                    print(f"The frequency {frequency} was not found in the piezo characterization dataframe")
                else:
                    fi = piezoChar['fi_degrees'].item() # In degrees
                    amp_quotient = piezoChar['amp_quotient'].item()
            zheight, deflection, _ =\
                detrend_rolling_average(frequency, zheight, deflection, time, 'zheight', 'deflection', [])
            Bh, Hd, gamma2 = ComputeBh(
                deflection, zheight, poc, spring_constant, fs, frequency, fi=fi, amp_quotient=amp_quotient
            )
            results.append((frequency, Bh, Hd, gamma2))

        results = sorted(results, key=lambda x: int(x[0]))
        frequencies_results = [x[0] for x in results]
        Bh_results = [x[1] for x in results]
        Hd_results = [x[2] for x in results]

        curve_results["frequency"] = frequencies_results
        curve_results["Bh"] = Bh_results
        curve_results["Hd"] = Hd_results
        curve_results["distance"] = distance_from_sample * 1e-9
        curve_results["file_id"] = file_id
        curve_results["inVols"] = deflection_sensitivity
        curve_results["k"] = spring_constant
        curve_results["curve_idx"] = curve_idx

        file_results.append(curve_results)
    
    return pd.concat(file_results, ignore_index=True)

if __name__ == "__main__":
    file_path = '/Users/javierlopez/Documents/Marseille/29102021/Viscous Drag Correction - H20/Pos1/force-save-2021.10.29-16.43.34.600.jpk-force'
    result = doViscousDragMultiFreq(file_path)
    print(result)