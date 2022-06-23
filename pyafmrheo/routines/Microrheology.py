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
from pyafmrheo.models.old.rheology import ComputeComplexModulus


def doMicrorheologyAnalysis(
    file_path, poc_win_size, indenter_shape, tip_parameter, hertz_p0=None, poisson_ratio=0.5,
    deflection_sensitivity=None, spring_constant=None, bcoef=0, piezoCharData=None
):

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

    fi = 0  # Default fi, in degrees
    amp_quotient = 1 # Default amplitude quotient

    for curve_idx in range(len(file_data)):
        curve_results = pd.DataFrame(
            columns=[
                "file_id", "curve_idx", "seg_id", "indenter_shape", "hertz_E", "tip_parameter",
                "frequency", "G_storage", "G_loss", "loss_tan", "fi", "amp_quotient", "inVols", "k"
            ]
        )

        force_curve = file_data[curve_idx]
        extend_segments = force_curve.extend_segments
        pause_segments = force_curve.pause_segments
        modulation_segments = force_curve.modulation_segments
        retract_segments = force_curve.retract_segments
        force_curve_segments = [*extend_segments, *pause_segments, *modulation_segments, *retract_segments]
        force_curve_segments = sorted(force_curve_segments, key=lambda x: int(x[0]))

        # Get approach, first extend segment
        _, first_ext_seg = extend_segments[0]
        app_deflection, app_height, app_time =\
            preprocess_segment(first_ext_seg, height_channel, deflection_sensitivity)

        # Get retract, last retract segment
        _, last_ret_seg = retract_segments[-1]
        ret_deflection, ret_height, ret_time =\
            preprocess_segment(last_ret_seg, height_channel, deflection_sensitivity)

        if pause_segments:
            # Get first pause segment
            _, first_pause_seg = pause_segments[0]
            pau_deflection, pau_height, _ =\
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

        # Find initial PoC, you may get some warnings.
        poc = get_poc_RoV_method(app_height, app_deflection, win_size=poc_win_size)

        # Hertz fit
        poc[1] = 0

        app_indentation, app_force = get_force_vs_indentation_curve(app_height, app_deflection, poc, spring_constant)

        # Initial parameters
        if hertz_p0 is None:
            # delta0, f0, slope, E0
            hertz_p0 = [0, 0, 0, 100000]

        hertz_result = HertzFit(app_indentation, app_force, indenter_shape, tip_parameter, hertz_p0, poisson_ratio)

        hertz_E = hertz_result.best_values['E0']
        hertz_d0 = hertz_result.best_values['delta0']

        # Get force vs indentation for all segments
        poc[0] += hertz_d0
        poc[1] = 0

        # Get indentation and force
        app_indentation, app_force = get_force_vs_indentation_curve(app_height, app_deflection, poc, spring_constant)

        # Get working indentation
        wc = app_indentation.max()

        results = []
        poc = [0, 0] # Assume d0 as 0, since we are in contact.

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
            G_storage, G_loss, gamma2 =\
                ComputeComplexModulus(
                    deflection, zheight, poc, spring_constant, fs, frequency, indenter_shape,
                    tip_parameter, wc, poisson_ratio, fi, amp_quotient, bcoef
                )
            loss_tan = G_loss / G_storage
            results.append((seg_id, frequency, G_storage, G_loss, loss_tan, gamma2))

        results = sorted(results, key=lambda x: int(x[0]))
        seg_id_results = [x[0] for x in results]
        frequencies_results = [x[1] for x in results]
        G_storage_results = [x[2] for x in results]
        G_loss_results = [x[3] for x in results]
        loss_tan_results = [x[4] for x in results]

        curve_results["frequency"] = frequencies_results
        curve_results["G_storage"] = G_storage_results
        curve_results["G_loss"] = G_loss_results
        curve_results["loss_tan"] = loss_tan_results
        curve_results["fi"] = fi
        curve_results["amp_quotient"] = amp_quotient
        curve_results["hertz_E"] = hertz_E
        curve_results["indenter_shape"] = indenter_shape
        curve_results["tip_parameter"] = tip_parameter
        curve_results["seg_id"] = seg_id_results
        curve_results["file_id"] = file_id
        curve_results["inVols"] = deflection_sensitivity
        curve_results["k"] = spring_constant
        curve_results["curve_idx"] = curve_idx

        file_results.append(curve_results)
    
    return pd.concat(file_results, ignore_index=True)


if __name__ == "__main__":
    file_path = '/Users/javierlopez/Desktop/Gels-16122021/AGAROSE1/HeadRheo/map-data-2021.12.16-19.30.16.885.jpk-force-map'
    poc_win_size = 50
    indenter_shape = "paraboloid"
    tip_parameter = 75 * 1e-9 # meters
    # tip_parameter = 35 # degrees
    # Poisson ratio
    poisson_ratio = 0.5
    result = doMicrorheologyAnalysis(file_path, poc_win_size, indenter_shape, tip_parameter, poisson_ratio=poisson_ratio)
    print(result)