# Import libraries we will need
import pandas as pd

# Get file reader from library
from jpkreader import load_jpk_file

# Get data analysis tools
from pyafmrheo.utils.force_curves import *
from pyafmrheo.utils.signal_processing import *
from pyafmrheo.models.old.rheology import ComputePiezoLag

def doPiezoCharacterization(file_path, deflection_sensitivity=None, spring_constant=None):

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
            columns=["file_id", "curve_idx", "frequency", "fi_degrees", "amp_quotient", "inVols"])
        force_curve = file_data[curve_idx]
        extend_segments = force_curve.extend_segments
        pause_segments = force_curve.pause_segments
        modulation_segments = force_curve.modulation_segments
        retract_segments = force_curve.retract_segments
        force_curve_segments = [*extend_segments, *pause_segments, *modulation_segments, *retract_segments]
        force_curve_segments = sorted(force_curve_segments, key=lambda x: int(x[0]))

        # Get retract, last retract segment
        _, last_ret_seg = retract_segments[-1]
        _, ret_height, _ =\
            preprocess_segment(last_ret_seg, height_channel, deflection_sensitivity)

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
        if modulation_segments:
            for segment_data in modulation_data.values():
                segment_data['height'] = xzero - segment_data['height']

        results = []

        for seg_id, seg_data in modulation_segments:
            frequency = seg_data.segment_metadata["frequency"]
            data = modulation_data[seg_id]
            time = data['time']
            zheight = data['height']
            delfection = data['deflection']
            deltat = time[1] - time[0]
            fs = 1 / deltat
            ntra_in, ntra_out, _ =\
                detrend_rolling_average(frequency, zheight, delfection, time, 'indentation', 'force', [])
            fi, amp_quotient, gamma2 =\
                ComputePiezoLag(ntra_in, ntra_out, fs, frequency)
            results.append((frequency, fi, amp_quotient, gamma2))

        results = sorted(results, key=lambda x: int(x[0]))
        frequencies_results = [x[0] for x in results]
        fi_results = [x[1] for x in results]
        amp_quotient_results = [x[2] for x in results]

        curve_results["frequency"] = frequencies_results
        curve_results["fi_degrees"] = fi_results
        curve_results["amp_quotient"] = amp_quotient_results
        curve_results["file_id"] = file_id
        curve_results["inVols"] = deflection_sensitivity
        curve_results["curve_idx"] = curve_idx

        file_results.append(curve_results)
    
    return pd.concat(file_results, ignore_index=True)

if __name__ == "__main__":
    file_path = '/Users/javierlopez/Desktop/Data/javistuff/Gels-15122021/HeadPiezo/Calibration/force-save-2021.12.15-16.00.36.002.jpk-force'
    result = doPiezoCharacterization(file_path)
    print(result)




