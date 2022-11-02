# Import libraries we will need
import numpy as np
from ..utils.signal_processing import detrend_rolling_average
from ..models.rheology import ComputeBh

def get_retract_ramp_sizes(force_curve):
    x0 = 0
    distances = []
    sorted_ret_segments = sorted(force_curve.retract_segments, key=lambda x: int(x[0]))
    print(sorted_ret_segments)
    for _, ret_seg in sorted_ret_segments[:-1]:
        # Maybe in the future do not use the ramp size from header and compute
        # ramp size as zmax - zmin?
        distance_from_sample = -1 * ret_seg.segment_metadata['ramp_size'] + x0 # Negative
        distances.append(distance_from_sample * 1e-9) # in nm
        x0 = distance_from_sample
    return distances

def doViscousDragSteps(fdc, param_dict):
    # Declare preset params for correcting the raw signals
    fi = 0
    amp_quotient = 1
    # Get list with the distances from the sample of each segment
    distances = get_retract_ramp_sizes(fdc)
    # Declare empty list to save the results of the different
    # modulation segments of the curve
    results = []
    # Iterate thorugh the modulation segments
    # and perform the analysis 
    for seg_id, segment in fdc.modulation_segments:
        time = segment.time
        zheight = segment.zheight
        deflection = segment.vdeflection
        frequency = segment.segment_metadata['frequency']
        # The user can determine a maximum frequency to analyze
        # If the frequency of the segment is higher than the threshold frequency
        # skip this segment
        if param_dict['max_freq'] != 0 and frequency > param_dict['max_freq']:
            continue
        deltat = time[1] - time[0]
        fs = 1 / deltat
        # If piezo characterization data has been provided get fi and amp_quotient
        # for the segment's frequency
        if param_dict['piezo_char_data'] is not None:
            piezoChar =  param_dict['piezo_char_data'].loc[param_dict['piezo_char_data']['frequency'] == frequency]
            if len(piezoChar) == 0:
                print(f"The frequency {frequency} was not found in the piezo characterization dataframe")
            else:
                fi = piezoChar['fi_degrees'].item() # In degrees
                if param_dict['corr_amp']:
                    amp_quotient = piezoChar['amp_quotient'].item()
                else:
                    amp_quotient = 1
        # Detrend the input signals using the rolling average method
        zheight, deflection, _ =\
            detrend_rolling_average(frequency, zheight, deflection, time, 'zheight', 'deflection', [])
        # Get Bh
        Bh, Hd, gamma2 =\
            ComputeBh(
                deflection, zheight, [0, 0], param_dict['k'],
                fs, frequency, fi=fi, amp_quotient=amp_quotient
            )
        # Append segment results
        results.append((seg_id, frequency, Bh, Hd, gamma2, fi, amp_quotient))
    # Organize and unpack the results for the different segments
    # As in this routine we expect to have the same frequency on all segments,
    # the segment ID is used to organize the data instead.
    results = sorted(results, key=lambda x: int(x[0]))
    frequencies_results = [x[1] for x in results]
    Bh_results = [x[2] for x in results]
    Hd_results = np.array([x[3] for x in results])
    gamma2_results = [x[4] for x in results]
    fi_results = [x[5] for x in results]
    amp_quotient_results = [x[6] for x in results]
    return (frequencies_results, Bh_results, Hd_results, gamma2_results, distances, fi_results, amp_quotient_results)