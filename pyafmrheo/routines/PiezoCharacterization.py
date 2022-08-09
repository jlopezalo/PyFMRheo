from ..utils.signal_processing import detrend_rolling_average
from ..models.rheology import ComputePiezoLag

def doPiezoCharacterization(fdc, param_dict):
    results = []
    for _, segment in fdc.modulation_segments:
        time = segment.time
        zheight = segment.zheight
        deflection = segment.vdeflection
        frequency = segment.segment_metadata['frequency']
        if param_dict['max_freq'] != 0 and frequency > param_dict['max_freq']:
            continue
        deltat = time[1] - time[0]
        fs = 1 / deltat
        ntra_in, ntra_out, _ =\
            detrend_rolling_average(frequency, zheight, deflection, time, 'zheight', 'deflection', [])
        fi, amp_quotient, gamma2 =\
            ComputePiezoLag(ntra_in, ntra_out, fs, frequency)
        results.append((frequency, fi, amp_quotient, gamma2))
    results = sorted(results, key=lambda x: int(x[0]))
    frequencies_results = [x[0] for x in results]
    fi_results = [x[1] for x in results]
    amp_quotient_results = [x[2] for x in results]
    gamma2_results = [x[3] for x in results]

    return (frequencies_results, fi_results, amp_quotient_results, gamma2_results)
