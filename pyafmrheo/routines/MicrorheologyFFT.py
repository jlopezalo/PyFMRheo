# Get data analysis tools
from ..utils.force_curves import get_poc_RoV_method, get_poc_regulaFalsi_method
from ..utils.signal_processing import detrend_rolling_average
from .HertzFit import doHertzFit
from ..models.rheology import ComputeComplexModulusFFT


def doMicrorheologyFFT(fdc, param_dict):
    # Declare preset params for correcting the raw signals
    fi = 0
    amp_quotient = 1
    # Get segment data to obtain the working indentation 
    if param_dict['curve_seg'] == 'extend':
        segment_data = fdc.extend_segments[0][1]
    else:
        segment_data = fdc.retract_segments[-1][1]
        segment_data.zheight = segment_data.zheight[::-1]
        segment_data.vdeflection = segment_data.vdeflection[::-1]
    # Get initial estimate of PoC
    if param_dict['poc_method'] == 'RoV':
        comp_PoC = get_poc_RoV_method(
            segment_data.zheight, segment_data.vdeflection, param_dict['poc_win'])
    else:
        comp_PoC = get_poc_regulaFalsi_method(
            segment_data.zheight, segment_data.vdeflection, param_dict['sigma'])
    poc = [comp_PoC[0], 0]
    # Perform HertzFit to obtain refined posiiton of PoC
    hertz_result = doHertzFit(fdc, param_dict)
    hertz_d0 = hertz_result.delta0
    poc[0] += hertz_d0
    poc[1] = 0
    # Get force vs indentation data
    segment_data.get_force_vs_indentation(poc, param_dict['k'])
    app_indentation = segment_data.indentation
    # Get working indentation from the parameters or the approach segment.
    # Some SFC do not have a good baseline in the approach segment.
    # Give the user the option to provide a value of working indentation 
    # to compute G*.
    if param_dict.get('wc') is None:
        wc = app_indentation.max()
    else:
        wc = param_dict.get('wc')
    # Get bcoef
    bcoef = param_dict['bcoef']
    # Declare empty list to save the results of the different
    # modulation segments of the curve
    results = []
    # Assume d0 as 0, since we are in contact
    poc = [0, 0]
    # Iterate thorugh the modulation segments
    # and perform the analysis 
    for _, segment in fdc.modulation_segments:
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
        # Get G' and G" using the transfer function method
        G_storage, G_loss, gamma2 =\
            ComputeComplexModulusFFT(
                deflection, zheight, poc, param_dict['k'], fs, frequency, param_dict['contact_model'],
                param_dict['tip_param'], wc, param_dict['poisson'], fi=fi, amp_quotient=amp_quotient, bcoef=bcoef
            )
        # Append segment results
        results.append((frequency, G_storage, G_loss, gamma2, fi, amp_quotient))
    # Organize and unpack the results for the different segments
    results = sorted(results, key=lambda x: int(x[0]))
    frequencies_results = [x[0] for x in results]
    G_storage_results = [x[1] for x in results]
    G_loss_results = [x[2] for x in results]
    gamma2_results = [x[3] for x in results]
    fi_results = [x[4] for x in results]
    amp_quotient_results = [x[5] for x in results]
    return (frequencies_results, G_storage_results, G_loss_results, gamma2_results, fi_results,  amp_quotient_results, bcoef, wc)