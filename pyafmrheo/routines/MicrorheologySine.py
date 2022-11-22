# Get data analysis tools
import numpy as np
from ..utils.force_curves import get_poc_RoV_method, get_poc_regulaFalsi_method
from ..utils.signal_processing import detrend_rolling_average
from .HertzFit import doHertzFit
from ..models.sine import SineWave
from ..models.rheology import ComputeComplexModulusSine

def doMicrorheologySine(fdc, param_dict):
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
    # Assume d0 as 0, since we are in contact.
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
        # Detrend input signals using the rolling average method
        zheight, deflection, time =\
            detrend_rolling_average(frequency, zheight, deflection, time, 'zheight', 'deflection', [])
        # Get indentation asuming d0 = 0
        indentation = zheight - deflection
        # Get angular frequency
        omega = 2.*np.pi*frequency
        # Fit Sine to indentation wave
        ind_sine_wave = SineWave(omega)
        ind_sine_wave.amplitude_init = np.std(indentation) * 2.**0.5
        ind_sine_wave.phase_init = 0.
        ind_sine_wave.offset_init = np.mean(indentation)
        ind_sine_wave.fit(time, indentation)
        # Fit Sine to deflection wave
        delf_sine_wave = SineWave(omega)
        delf_sine_wave.amplitude_init = np.std(deflection) * 2.**0.5
        delf_sine_wave.phase_init = 0.
        delf_sine_wave.offset_init = np.mean(deflection)
        delf_sine_wave.fit(time, deflection)
        # Get Amplitude results from fit
        A_ind = ind_sine_wave.amplitude
        A_defl = delf_sine_wave.amplitude
        # Get Phase results from fit
        Phi_ind = ind_sine_wave.phase
        Phi_defl = delf_sine_wave.phase
        # If the amplitude is negative make it positive
        # before computing G'and G"
        # The angle is fliped too
        if A_ind < 0:
            A_ind = -A_ind
            Phi_ind += np.pi
        if A_defl < 0:
            A_defl = -A_defl
            Phi_defl += np.pi
        # Compute delta Phi
        dPhi = Phi_defl - Phi_ind
        # Get G* using the amplitude and phase from the fit
        G = ComputeComplexModulusSine(
            A_defl, A_ind, wc, dPhi, frequency, param_dict['contact_model'],
            param_dict['tip_param'], param_dict['k'], fi=fi, amp_quotient=amp_quotient,
            bcoef=bcoef, poisson_ratio=param_dict['poisson']
        )
        # Append results of each segment
        results.append((frequency, G.real, G.imag, ind_sine_wave, delf_sine_wave, fi, amp_quotient))
    # Organize and unpack the results for the different segments
    results = sorted(results, key=lambda x: int(x[0]))
    frequencies_results = [x[0] for x in results]
    G_storage_results = [x[1] for x in results]
    G_loss_results = [x[2] for x in results]
    ind_sinfit_results = [x[3] for x in results]
    defl_sinfit_results = [x[4] for x in results]
    fi_results = [x[5] for x in results]
    amp_quotient_results = [x[6] for x in results]
    return (frequencies_results, G_storage_results, G_loss_results, ind_sinfit_results, defl_sinfit_results, fi_results, amp_quotient_results, bcoef, wc)