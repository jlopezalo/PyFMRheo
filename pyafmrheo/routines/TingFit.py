import numpy as np

from ..utils.force_curves import get_poc_RoV_method, get_poc_regulaFalsi_method, correct_viscous_drag, correct_tilt, correct_offset
from .HertzFit import doHertzFit
from ..models.ting import TingModel

def doTingFit(fdc, param_dict):
    # Get data from the first extend segments and last retract segment
    ext_data = fdc.extend_segments[0][1]
    ret_data = fdc.retract_segments[-1][1]
    # Perform tilt correction
    height = np.r_[ext_data.zheight, ret_data.zheight]
    deflection = np.r_[ext_data.vdeflection, ret_data.vdeflection]
    idx = len(ext_data.zheight)
    if param_dict['offset_type'] == 'percentage':
        deltaz = height.max() - height.min()
        maxoffset = height.min() + deltaz * param_dict['max_offset']
        minoffset = height.min() + deltaz * param_dict['min_offset']
    else:
        maxoffset = param_dict['max_offset']
        minoffset = param_dict['min_offset']
    
    if param_dict['correct_tilt']:
        corr_defl = correct_tilt(height, deflection, maxoffset, minoffset)
    else:
        corr_defl = correct_offset(height, deflection, maxoffset, minoffset)
    ext_data.vdeflection = corr_defl[:idx]
    ret_data.vdeflection = corr_defl[idx:]
    # Get initial estimate of PoC
    if param_dict['poc_method'] == 'RoV':
        comp_PoC = get_poc_RoV_method(
            ext_data.zheight, ext_data.vdeflection, param_dict['poc_win'])
    else:
        comp_PoC = get_poc_regulaFalsi_method(
            ext_data.zheight, ext_data.vdeflection, param_dict['sigma'])
    poc = [comp_PoC[0], 0]
    # Perform HertzFit to obtain refined posiiton of PoC
    hertz_result = doHertzFit(fdc, param_dict)
    hertz_d0 = hertz_result.delta0
    hertz_E0 = hertz_result.E0
    # Shift PoC using d0 obtained in HertzFit
    poc[0] += hertz_d0
    poc[1] = 0
    # Compute force and indentation with new PoC
    fdc.get_force_vs_indentation(poc, param_dict['k'])
    # Get force, indentation and time from the extend and retract segments
    ext_indentation = ext_data.indentation
    ext_force = ext_data.force
    ext_time = ext_data.time
    ret_indentation = ret_data.indentation
    ret_force = ret_data.force
    ret_time = ret_data.time
    # Check time offset
    # If the time offset is 2 times larger than
    # the deltat between points, add the time offset to 
    # the retract segment. This is important to ensure continuity
    # in the time vector and so the Ting model can be fitted
    # properly to the data
    t_offset = np.abs(ext_data.zheight[-1] - ret_data.zheight[0]) / (ext_data.velocity * -1e-9)
    dt = np.abs(ext_data.time[1] - ext_data.time[0])
    if t_offset > 2*dt:
        ret_time = ret_time + t_offset
    # Correct for viscous drag by fitting a line on the
    # extend and retract baselines. This is an alternative
    # to correcting using the vdrag value
    if param_dict['vdragcorr']:
        ext_force, ret_force = correct_viscous_drag(
            ext_indentation, ext_force, ret_indentation, ret_force,
            poly_order=param_dict['polyordr'], speed=param_dict['rampspeed']
        )
    # The used may decide to compute the velocities of 
    # approach and retract withing the model by fitting a 
    # straight line, where the slope will be the velocity.
    # By default, the speed is determined using parameters
    # from the file header
    if not param_dict['compute_v_flag']:
        v0t = np.abs(ext_data.zheight.min() - ext_data.zheight.max())/ext_data.segment_metadata['duration']
        v0r = np.abs(ret_data.zheight.min() - ret_data.zheight.max())/ret_data.segment_metadata['duration']
    else:
        v0t, v0r = None, None
    # Prepare data for TingFit and compute initial
    # values for tc and tm
    idx_tc = (np.abs(ext_indentation - 0)).argmin()
    t0 = ext_time[-1]
    indentation = np.r_[ext_indentation, ret_indentation]
    time = np.r_[ext_time, ret_time + t0]
    force = np.r_[ext_force, ret_force]
    fit_mask = indentation > (-1 * param_dict['contact_offset'])
    tc = time[idx_tc]
    ind_fit = indentation[fit_mask]
    force_fit = force[fit_mask]
    force_fit = force_fit - force_fit[0]
    time_fit = time[fit_mask]
    tc_fit = tc-time_fit[0]
    time_fit = time_fit - time_fit[0] - tc_fit
    tc_fit = 0.0
    # Compute downfactor
    downfactor= len(time_fit) // param_dict['pts_downsample']
    # Get indices to downsample signal
    idxDown = list(range(0, len(time_fit), downfactor))
    # Compute tm and F0 using the downsampled signal
    idx_tm = np.argmax(force_fit[idxDown])
    f0idx = np.where(time_fit==0)[0]
    F0_init=force_fit[f0idx]-param_dict['vdrag']*v0t
    # Compute bounds for tc and F0
    tc_max = tc_fit+downfactor/(1/(time_fit[1]-time_fit[0]))*10
    tc_min = tc_fit-downfactor/(1/(time_fit[1]-time_fit[0]))*10
    f0_max = F0_init+100e-12
    f0_min = F0_init-100e-12
    # Set params for betaE
    if param_dict['auto_init_betaE']:
        betaE_init = 0.05 if hertz_E0 > 10e3 else 0.25
    else:
        betaE_init = param_dict['fluid_exp']
    # In case the model is paraboloid, we force the bounds 0.01 and 0.49 to
    # evade the singularity of the hypergeometric function at betaE = 0.5
    if param_dict['contact_model'] == 'paraboloid':
        betaE_min, betaE_max = 0.01, 0.49
    else:
        betaE_min, betaE_max = 0.01, 0.99
    # Build Ting model
    ting_model = TingModel(param_dict['contact_model'], param_dict['tip_param'], param_dict['model_type'])
    # Assign params initial values and bounds
    # E0
    ting_model.E0_init = hertz_E0
    ting_model.E0_min = hertz_E0/1000
    ting_model.E0_max = np.inf
    # tc
    ting_model.tc_init = tc_fit
    ting_model.tc_min = tc_min
    ting_model.tc_max = tc_max
    # betaE
    ting_model.betaE_init = betaE_init
    ting_model.betaE_min = betaE_min
    ting_model.betaE_max = betaE_max
    # F0
    ting_model.F0_init = F0_init[0]
    ting_model.F0_min = f0_min[0]
    ting_model.F0_max = f0_max[0]
    # vdrag
    ting_model.vdrag = param_dict['vdrag']

    # Do fit
    ting_model.fit(
        time_fit[idxDown], force_fit[idxDown], ind_fit[idxDown],
        t0=param_dict['t0'], idx_tm=idx_tm, smooth_w=param_dict['smoothing_win'],
        v0t=v0t, v0r=v0r
    )

    # Return the results of the TingFit and HertzFit
    return ting_model, hertz_result