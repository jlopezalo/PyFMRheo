import numpy as np

import matplotlib.pyplot as plt

from ..utils.force_curves import get_poc_RoV_method, correct_tilt
from ..models.hertz import HertzModel

def doHertzFit(fdc, param_dict):
    # Get segment data
    if param_dict['curve_seg'] == 'extend':
        segment_data = fdc.extend_segments[0][1]
    else:
        segment_data = fdc.retract_segments[-1][1]
        segment_data.zheight = segment_data.zheight[::-1]
        segment_data.vdeflection = segment_data.vdeflection[::-1]
    # Get initial estimate of PoC
    rov_PoC = get_poc_RoV_method(
        segment_data.zheight, segment_data.vdeflection, param_dict['poc_win'])
    poc = [rov_PoC[0], 0]
    # Perform tilt correction
    if param_dict['correct_tilt']:
        segment_data.vdeflection =\
            correct_tilt(
                segment_data.zheight, segment_data.vdeflection, poc[0],
                param_dict['tilt_max_offset'], param_dict['tilt_min_offset']
            )
    # Prepare data for the fit
    segment_data.get_force_vs_indentation(poc, param_dict['k'])
    indentation = segment_data.indentation
    force = segment_data.force
    force = force - force[0]
    contact_mask = indentation >= 0
    ncont_ind = indentation[~contact_mask]
    cont_ind = indentation[contact_mask]
    ncont_force = force[~contact_mask]
    cont_force = force[contact_mask]
    if param_dict['fit_range_type'] == 'indentation':
        mask = (cont_ind >= param_dict['min_ind']) & (cont_ind <= param_dict['max_ind'])
        cont_ind, cont_force = cont_ind[mask], cont_force[mask]
    elif param_dict['fit_range_type'] == 'force':
        mask = (cont_force >= param_dict['min_force']) & (cont_force <= param_dict['max_force'])
        cont_ind, cont_force = cont_ind[mask], cont_force[mask]
    indentation = np.r_[ncont_ind, cont_ind]
    force = np.r_[ncont_force, cont_force]
    plt.plot(indentation, force)
    plt.show()
    # Perform fit
    hertz_model = HertzModel(param_dict['contact_model'], param_dict['tip_param'])
    hertz_model.fit_hline_flag = param_dict['fit_line']
    hertz_model.d0_init = param_dict['d0']
    if not param_dict['auto_init_E0']:
        hertz_model.E0_init = param_dict['E0']
    hertz_model.f0_init = param_dict['f0']
    if param_dict['fit_line']:
        hertz_model.slope_init = param_dict['slope']
    hertz_model.fit(indentation, force)
    # Return fitted model object
    return hertz_model