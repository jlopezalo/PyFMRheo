import numpy as np

from ..utils.force_curves import correct_tilt, correct_offset, find_nearest
from ..models.dmt import DMTModel

def doDMTFit(fdc, param_dict):
    # Get segment data
    if param_dict['curve_seg'] == 'extend':
        raise Exception("DMT can only be fit to the retract segment of the curve!")
    else:
        segment_data = fdc.retract_segments[-1][1]
        segment_data.zheight = segment_data.zheight[::-1]
        segment_data.vdeflection = segment_data.vdeflection[::-1]
    # Perform tilt correction
    if param_dict['offset_type'] == 'percentage':
        deltaz = segment_data.zheight.max() - segment_data.zheight.min()
        maxoffset = segment_data.zheight.min() + deltaz * param_dict['max_offset']
        minoffset = segment_data.zheight.min() + deltaz * param_dict['min_offset']
    else:
        maxoffset = param_dict['max_offset']
        minoffset = param_dict['min_offset']

    if param_dict['correct_tilt']:
        segment_data.vdeflection =\
            correct_tilt(
                segment_data.zheight, segment_data.vdeflection, maxoffset, minoffset
            )
    elif param_dict['correct_offset']:
        segment_data.vdeflection =\
            correct_offset(
                segment_data.zheight, segment_data.vdeflection, maxoffset, minoffset
            )
    # Downsample signal
    if param_dict['downsample_flag']:
        downfactor= len(segment_data.zheight) // param_dict['pts_downsample']
        idxDown = list(range(0, len(segment_data.zheight), downfactor))
        segment_data.zheight = segment_data.zheight[idxDown]
        segment_data.vdeflection = segment_data.vdeflection[idxDown]
    # Get initial estimate of PoC
    segment_data.force =  np.array(segment_data.vdeflection * param_dict['k'])
    force = segment_data.force - force[0]
    if param_dict['adhesionForce'] is None:
        adhesionForce = np.min(force)
    else:
        adhesionForce = param_dict['adhesionForce']
    tipPosition = np.array(segment_data.zheight - segment_data.vdeflection)
    comp_PoC_idx = find_nearest(force, adhesionForce)
    comp_PoC = tipPosition[comp_PoC_idx]
    # Prepare data for the fit
    indentation = tipPosition - comp_PoC
    contact_mask = force >= adhesionForce
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
    # Perform fit
    dmt_model = DMTModel(param_dict['contact_model'], param_dict['tip_param'])
    dmt_model.delta0 = comp_PoC[0]
    dmt_model.adhesion_force = adhesionForce
    if not param_dict['auto_init_E0']:
        dmt_model.E0_init = param_dict['E0']
    if param_dict.get('fit_method', None) is not None:
        dmt_model.fit_method = param_dict['fit_method']
    dmt_model.fit(indentation, force)
    # Return fitted model object
    return dmt_model