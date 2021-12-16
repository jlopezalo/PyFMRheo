import numpy as np

from ..utils.signal_processing import smoothM
from .hertz import hertz_model_params


def simple_power_law(time, E0, dTp, alpha):
    return E0 * (time/dTp) ** -alpha


def ting_numerical(
    indentation, d0, f0, slope, E0, alpha, dT, smoothing_window,
    dTp, time, relaxation_function, indenter_shape, tip_parameter, poisson_ratio
):

    # Get coefficient function and exponent
    coeff_func, n = hertz_model_params[indenter_shape]
    coeff = coeff_func(tip_parameter, poisson_ratio)

    # Output variable
    y_out = np.zeros(indentation.shape)

    contact_mask = indentation > d0

    # Get time data for contact
    ting_time = time[contact_mask]
    ting_time = ting_time - ting_time[0]

    # Get contact point
    d0 = indentation[contact_mask][0]

    # Compute indentation
    fit_ind = indentation - d0

    # Fit the ting model in the contact part
    ting_ind = fit_ind[contact_mask]
    ting_out = np.zeros(ting_ind.shape)

    max_ind_indx = np.argmax(ting_ind)

    # Fit a line y = mx + x0 on the non contact part
    y_out[~contact_mask] = fit_ind[~contact_mask] * slope + f0

    # Get the guesses for Et using the PLR model
    Et = relaxation_function(ting_time, E0, dTp, alpha)

    if np.isinf(Et[0]):
        Et[0] = 2*Et[1]
    if np.isnan(Et[0]):
        Et[0] = Et[1]+(Et[1]-Et[2])

    # Compute indspeed
    ind2speed = np.diff(ting_ind**n)/dT
    ind2speed = np.append(ind2speed, ind2speed[-1])
    ind2speed = smoothM(ind2speed, smoothing_window)

    # Compute ind2speed
    indspeed = np.diff(ting_ind)/dT
    indspeed = np.append(indspeed, indspeed[-1])
    indspeed = smoothM(indspeed, smoothing_window)

    # Integrate in approach segment
    y_app = np.zeros(ting_ind.shape)
    for i in range(max_ind_indx+1):
        y_app[i] = coeff * (np.trapz(Et[i::-1]*ind2speed[:i+1], dx=dT))

    ting_out[:max_ind_indx+1] = y_app[:max_ind_indx+1]

    # Integrate in retract segment
    b = max_ind_indx
    t1_ndx = np.zeros(len(ting_time), dtype=int)
    for i in range(max_ind_indx+1, len(ting_time)):
        res2 = np.zeros(len(ting_time))
        localend = 0

        for j in range(b, localend-1, -1):
            if localend == 0:
                res2[j] = np.trapz(Et[i-j::-1]*indspeed[j:i+1], dx=dT)
                if res2[j] > 0:
                    localend = j

        if abs(res2[localend]) <= abs(res2[localend+1]):
            Imin = localend
        else:
            Imin = localend+1

        if (Imin > max_ind_indx+2):
            t1_ndx[i] = Imin-1

        elif (Imin <= 1):
            t1_ndx[i] = Imin
            t1_ndx[i+1] = 1
            break

        else:
            b = Imin
            t1_ndx[i] = Imin

        ijk = t1_ndx[i]
        if ijk == i:
            ijk = i-1

        ting_out[i] = coeff * \
            (np.trapz(Et[i:i-ijk-1:-1]*ind2speed[:ijk+1], dx=dT))
    
    y_out[contact_mask] = ting_out + f0

    return  y_out