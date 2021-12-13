# Imports
import numpy as np
from .bec import (
    bec_dimitriadis_paraboloid_bonded, bec_dimitriadis_paraboloid_not_bonded,
    bec_gavara_cone, bec_managuli_cone, bec_garcia_garcia
)

# Paraboloid params
paraboloid_coeff = lambda radius, poisson_ratio: 4/3 * np.sqrt(radius) * 1/(1-poisson_ratio**2)
paraboloid_n = 3 / 2

# Pyramid params
pyramidal_coeff = lambda half_angle, poisson_ratio: 1/np.sqrt(2) * np.tan(np.radians(half_angle)) * 1/(1-poisson_ratio**2)
pyramidal_n = 2

# Blunted pyramid params
blunted_pyramidal_coeff = lambda half_angle, poisson_ratio: 3/4 * np.tan(np.radians(half_angle)) * 1/(1-poisson_ratio**(1/2))
blunted_pyramidal_n = 2

# Cone geometry params
conical_coeff = lambda half_angle, poisson_ratio: 2/np.pi * np.tan(np.radians(half_angle)) * 1/(1-poisson_ratio**2)
conical_n = 2

# Flat punch geometry params
flat_punch_coeff = lambda radius, poisson_ratio: 2 * radius * 1/(1-poisson_ratio**2)
flat_punch_n = 2

# Hertz model params key
hertz_model_params = {
    "paraboloid": (paraboloid_coeff, paraboloid_n),
    "pyramid": (pyramidal_coeff, pyramidal_n),
    "blunted_pyramid": (blunted_pyramidal_coeff, blunted_pyramidal_n),
    "cone": (conical_coeff, conical_n),
    "flat_punch": (flat_punch_coeff, flat_punch_n)
}

# Bottom effect models params key
bec_models_key = {
    "dimitriadis_paraboloid_bonded": bec_dimitriadis_paraboloid_bonded,
    "dimitriadis_paraboloid_not_bonded": bec_dimitriadis_paraboloid_not_bonded,
    "gavara_cone": bec_gavara_cone,
    "managuli_cone": bec_managuli_cone,
    "garcia_garcia": bec_garcia_garcia
}

# Hertz model function to minimize
def hertz_model(
    indentation, delta0, E0, f0, slope, poisson_ratio,
    indenter_shape, tip_parameter, sample_height, bec_model
):

    # Define output array
    force = np.zeros(indentation.shape)

    # Find the index where indentation is 0
    idx = (np.abs(indentation - delta0)).argmin()

    # Get the value of the contact point
    delta0 = indentation[idx]

    # Get coefficient function and exponent
    coeff_func, n = hertz_model_params[indenter_shape]
    coeff = coeff_func(tip_parameter, poisson_ratio)

    # Get bottom effect correction coefficients
    if bec_model and sample_height:
        bec_model_func = bec_models_key[bec_model]
        bec_coeffs = bec_model_func(sample_height, (indentation - delta0), indenter_shape, tip_parameter)
    else:
        bec_coeffs = np.ones(indentation.shape)
    
    # Compute the force using hertz model
    for i in range(len(force)):
        if indentation[i] < delta0:
            # Fit a line on the non contact part
            force[i] = (indentation[i] - delta0) * slope + f0
        else:
            # Fit Hertz model on the contact part
            # F = F0 * BEC_Correction
            force[i] = coeff * bec_coeffs[i] * E0 * np.power((indentation[i] - delta0), n) + f0
    
    return force
