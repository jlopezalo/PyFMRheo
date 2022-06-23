import numpy as np

def get_coeff(ind_geom,  tip_parameter, poisson_ratio):
    if ind_geom == 'paraboloid':
        coeff = 4/3 * np.sqrt(tip_parameter) * 1/(1-poisson_ratio**2)
        n = 3 / 2
    elif ind_geom == 'pyramid':
        coeff = 1/np.sqrt(2) * np.tan(np.radians(tip_parameter)) * 1/(1-poisson_ratio**2)
        n = 2
    elif ind_geom == 'blunted_pyramid':
        coeff = 1/np.sqrt(2) * np.tan(np.radians(tip_parameter)) * 1/(1-poisson_ratio**2)
        n = 2
    elif ind_geom == 'cone':
        coeff = 2/np.pi * np.tan(np.radians(tip_parameter)) * 1/(1-poisson_ratio**2)
        n = 2
    elif ind_geom == 'flat_punch':
        coeff = 2 * tip_parameter * 1/(1-poisson_ratio**2)
        n = 2
    else:
        raise Exception('Non valid indenter geometry')
    return coeff, n