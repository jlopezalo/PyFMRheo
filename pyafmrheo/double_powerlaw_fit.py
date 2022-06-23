from lmfit import Model, Parameters
import numpy as np

from .models.old.rheology import pwl_damping_model

def DoublePWLFit(frequencies, G, w0, p0):

    # Prepare data for complex values fit
    freq_fit = np.r_[frequencies, frequencies]
    G_fit = np.r_[G.real, G.imag]
    split_indx = len(frequencies)

    params = Parameters()

    # Define varying parameters for the hertz fit
    params.add('A', value=p0[0], min=0)
    params.add('B', value=p0[1], min=0)
    params.add('alpha', value=p0[2], min=0)
    params.add('beta', value=p0[3], min=0)

    fixed_params = {'w0': w0, 'split_indx': split_indx}

    funcpwl = Model(lambda freq, A, B, alpha, beta: pwl_damping_model(freq, A, B, alpha, beta, **fixed_params))

    print(f'Dounle Powerlaw parameter names: {funcpwl.param_names}')
    print(f'Dounle Powerlaw independent variables: {funcpwl.independent_vars}')

    return funcpwl.fit(G_fit, params, freq=freq_fit)