from lmfit import Model, Parameters
import numpy as np

from pyafmrheo.models.ting import Ting
from pyafmrheo.models.hertz import hertz_model_params

def TingFit(force, delta, time, model_probe, tip_parameter, p0, modelFt, poisson_ratio=0.5, vdrag=0, smooth_w=1, idx_tm=None):

    params = Parameters()

    # Normalization to improve the fit quality
    NF = (force.max()-force.min())/10

    # Define varying parameters for the hertz fit
    params.add('t0', value=p0[0], vary=False)
    params.add('E0', value=p0[1]/NF, min=0)
    params.add('tc', value=p0[2], min=np.min(time), max=np.max(time))
    params.add('betaE', value=p0[3], min=0, max=1)
    params.add('F0', value=p0[4])

    # Get coefficient function and exponent
    coeff_func, n = hertz_model_params[model_probe]
    coeff = coeff_func(tip_parameter, poisson_ratio)

    idx_tm = np.argmax(force)

    fixed_params = {
        'F': force,
        'delta': delta,
        'model_probe': model_probe,
        'geom_coeff': coeff,
        'geom_exp': n,
        'modelFt': modelFt,
        'vdrag': vdrag,
        'smooth_w': smooth_w,
        'idx_tm': idx_tm,
    }


    functing = Model(lambda time, t0, E0, tc, betaE, F0: Ting(time, t0, E0, tc, betaE, F0, **fixed_params))

    print(f'Ting parameter names: {functing.param_names}')
    print(f'Ting independent variables: {functing.independent_vars}')

    return functing.fit(force, params, time=time)