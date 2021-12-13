from lmfit import Model, Parameters
import numpy as np

from .models.hertz import hertz_model

def HertzFit(
    approach_indentation, approach_force, ind_shape, tip_parameter,
    p0, poisson_ratio=0.5, sample_height=None, bec_model=None):

    params = Parameters()

    # Define varying parameters for the hertz fit
    params.add('delta0', value=p0[0])
    params.add('f0', value=p0[1])
    params.add('slope', value=p0[2])
    params.add('E0', value=p0[3], min=0)

    fixed_params = {
        'poisson_ratio': poisson_ratio, 'indenter_shape': ind_shape,
        'tip_parameter':tip_parameter, 'sample_height': sample_height,
        'bec_model': bec_model
    
    }

    funchertz = Model(lambda indentation, delta0, E0, f0, slope: hertz_model(indentation, delta0, E0, f0, slope, **fixed_params))

    print(f'Hertz parameter names: {funchertz.param_names}')
    print(f'Hertz independent variables: {funchertz.independent_vars}')

    return funchertz.fit(approach_force, params, indentation=approach_indentation)