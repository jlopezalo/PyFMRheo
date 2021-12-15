from lmfit import Model, Parameters

from .models.ting_numerical import ting_numerical

def TingNumericalFit(
    indentation, force, time, ind_shape, tip_parameter,
    p0, dT, dTp, smoothing_window, relaxation_function, 
    poisson_ratio=0.5
    ):

    params = Parameters()

    # Normalization to improve the fit quality
    NF = (force.max()-force.min())/10

    # Define varying parameters for the hertz fit
    params.add('d0', value=p0[0])
    params.add('f0', value=p0[1])
    params.add('slope', value=p0[2])
    params.add('E0', value=p0[3]/NF, min=0)
    params.add('alpha', value=p0[4], min=0, max=1)

    fixed_params = {
        'dT': dT,'dTp':dTp, 'time': time,
        'smoothing_window': smoothing_window, 'relaxation_function': relaxation_function,
        'indenter_shape': ind_shape, 'tip_parameter': tip_parameter,
        'poisson_ratio': poisson_ratio
    }

    functing = Model(lambda indentation, d0, f0, slope, E0, alpha: ting_numerical(indentation, d0, f0, slope, E0, alpha, **fixed_params))

    print(f'Ting Numerical parameter names: {functing.param_names}')
    print(f'Ting Numerical independent variables: {functing.independent_vars}')

    return functing.fit(force, params, indentation=indentation)