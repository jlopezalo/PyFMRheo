from lmfit import Model, Parameters

from .models.ting_analytical import ting_analytical_cone

def TingAnaliticalFit(
    force, time, ind_shape, tip_parameter,
    p0, t0, v0r, v0t, poisson_ratio=0.5
    ):

    params = Parameters()

    # Define varying parameters for the hertz fit
    params.add('beta', value=p0[0], min=0)
    params.add('tmax', value=p0[1], vary=False)
    params.add('E0', value=p0[2], min=0)
    params.add('slope', value=p0[3])
    params.add('f0', value=p0[4])

    fixed_params = {
        't0': t0, 'v0r': v0r, 'v0t': v0t,
        'ind_shape': ind_shape, 'half_angle': tip_parameter,
        'poisson_ratio':poisson_ratio
    }

    functing = Model(lambda time, beta, E0, slope, f0, tmax: ting_analytical_cone(time, beta, E0, slope, f0, tmax, **fixed_params))

    print(f'Ting Analytical parameter names: {functing.param_names}')
    print(f'Ting Analytical independent variables: {functing.independent_vars}')

    return functing.fit(force, params, time=time)