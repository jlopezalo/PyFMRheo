from lmfit import Model, Parameters

from .models.rheology import drag_sphere_model

def ViscousDragFit(distance, Bh, eta, p0):
    
    params = Parameters()

    params.add('a_eff', value=p0[0], min=0)
    params.add('h_eff', value=p0[1], min=0)

    fixed_params = {'dynamic_visc': eta}

    funcvds = Model(lambda distance, a_eff, h_eff: drag_sphere_model(distance, a_eff, h_eff, **fixed_params))

    print(f'Viscous Drag Sphere parameter names: {funcvds.param_names}')
    print(f'Viscous Drag Sphere independent variables: {funcvds.independent_vars}')

    return funcvds.fit(Bh, params, distance=distance)