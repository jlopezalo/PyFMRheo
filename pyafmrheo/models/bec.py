# Imports
import numpy as np

# Dimitriadis et al. 2002
# Source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1302067/pdf/11964265.pdf

# Equation (12)
def bec_dimitriadis_paraboloid_bonded(h, indentation, ind_shape, R):
    if ind_shape != "paraboloid":
        raise Exception(f"The Dimitriadis paraboloid bonded BEC model is not suitable for the {ind_shape} geometry.") 
    coefficients_out = []
    for i in range(len(indentation)):
        coeff = 1.0
        if h > 0:
            X = np.sqrt(indentation[i] * R) / h
            o2 = 1.133 * X + 1.283 * X ** 2
            o4 = 0.769 * X ** 3 + 0.0975 * X ** 4
            coeff = 1 + o2 + o4
        coefficients_out.append(coeff)
            
    return coefficients_out

# Equation (11)
def bec_dimitriadis_paraboloid_not_bonded(h, indentation, ind_shape, R):
    if ind_shape != "paraboloid":
        raise Exception(f"The Dimitriadis paraboloid not bonded BEC model is not suitable for the {ind_shape} geometry.") 
    coefficients_out = []
    for i in range(len(indentation)):
        coeff = 1.0
        if h > 0:
            X = np.sqrt(indentation[i] * R) / h
            o2 = 0.884 * X + 0.781 * X ** 2
            o4 = 0.386 * X ** 3 + 0.0048 * X ** 4
            coeff = 1 + o2 + o4
        coefficients_out.append(coeff)
            
    return coefficients_out

# Gavara et al. 2012
# Source: https://www.nature.com/articles/nnano.2012.163

def bec_gavara_cone(h, indentation, ind_shape, half_opening_angle):
    if ind_shape != "cone":
        raise Exception(f"The Gavara cone BEC model is not suitable for the {ind_shape} geometry.") 
    coefficients_out = []
    for i in range(len(indentation)):
        coeff = 1.0
        if h > 0:
            X = indentation[i] / h
            tan_angle = np.tan(half_opening_angle)
            o1 = 1.7795 * (2 * tan_angle / np.pi ** 2) * X
            o2 = 16.0 * (1.7795) ** 2 * tan_angle ** 2 * X ** 2
            coeff = 1 + o1 + o2
        coefficients_out.append(coeff)
            
    return coefficients_out

# Managuli et al. 2018
# Source: https://link.springer.com/article/10.1007/s40799-018-0268-8

# Equation (1)
def bec_managuli_cone(h, indentation, ind_shape, half_opening_angle):
    if ind_shape != "cone":
        raise Exception(f"The Managuli cone BEC model is not suitable for the {ind_shape} geometry.") 
    coefficients_out = []
    for i in range(len(indentation)):
        coeff = 1.0
        if h > 0:
            C = (1.7795 * np.tan(half_opening_angle)) / np.pi ** 2
            o1 = 4 * C * indentation[i] / h
            o2 = 20 * C ** 2 * indentation[i] ** 2 / h ** 2
            coeff = 1 + o1 + o2
        coefficients_out.append(coeff)
            
    return coefficients_out

# Garcia et al. 2018
# Source: https://www.cell.com/biophysj/pdf/S0006-3495(18)30590-3.pdf

paraboloid_model_factors = [
        lambda h, indentation, R : (1.133 * np.sqrt(indentation * R)) / h,                                                     # Order 1
        lambda h, indentation, R : (1.497 * indentation * R )/ h ** 2,                                                         # Order 2
        lambda h, indentation, R : (1.469 * indentation * R * np.sqrt(indentation * R)) / h ** 3,                              # Order 3
        lambda h, indentation, R : (0.755 * indentation ** 2 * R ** 2) / h ** 4                                                # Order 4
    ]

conical_model_factors = [
        lambda h, indentation, half_opening_angle : (0.721 * indentation * np.tan(half_opening_angle)) / h,                   # Order 1
        lambda h, indentation, half_opening_angle : (0.650 * indentation ** 2 * np.tan(half_opening_angle) ** 2) / h ** 2,    # Order 2
        lambda h, indentation, half_opening_angle : (0.491 * indentation ** 3 * np.tan(half_opening_angle) ** 3) / h ** 3,    # Order 3
        lambda h, indentation, half_opening_angle : (0.225 * indentation ** 4 * np.tan(half_opening_angle) ** 4) / h ** 4     # Order 4
    ]

flat_punch_model_factors = [
        lambda h, _, R : (1.133 * R) / h,                                                                                   # Order 1
        lambda h, _, R : (1.283 * R ** 2) / h ** 2,                                                                         # Order 2
        lambda h, _, R : (0.598 * R ** 3) / h ** 3,                                                                         # Order 3
        lambda h, _, R : - (0.291 * R ** 4) / h ** 4                                                                        # Order 4
    ]

garcia_garcia_factors = {
    "paraboloid": paraboloid_model_factors,
    "conical": conical_model_factors,
    "flat_punch": flat_punch_model_factors
}
    
def bec_garcia_garcia(h, indentation, ind_shape, tip_parameter, order=4):
    coefficients_out = []
    model_factors = garcia_garcia_factors.get(ind_shape, None)
    if not model_factors:
        raise Exception(f"The Garcia, Garcia BEC model is not suitable for the {ind_shape} geometry.") 
    for i in range(len(indentation)):
        # The correction factor is computed as:
        # coef = O0 + O1 + On...
        # Order 0 coefficient common in all models
        coeff = 1.0
        if h > 0:
            for j in range(order):
                O = model_factors[j]
                coeff += O(h, indentation[i], tip_parameter)
        coefficients_out.append(coeff)
    return coefficients_out