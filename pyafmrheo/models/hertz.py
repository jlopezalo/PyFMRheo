import numpy as np
from scipy.optimize import curve_fit

from .bec import (
    bec_dimitriadis_paraboloid_bonded, bec_dimitriadis_paraboloid_not_bonded, 
    bec_gavara_cone, bec_managuli_cone, bec_garcia_garcia
)

from .geom_coeffs import get_coeff

class HertzModel:
    # Model abstract class
    def __init__(self, ind_geom, tip_param, bec_model=None) -> None:
        # Tip geomtry params
        self.ind_geom = ind_geom         # No units
        self.tip_parameter = tip_param   # If radius units is meters, If half angle units is degrees
        self.bec_model = bec_model
        # Compiutation params
        self.fit_hline_flag = False
        self.apply_bec_flag = False
        # Model params #####################
        self.n_params = None
        # Contact point
        self.delta0 = 0
        self.delta0_init = 0
        self.delta0_min = -np.inf
        self.delta0_max = np.inf
        # Apparent Young's Modulus
        self.E0 = 1000
        self.E0_init = 1000
        self.E0_min = -np.inf
        self.E0_max = np.inf
        # Contact force
        self.f0 = 0
        self.f0_init = 0
        self.f0_min = -np.inf
        self.f0_max = np.inf
        # Baseline slope
        self.slope = None
        self.slope_init = 0
        self.slope_min = -np.inf
        self.slope_max = np.inf
        # Poisson ratio
        self.poisson_ratio = 0.5
    
    def get_bec_coeffs(self, sample_height, indentation):
        bec_params = [sample_height, (indentation - self.delta0), self.indenter_shape, self.tip_parameter]
        if self.bec_model == 'dimitriadis_paraboloid_bonded':
            return bec_dimitriadis_paraboloid_bonded(*bec_params)
        elif self.bec_model == 'dimitriadis_paraboloid_not_bonded':
            return bec_dimitriadis_paraboloid_not_bonded(*bec_params)
        elif self.bec_model == 'gavara_cone':
            return bec_gavara_cone(*bec_params)
        elif self.bec_model == 'managuli_cone':
            return bec_managuli_cone(*bec_params)
        elif self.bec_model == 'garcia_garcia':
            return bec_garcia_garcia(*bec_params)
        else:
            # TO DO: Implement custom exception
            raise Exception('BEC model not implemented')

    def objective(self, indentation, delta0, E0, f0, slope=None, sample_height=None):
        # Define output array
        force = np.zeros(indentation.shape)
        # Find the index where indentation is 0
        idx = (np.abs(indentation - delta0)).argmin()
        # Get the value of the contact point
        delta0 = indentation[idx]
        # Get indenter shape coefficient and exponent
        coeff, n = get_coeff(self.ind_geom, self.tip_parameter, self.poisson_ratio)
        # Get bottom effect correction coefficients
        if self.bec_model and sample_height:
            bec_coeffs = self.get_bec_coeffs(sample_height, indentation)
        else:
            bec_coeffs = np.ones(indentation.shape)
        # Compute the force using hertz model
        for i in range(len(force)):
            if indentation[i] < delta0:
                if self.fit_hline_flag:
                    # Fit a line on the non contact part
                    force[i] = (indentation[i] - delta0) * slope + f0
                else:
                    # Assign f0 as force value
                    force[i] = f0
            else:
                # Fit Hertz model on the contact part
                # F = F0 * BEC_Correction
                force[i] = coeff * bec_coeffs[i] * E0 * np.power((indentation[i] - delta0), n) + f0
        return force

    def fit(self, indentation, force, sample_height=None):
        # Use log to make params scale more equal during fit?
        # Param order:
        # delta0, E0, f0, slope
        if self.fit_hline_flag:
            p0 = [self.delta0_init, self.E0_init, self.f0_init, self.slope_init]
            self.n_params = len(p0)
            bounds = [
                [self.delta0_min, self.E0_min, self.f0_min, self.slope_min],
                [self.delta0_max, self.E0_max, self.f0_max, self.slope_max]
            ]
            hertzmodel =\
             lambda indentation, delta0, E0, f0, slope: self.objective(indentation, delta0, E0, f0, slope, sample_height)
        else:
            p0 = [self.delta0_init, self.E0_init, self.f0_init]
            self.n_params = len(p0)
            bounds = [
                [self.delta0_min, self.E0_min, self.f0_min],
                [self.delta0_max, self.E0_max, self.f0_max]
            ]
            hertzmodel =\
             lambda indentation, delta0, E0, f0: self.objective(indentation, delta0, E0, f0, self.slope, sample_height)
        
        # Do fit
        res, _ = curve_fit(
            hertzmodel, indentation, force, p0, bounds=bounds,
            method='trf', ftol=1e-08, xtol=1e-08, gtol=1e-08)

        # Assign fit results to model params
        self.delta0 = res[0]
        self.E0 = res[1]
        self.f0 = res[2]
        if self.fit_hline_flag:
            self.slope = res[3]
        
        # Get goodness of fit params
        redchi = self.get_red_chisq(indentation, force)

        return res, redchi

    def eval(self, indentation, sample_height=None):
        return self.objective(indentation, self.delta0, self.E0, self.f0, self.slope, sample_height)

    def get_chisq(self, x, y, sample_height=None):
        return np.sum(((y - self.eval(x, sample_height))/np.std(y))**2)
    
    def get_red_chisq(self, x, y, sample_height=None):
        return self.get_chisq(x, y, sample_height) / self.n_params