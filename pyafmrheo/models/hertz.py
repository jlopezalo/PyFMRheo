import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
import warnings

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
        # Sample height
        self.sample_height = None
        # Goodness of fit metrics
        self.MAE = None
        self.SE = None
        self.MSE = None
        self.RMSE = None
        self.Rsquared = None
        self.chisq = None
        self.redchi = None
    
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
        # If sample height is given, assign sample height
        self.sample_height = sample_height
        # Define initial guess for E0
        coeff, n = get_coeff(self.ind_geom, self.tip_parameter, self.poisson_ratio)
        self.E0_init = coeff * (np.max(force) / np.power(np.max(indentation), n))
        # Param order:
        # delta0, E0, f0, slope
        if self.fit_hline_flag:
            p0 = [self.delta0_init, self.E0_init, self.f0_init, self.slope_init]
            bounds = [
                [self.delta0_min, self.E0_min, self.f0_min, self.slope_min],
                [self.delta0_max, self.E0_max, self.f0_max, self.slope_max]
            ]
            hertzmodel =\
             lambda indentation, delta0, E0, f0, slope: self.objective(indentation, delta0, E0, f0, slope, self.sample_height)
        else:
            p0 = [self.delta0_init, self.E0_init, self.f0_init]
            bounds = [
                [self.delta0_min, self.E0_min, self.f0_min],
                [self.delta0_max, self.E0_max, self.f0_max]
            ]
            hertzmodel =\
             lambda indentation, delta0, E0, f0: self.objective(indentation, delta0, E0, f0, self.slope, self.sample_height)
        
        # Do fit
        self.n_params = len(p0)
        res, _ = curve_fit(hertzmodel, indentation, force, p0, bounds=bounds)

        # Assign fit results to model params
        self.delta0 = res[0]
        self.E0 = res[1]
        self.f0 = res[2]
        if self.fit_hline_flag:
            self.slope = res[3]
        
        modelPredictions = self.eval(indentation, sample_height)

        absError = modelPredictions - force

        self.MAE = np.mean(absError) # mean absolute error
        self.SE = np.square(absError) # squared errors
        self.MSE = np.mean(self.SE) # mean squared errors
        self.RMSE = np.sqrt(self.MSE) # Root Mean Squared Error, RMSE
        self.Rsquared = 1.0 - (np.var(absError) / np.var(force))
        
        # Get goodness of fit params
        self.chisq = self.get_chisq(indentation, force, sample_height)
        self.redchi = self.get_red_chisq(indentation, force, sample_height)

    def eval(self, indentation, sample_height=None):
        return self.objective(indentation, self.delta0, self.E0, self.f0, self.slope, sample_height)
    
    def get_residuals(self, indentation, force,  sample_height=None):
        return force - self.eval(indentation, sample_height)

    def get_chisq(self, indentation, force, sample_height=None):
        return np.sum((self.get_residuals(indentation, force, sample_height)**2/force)) 
    
    def get_red_chisq(self, indentation, force, sample_height=None):
        return self.get_chisq(indentation, force, sample_height) / self.n_params
    
    def fit_report(self):
        print(f"""
        # Fit parameters
        Indenter shape: {self.ind_geom}\n
        Tip paraneter: {self.tip_parameter}\n
        BEC Model: {self.bec_model}\n
        Number of free parameters: {self.n_params}\n
        delta0: {self.delta0}\n
        E0: {self.E0}\n
        f0: {self.f0}\n
        slope: {self.slope}\n
        # Fit metrics
        MAE: {self.MAE}\n
        SE: {self.SE}\n
        MSE: {self.MSE}\n
        RMSE: {self.RMSE}\n
        Rsq: {self.Rsquared}\n
        Chisq: {self.chisq}\n
        RedChisq: {self.redchi}\n
        """
        )