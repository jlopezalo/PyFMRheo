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
    

    # function for genetic algorithm to minimize (sum of squared error)
    def sumOfSquaredError(self, parameterTuple, indentation, force, hertzmodel):
        warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
        val = hertzmodel(indentation, *parameterTuple)
        return np.sum((force - val) ** 2.0)
    
    def generate_Initial_Parameters(self, indentation, force, hertzmodel):
        parameterBounds = [[np.min(indentation), np.max(indentation)], [0, 1e12], [np.min(force), np.max(force)]]
        if self.fit_hline_flag:
            parameterBounds.append([0, 1]) # search bounds for slope

        # "seed" the numpy random number generator for repeatable results
        result = differential_evolution(self.sumOfSquaredError, parameterBounds, args=(indentation, force, hertzmodel), seed=3)
        return result.x

    def fit(self, indentation, force, sample_height=None):
        # Use log to make params scale more equal during fit?
        # Param order:
        # delta0, E0, f0, slope
        self.sample_height = sample_height
        if self.fit_hline_flag:
            hertzmodel =\
             lambda indentation, delta0, E0, f0, slope: self.objective(indentation, delta0, E0, f0, slope, self.sample_height)
        else:
            hertzmodel =\
             lambda indentation, delta0, E0, f0: self.objective(indentation, delta0, E0, f0, self.slope, self.sample_height)
        
        # Do fit
        p0 = self.generate_Initial_Parameters(indentation, force, hertzmodel)
        self.n_params = len(p0)
        res, _ = curve_fit(hertzmodel, indentation, force, p0)

        # Assign fit results to model params
        self.delta0 = res[0]
        self.E0 = res[1]
        self.f0 = res[2]
        if self.fit_hline_flag:
            self.slope = res[3]
        
        modelPredictions = self.eval(indentation, sample_height)

        absError = modelPredictions - force

        MAE = np.mean(absError) # mean absolute error
        SE = np.square(absError) # squared errors
        MSE = np.mean(SE) # mean squared errors
        RMSE = np.sqrt(MSE) # Root Mean Squared Error, RMSE
        Rsquared = 1.0 - (np.var(absError) / np.var(force))
        
        # Get goodness of fit params
        redchi = self.get_red_chisq(indentation, force)

        return res, redchi

    def eval(self, indentation, sample_height=None):
        return self.objective(indentation, self.delta0, self.E0, self.f0, self.slope, sample_height)

    def get_chisq(self, x, y, sample_height=None):
        return np.sum(((y - self.eval(x, sample_height))/np.std(y))**2)
    
    def get_red_chisq(self, x, y, sample_height=None):
        return self.get_chisq(x, y, sample_height) / self.n_params