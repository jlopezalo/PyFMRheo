import numpy as np
from lmfit import Model, Parameters

from .geom_coeffs import get_coeff

class DMTModel:
    # Model abstract class
    def __init__(self, ind_geom, tip_param) -> None:
        # Tip geomtry params
        self.ind_geom = ind_geom         # No units
        self.tip_parameter = tip_param   # If radius units is meters, If half angle units is degrees
        # Model params #####################
        self.n_params = None
        # Adheshion force
        self.adhesion_force = 0.0
        # Contact point
        self.delta0 = 0.0
        # Apparent Young's Modulus
        self.E0 = 1000
        self.E0_init = 1000
        self.E0_min = 0
        self.E0_max = np.inf
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
    
    def build_params(self):
        params = Parameters()
        params.add('E0', value=self.E0_init, min=self.E0_min, max=self.E0_max)
        return params

    def model(self, indentation, E0, delta0):
        # Define output array
        force = np.zeros(indentation.shape)
        # Find the index where indentation is 0
        idx = (np.abs(indentation - delta0)).argmin()
        # Get the value of the contact point
        delta0 = indentation[idx]
        # Get indenter shape coefficient and exponent
        coeff, n = get_coeff(self.ind_geom, self.tip_parameter, self.poisson_ratio)
        # Compute the force using hertz model
        for i in range(len(force)):
            if indentation[i] < delta0:
                # Assign 0.0 N as force value
                force[i] = 0.0
            else:
                # Fit Hertz model on the contact part
                # F = FHertz + FAdhesion
                force[i] = coeff * E0 * np.power((indentation[i] - delta0), n) + self.adhesion_force
        return force

    def fit(self, indentation, force):
        coeff, n = get_coeff(self.ind_geom, self.tip_parameter, self.poisson_ratio)
        self.E0_init = np.max(force) / coeff / np.max(indentation) ** n
        # Param order:
        # E0, delta0
        DMTmodel =\
            lambda indentation, E0: self.model(indentation, E0, self.delta0)
        
        DMTmodelfit = Model(DMTmodel)
        
        # Define free params
        params = self.build_params()
        
        # Do fit
        self.n_params = len(DMTmodelfit.param_names)
        result_hertz = DMTmodelfit.fit(force, params, indentation=indentation)

        # Assign fit results to model params
        self.E0 = result_hertz.best_values['E0']
        
        modelPredictions = self.eval(indentation)

        absError = modelPredictions - force

        self.MAE = np.mean(absError) # mean absolute error
        self.SE = np.square(absError) # squared errors
        self.MSE = np.mean(self.SE) # mean squared errors
        self.RMSE = np.sqrt(self.MSE) # Root Mean Squared Error, RMSE
        self.Rsquared = 1.0 - (np.var(absError) / np.var(force))
        
        # Get goodness of fit params
        self.chisq = self.get_chisq(indentation, force)
        self.redchi = self.get_red_chisq(indentation, force)

    def eval(self, indentation):
        return self.model(indentation, self.delta0, self.E0)
    
    def get_residuals(self, indentation, force):
        return force - self.eval(indentation)

    def get_chisq(self, indentation, force):
        a = (self.get_residuals(indentation, force)**2/force)
        return np.sum(a[np.isfinite(a)])
    
    def get_red_chisq(self, indentation, force):
        return self.get_chisq(indentation, force) / self.n_params
    
    def fit_report(self):
        print(f"""
        # Fit parameters
        Indenter shape: {self.ind_geom}\n
        Tip paraneter: {self.tip_parameter}\n
        Adhesion force: {self.adhesion_force}\n
        delta0: {self.delta0}\n
        Number of free parameters: {self.n_params}\n
        E0: {self.E0}\n
        # Fit metrics
        MAE: {self.MAE}\n
        MSE: {self.MSE}\n
        RMSE: {self.RMSE}\n
        Rsq: {self.Rsquared}\n
        Chisq: {self.chisq}\n
        RedChisq: {self.redchi}\n
        """
        )