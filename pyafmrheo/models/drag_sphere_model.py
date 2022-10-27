from ast import Mod
import numpy as np
from lmfit import Model, Parameters

class DragSphereModel:
    def __init__(self) -> None:
        # Effecive area of the cantilever
        self.a_eff = None
        self.a_eff_init = 1
        self.a_eff_min = 0
        self.a_eff_max = np.inf
        # Effective height of the cantilever
        self.h_eff = None
        self.h_eff_init = 1
        self.h_eff_min = 0
        self.h_eff_max = np.inf
        # Dynamic viscosity
        self.dynamic_visc = None
    
    def build_params(self):
        params = Parameters()
        params.add('a_eff', value=self.a_eff_init, min=self.a_eff_min, max=self.a_eff_max)
        params.add('h_eff', value=self.h_eff_init, min=self.h_eff_min, max=self.h_eff_max)
        return params

    def model(self, distance, a_eff, h_eff, dynamic_visc):
        # Model for computing viscous drag factor
        # Reference: https://pubs.acs.org/doi/10.1021/la0110850
        # b(h)) = (6 * pi * dynamic_visc * aeffˆ2)/(h+heff)
        return (6 * np.pi * dynamic_visc * a_eff ** 2) / (distance + h_eff)

    def fit(self, distance, Bh, dynamic_visc):

        self.dynamic_visc = dynamic_visc

        drag_sphere_model =\
            lambda distance, a_eff, h_eff: self.model(distance, a_eff, h_eff, self.dynamic_visc)

        drag_sphere_model_fit = Model(drag_sphere_model)

        # Define free params
        params = self.build_params()
        
        # Do fit
        result_drag_sphere = drag_sphere_model_fit.fit(Bh, params, distance=distance)
        
        # Assign fit results to model params
        self.a_eff = result_drag_sphere.best_values['a_eff']
        self.h_eff = result_drag_sphere.best_values['h_eff']
        
        # Compute metrics
        modelPredictions = self.eval(distance, dynamic_visc)

        absError = modelPredictions - Bh

        self.MAE = np.mean(absError) # mean absolute error
        self.SE = np.square(absError) # squared errors
        self.MSE = np.mean(self.SE) # mean squared errors
        self.RMSE = np.sqrt(self.MSE) # Root Mean Squared Error, RMSE
        self.Rsquared = 1.0 - (np.var(absError) / np.var(Bh))

        # Get goodness of fit params
        self.chisq = self.get_chisq(distance, dynamic_visc)
        self.redchi = self.get_red_chisq(distance, dynamic_visc)

    def eval(self, distance, dynamic_visc):
        return self.model(distance, self.a_eff, self.h_eff, dynamic_visc)
    
    def get_chisq(self, x, y, dynamic_visc):
        return np.sum(((y - self.eval(x, dynamic_visc))/np.std(y))**2)
    
    def get_red_chisq(self, x, y, dynamic_visc):
        return self.get_chisq(x, y, dynamic_visc) / self.n_params

    
        

        