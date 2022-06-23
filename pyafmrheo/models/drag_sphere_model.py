import numpy as np
from scipy.optimize import curve_fit

class DragSphereModel:
    def __init__(self) -> None:
        # Effecive area of the cantilever
        self.a_eff = None
        self.a_eff_init = 1
        self.a_eff_min = -np.inf
        self.a_eff_max = np.inf
        # Effective height of the cantilever
        self.h_eff = None
        self.h_eff_init = 1
        self.h_eff_min = -np.inf
        self.h_eff_max = np.inf

    def objective(self, distance, a_eff, h_eff, dynamic_visc):
        # Model for computing viscous drag factor
        # Reference: https://pubs.acs.org/doi/10.1021/la0110850
        # b(h)) = (6 * pi * dynamic_visc * aeffˆ2)/(h+heff)
        return (6 * np.pi * dynamic_visc * a_eff ** 2) / (distance + h_eff)

    def fit(self, distance, Bh, dynamic_visc):
        p0 = [self.a_eff_init, self.h_eff_init]
        self.n_params = len(p0)
        bounds = [
                [self.a_eff_min, self.h_eff_min],
                [self.a_eff_max, self.h_eff_max]
            ]
        drag_sphere_model =\
            lambda distance, a_eff, h_eff: self.objective(distance, a_eff, h_eff, dynamic_visc)
        
        # Do fit
        res, _ = curve_fit(
            drag_sphere_model, distance, Bh, p0, bounds,
            method='trf', ftol=1e-08, xtol=1e-08, gtol=1e-08)
        
        # Assign fit results to model params
        self.a_eff = res[0]
        self.h_eff = res[1]
        
        # Get goodness of fit params
        redchi = self.get_red_chisq(distance, Bh)

        return res, redchi

    def eval(self, distance, dynamic_visc):
        return self.objective(distance, self.a_eff, self.h_eff, dynamic_visc)
    
    def get_chisq(self, x, y, dynamic_visc):
        return np.sum(((y - self.eval(x, dynamic_visc))/np.std(y))**2)
    
    def get_red_chisq(self, x, y, dynamic_visc):
        return self.get_chisq(x, y, dynamic_visc) / self.n_params

    
        

        