import numpy as np
from scipy.optimize import curve_fit

class DoublePowerLawModel:
    def __init__(self) -> None:
        # A factor
        self.A = None
        self.A_init = None
        self.A_min = None
        self.A_max = None
        # Alpha
        self.alpha = None
        self.alpha_init = None
        self.alpha_min = None
        self.alpha_max = None
        # B factor
        self.B = None
        self.B_init = None
        self.B_min = None
        self.B_max = None
        # Beta
        self.beta = None
        self.beta_init = None
        self.beta_min = None
        self.beta_max = None

    def objective(self, freq, A, B, alpha, beta, w0, split_indx):
        if alpha > beta: alpha, beta = beta, alpha

        G = np.zeros(freq.shape)

        G[:split_indx] = A * np.cos(np.pi * alpha / 2) * (freq[:split_indx] / w0) ** alpha +\
                        B * np.cos(np.pi * beta / 2) * (freq[:split_indx] / w0) ** beta

        G[split_indx:] = A * np.sin(np.pi * alpha / 2) * (freq[split_indx:] / w0) ** alpha +\
                        B * np.sin(np.pi * beta / 2) * (freq[split_indx:] / w0) ** beta

        return G

    def fit(self, freq, G, w0, split_indx):
        p0 = [self.A_init, self.B_init, self.alpha_init, self.beta_init]
        bounds = [
            [self.A_min, self.B_min, self.alpha_min, self.beta_min]
            [self.A_max, self.B_max, self.alpha_max, self.beta_max]
        ]
        double_pwl_model =\
            lambda freq, A, B, alpha, beta: self.objective(freq, A, B, alpha, beta, w0, split_indx)

        # Do fit
        res, _ = curve_fit(
            double_pwl_model, freq, G, p0, bounds,
            method='trf', ftol=1e-08, xtol=1e-08, gtol=1e-08)

        # Assign fit results to model params
        self.A = res[0]
        self.B = res[1]
        self.alpha = res[2]
        self.beta = res[3]
        
        # Get goodness of fit params
        redchi = self.get_red_chisq(freq, G, w0, split_indx)

        return res, redchi

    def eval(self, freq, w0, split_indx):
        return self.objective(freq, self.A, self.B, self.alpha, self.beta, w0, split_indx)

    def get_chisq(self, x, y, w0, split_indx):
        return np.sum(((y - self.eval(x, w0, split_indx))/np.std(y))**2)
    
    def get_red_chisq(self, x, y, w0, split_indx):
        return self.get_chisq(x, y, w0, split_indx) / self.n_params