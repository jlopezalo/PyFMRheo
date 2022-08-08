import numpy as np
from lmfit import Model, Parameters

class DoublePowerLawModel:
    def __init__(self) -> None:
        # A factor
        self.A = None
        self.A_init = None
        self.A_min = 0
        self.A_max = np.inf
        # Alpha
        self.alpha = None
        self.alpha_init = None
        self.alpha_min = 0
        self.alpha_max = np.inf
        # B factor
        self.B = None
        self.B_init = None
        self.B_min = 0
        self.B_max = np.inf
        # Beta
        self.beta = None
        self.beta_init = None
        self.beta_min = 0
        self.beta_max = np.inf
    
    def build_params(self):
        params = Parameters()
        params.add('A', value=self.A_init, min=self.A_min, max=self.A_max)
        params.add('B', value=self.B_init, min=self.B_min, max=self.B_max)
        params.add('alpha', value=self.alpha_init, min=self.alpha_min, max=self.alpha_max)
        params.add('beta', value=self.beta_init, min=self.beta_min, max=self.beta_max)
        return params

    def model(self, freq, A, B, alpha, beta, w0, split_indx):
        if alpha > beta: alpha, beta = beta, alpha

        G = np.zeros(freq.shape)

        G[:split_indx] = A * np.cos(np.pi * alpha / 2) * (freq[:split_indx] / w0) ** alpha +\
                        B * np.cos(np.pi * beta / 2) * (freq[:split_indx] / w0) ** beta

        G[split_indx:] = A * np.sin(np.pi * alpha / 2) * (freq[split_indx:] / w0) ** alpha +\
                        B * np.sin(np.pi * beta / 2) * (freq[split_indx:] / w0) ** beta

        return G

    def fit(self, freq, G, w0, split_indx):
        
        fixed_params = {'w0': w0, 'split_indx': split_indx}

        pwlmodel =\
            lambda freq, A, B, alpha, beta: self.model(freq, A, B, alpha, beta, **fixed_params)

        pwlmodelfit = Model(pwlmodel)

        # Define free params
        params = self.build_params()

        # Do fit
        self.n_params = len(pwlmodelfit.param_names)

        result_pwl = pwlmodelfit.fit(G, params, freq=freq)

        # Assign fit results to model params
        self.A = result_pwl.best_values['A']
        self.B = result_pwl.best_values['B']
        self.alpha = result_pwl.best_values['alpha']
        self.beta = result_pwl.best_values['beta']
        
        # Compute metrics
        modelPredictions = self.eval(freq, w0, split_indx)

        absError = modelPredictions - G

        self.MAE = np.mean(absError) # mean absolute error
        self.SE = np.square(absError) # squared errors
        self.MSE = np.mean(self.SE) # mean squared errors
        self.RMSE = np.sqrt(self.MSE) # Root Mean Squared Error, RMSE
        self.Rsquared = 1.0 - (np.var(absError) / np.var(G))

        # Get goodness of fit params
        self.chisq = self.get_chisq(freq, G, w0, split_indx)
        self.redchi = self.get_red_chisq(freq, G, w0, split_indx)

    def eval(self, freq, w0, split_indx):
        return self.model(freq, self.A, self.B, self.alpha, self.beta, w0, split_indx)

    def get_chisq(self, x, y, w0, split_indx):
        return np.sum(((y - self.eval(x, w0, split_indx))/np.std(y))**2)
    
    def get_red_chisq(self, x, y, w0, split_indx):
        return self.get_chisq(x, y, w0, split_indx) / self.n_params