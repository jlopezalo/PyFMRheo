import numpy as np
from scipy.optimize import curve_fit

class SHOModel:
    def __init__(self):
        self.n_params = None
        # A white
        self.Awhite = None
        self.Awhite_init = None
        self.Awhite_min = None
        self.Awhite_max = None
        # Amplitude
        self.A = None
        self.A_init = None
        self.A_min = None
        self.A_max = None
        # Resonance frequency of cantilver
        self.fR = None
        self.fR_init = None
        self.fR_min = None
        self.fR_max = None
        # Q factor
        self.Q = None
        self.Q_init = None
        self.Q_min = None
        self.Q_max = None
        # Goodness of sine signal
        self.MAE = None
        self.SE = None
        self.MSE = None
        self.RMSE = None
        self.Rsquared = None
        self.chisq = None
        self.redchi = None

    def objective(self, freq, Awhite, A, fR, Q):
        return Awhite**2 + A**2 * fR**4 / Q**2 * ((freq**2-fR**2)**2 + freq**2 * fR**2 / Q**2)**(-1)

    def fit(self, freq, ampl):
        max_amp_indx = np.argmax(ampl)
        max_amp_freq = freq[max_amp_indx]
        max_amp = ampl[max_amp_indx]

        # Define initial values for fit
        self.Awhite_init = self.Awhite_init or np.sqrt((ampl[1]))
        self.A_init = self.A_init or np.sqrt(max_amp)
        self.fR_init = self.fR_init or max_amp_freq
        self.Q_init = self.Q_init or 1
        
        # Define lower bounds for variables
        self.Awhite_min = self.Awhite_min or np.sqrt(np.min(ampl)/100)
        self.A_min = self.A_min or np.sqrt(np.max(ampl)/100)
        self.fR_min = self.fR_min or max_amp_freq/3
        self.Q_min = self.Q_min or 0.1

        # Define upper bounds
        self.Awhite_max = self.Awhite_max or np.sqrt(np.max(ampl)*10)
        self.A_max = self.A_max or np.sqrt(np.max(ampl)*10) 
        self.fR_max = self.fR_max or max_amp_freq*3
        self.Q_max = self.Q_max or 100

        p0 = [self.Awhite_init, self.A_init, self.fR_init, self.Q_init]
        
        bounds = [
            [self.Awhite_min, self.A_min, self.fR_min, self.Q_min],
            [self.Awhite_max, self.A_max, self.fR_max, self.Q_max]
        ]
        
        res, _ = curve_fit(self.objective, freq, ampl, p0, bounds=bounds)
        
        # Assign fit results to model params
        self.Awhite = res[0]
        self.A = res[1]
        self.fR = res[2]
        self.Q = res[3]

        self.n_params = 4

        modelPredictions = self.eval(freq)

        absError = modelPredictions - ampl

        self.MAE = np.mean(absError) # mean absolute error
        self.SE = np.square(absError) # squared errors
        self.MSE = np.mean(self.SE) # mean squared errors
        self.RMSE = np.sqrt(self.MSE) # Root Mean Squared Error, RMSE
        self.Rsquared = 1.0 - (np.var(absError) / np.var(ampl))
        
        # Get goodness of fit params
        self.chisq = self.get_chisq(freq, ampl)
        self.redchi = self.get_red_chisq(freq, ampl)
    
    def eval(self, freq):
        return self.objective(freq, self.Awhite, self.A, self.fR, self.Q)
    
    def get_residuals(self, time, wave):
        return wave - self.eval(time)

    def get_chisq(self, time, wave):
        a = (self.get_residuals(time, wave)**2/wave)
        return np.sum(a[np.isfinite(a)])
    
    def get_red_chisq(self, time, wave):
        return self.get_chisq(time, wave) / self.n_params
    
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
        MSE: {self.MSE}\n
        RMSE: {self.RMSE}\n
        Rsq: {self.Rsquared}\n
        Chisq: {self.chisq}\n
        RedChisq: {self.redchi}\n
        """
        )



