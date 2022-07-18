import numpy as np
from scipy.optimize import curve_fit

class SineModel:
    def __init__(self, ang_freq):
        # Angular frequency of sine signal
        self.ang_freq = ang_freq
        # Amplitude of sine signal
        self.amplitude = None
        self.amplitude_init = None
        # Phase of sine signal
        self.phase = None
        self.phase_init = None
        # Offset of sine signal
        self.offset = None
        self.offset_init = None
        # Goodness of sine signal
        self.MAE = None
        self.SE = None
        self.MSE = None
        self.RMSE = None
        self.Rsquared = None
        self.chisq = None
        self.redchi = None

    def objective(self, time, amplitude, phase, offset, ang_freq):
        return amplitude * np.sin(ang_freq * time + phase) + offset

    def fit(self, time, wave):
        p0 = [self.amplitude_init, self.phase_init, self.offset_init]
        sinemodel =\
             lambda time, amplitude, phase, offset: self.objective(time, amplitude, phase, offset, self.ang_freq)
        res, _ = curve_fit(sinemodel, time, wave, p0)
        
        # Assign fit results to model params
        self.amplitude = res[0]
        self.phase = res[1]
        self.offset = res[2]

        modelPredictions = self.eval(time)

        absError = modelPredictions - wave

        self.MAE = np.mean(absError) # mean absolute error
        self.SE = np.square(absError) # squared errors
        self.MSE = np.mean(self.SE) # mean squared errors
        self.RMSE = np.sqrt(self.MSE) # Root Mean Squared Error, RMSE
        self.Rsquared = 1.0 - (np.var(absError) / np.var(wave))
        
        # Get goodness of fit params
        self.chisq = self.get_chisq(time, wave)
        self.redchi = self.get_red_chisq(time, wave)
    
    def eval(self, time):
        return self.objective(time, self.amplitude, self.phase, self.offset, self.ang_freq)
    
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



