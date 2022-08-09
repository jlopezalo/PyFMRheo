from ast import Mod
import numpy as np
from lmfit import Model, Parameters

class SineWave:
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
    
    def build_params(self):
        params = Parameters()
        params.add('amplitude', value=self.amplitude_init)
        params.add('phase', value=self.phase_init)
        params.add('offset', value=self.offset_init)
        return params

    def model(self, time, amplitude, phase, offset, ang_freq):
        return amplitude * np.sin(ang_freq * time + phase) + offset

    def fit(self, time, wave):

        sinemodel =\
             lambda time, amplitude, phase, offset: self.model(time, amplitude, phase, offset, self.ang_freq)

        sinemodelfit = Model(sinemodel)

        # Define free params
        params = self.build_params()

        # Do fit
        self.n_params = len(sinemodelfit.param_names)
        result_sine = sinemodelfit.fit(wave, params, time=time)
        
        # Assign fit results to model params
        self.amplitude = result_sine.best_values['amplitude']
        self.phase = result_sine.best_values['phase']
        self.offset = result_sine.best_values['offset']

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
        return self.model(time, self.amplitude, self.phase, self.offset, self.ang_freq)
    
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



