import numpy as np

from ..models.sho import SHOModel
from ..models.calibration import Stark_Chi_force_constant


def doNonContactCal(
    freq, ampl, cantiparams, medium, Tc, RH, k0=None, CorrFact=None,
    initSHOparams=None, Beta=None, Chi=None, invOLSscaling=None,
    minfreq=None, maxfreq=None, GCIusername="", GCIpassword="", selectedCantCode=""
):
    
    # Select data
    mask = np.logical_and(freq >= minfreq, freq <= maxfreq)
    freq_fit = freq[mask]
    ampl_fit = ampl[mask]

    max_amp_indx = np.argmax(ampl)
    max_amp_freq = freq[max_amp_indx]
    max_amp = ampl[max_amp_indx]

    # Declare model and define initial params
    sho_model = SHOModel()

    sho_model.Awhite_init = np.sqrt((ampl[1]))
    sho_model.Awhite_min = np.sqrt(np.min(ampl)/100)
    sho_model.Awhite_max = np.sqrt(np.max(ampl)*10)

    if initSHOparams is not None:
        sho_model.A_init = initSHOparams['A']
        sho_model.fR_init = initSHOparams['fR']
        sho_model.Q_init = initSHOparams['Q']
    else:
        sho_model.A_init = np.sqrt(max_amp)
        sho_model.fR_init = max_amp_freq
        sho_model.Q_init = 1
        
    sho_model.A_min = np.sqrt(np.max(ampl)/100)
    sho_model.fR_min = max_amp_freq/3
    sho_model.Q_min = 0.1

    sho_model.A_max = np.sqrt(np.max(ampl)*10) 
    sho_model.fR_max = max_amp_freq*3 
    sho_model.Q_max = 100
    
    # Do fit
    sho_model.fit(freq_fit, ampl_fit)

    # Compute
    return Stark_Chi_force_constant(
        cantiparams['cantiWidth'], cantiparams['cantiLen'], cantiparams['cantiWidthLegs'],
        sho_model.A, sho_model.fR, sho_model.Q, Tc, RH, medium, cantiparams['cantType'],
        k0, CorrFact, Beta, Chi, invOLSscaling, GCIusername, GCIpassword, selectedCantCode
    )