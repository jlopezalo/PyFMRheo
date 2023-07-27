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

    # Declare model and define initial params
    sho_model = SHOModel()

    if initSHOparams is not None:
        sho_model.A_init = initSHOparams['A']
        sho_model.fR_init = initSHOparams['fR']
        sho_model.Q_init = initSHOparams['Q']
    
    # Do fit
    sho_model.fit(freq_fit, ampl_fit)

    # Compute
    return Stark_Chi_force_constant(
        cantiparams['cantiWidth'], cantiparams['cantiLen'], cantiparams['cantiWidthLegs'],
        sho_model.A, sho_model.fR, sho_model.Q, Tc, RH, medium, cantiparams['cantType'],
        k0, CorrFact, Beta, Chi, invOLSscaling, GCIusername, GCIpassword, selectedCantCode
    )