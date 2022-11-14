# coding: utf-8
import numpy as np
from scipy.special import kv

from .sader import SaderGCI_CalculateK

BoltzmannConst = 1.380649e-23 # J⋅K−1

def qsat(Ta,Pa=None):
    if not Pa:
        P_default = 1020     # default air pressure for Kinneret [mbars]
        Pa=P_default # pressure in mb
    ew = 6.1121*(1.0007+3.46e-6*Pa)*np.exp((17.502*Ta)/(240.97+Ta)) # in mb
    return 0.62197*(ew/(Pa-0.378*ew)) # mb -> kg/kg

def air_dens(Ta, RH, Pa=None):
    eps_air = 0.62197    # molecular weight ratio (water/air)
    CtoK = 273.16        # conversion factor for [C] to [K]
    gas_const_R = 287.04 # gas constant for dry air [J/kg/K]
    if not Pa:
        P_default = 1020     # default air pressure for Kinneret [mbars]
        Pa = P_default
    o61 = 1/eps_air-1                 # 0.61 (moisture correction for temp.)
    Q = (0.01*RH)*qsat(Ta,Pa)     # specific humidity of air [kg/kg]
    T = Ta+CtoK                     # convert to K
    Tv = T*(1 + o61*Q)              # air virtual temperature
    return (100*Pa)/(gas_const_R*Tv);  # air density [kg/m^3]

def viscair(Ta):
    return 1.326e-5*(1 + 6.542e-3*Ta + 8.301e-6*Ta**2 - 4.84e-9*Ta**3)

def air_properties(T,RH):
    rho = air_dens(T,RH)
    eta = viscair(T)
    eta = eta*rho
    return rho, eta

def C_to_kelvin(C):
    # T (K) = T (°C) + 273.15
    return C + 273.15

def kelvin_to_C(kelvin):
    # T (°C) = T (K) + 273.15
    return kelvin - 273.15

def get_spring_constant(fR, Q, A1, temperature):
    abs_temp = C_to_kelvin(temperature)
    energy = BoltzmannConst * abs_temp
    return energy / ( np.pi / 2 * fR * np.abs(Q) * A1**2)

def reynolds_number_rect(rho, eta, omega, b):
    # Input:
        # rho:   density of surrounding fluid {kg/m^3}
        # eta:   viscosity of surrounding fluid {kg/m.s}
        # omega: cantilever-fluid resonant frequency
        # b:     width of the cantilever {m}
    # Output
        # Re:    Reynolds Number {unitless}
    return 0.250 * rho * omega * b**2 / eta

def reynolds_number_V(rho, eta, omega, d):
    # Input:
        # rho:   density of surrounding fluid {kg/m^3}
        # eta:   viscosity of surrounding fluid {kg/m.s}
        # omega: cantilever-fluid resonant frequency
        # d:     width of the legs of cantilever {m}
    # Output
        # Re:    Reynolds Number {unitless}
    return rho * omega * d**2 / eta

def omega(Re):
    tau = np.log10(Re)
    omega_real = (0.91324 - 0.48274 * tau + 0.46842 * tau**2 - 0.12886 * tau**3 \
        + 0.044055 * tau**4 - 0.0035117 * tau**5 + 0.00069085 * tau**6) \
        * (1 - 0.56964 * tau + 0.48690 * tau**2 - 0.13444 * tau**3 \
        + 0.045155 *  tau**4 - 0.0035862 * tau**5 \
        + 0.00069085 * tau**6)**-1
    omega_imag = (-0.024134 - 0.029256 * tau + 0.016294 * tau**2 \
        - 0.00010961 * tau**3 + 0.000064577 * tau**4 \
        - 0.000044510 * tau**5 )*( 1 - 0.59702 * tau + 0.55182 * tau**2 \
        - 0.18357 * tau**3 + 0.079156 * tau**4 - 0.014369 * tau**5 \
        + 0.0028361 * tau**6 )**-1
    return np.complex(omega_real, omega_imag)

def gamma_circ(Re):
    K1 = kv(1, -1j*np.sqrt(1j*Re))
    K0 = kv(0, -1j*np.sqrt(1j*Re))
    return 1 + 4*1j*K1 / (np.sqrt(1j*Re)*K0)

def gamma_rect(Re):
    return omega(Re) * gamma_circ(Re)

def force_constant(rho, eta, b, L, d, Q, omega, cantType):
    if cantType == 'Rectangular':
        Re = reynolds_number_rect(rho, eta, omega, b)
    elif cantType == 'V Shape':
        Re = reynolds_number_V(rho, eta, omega, d)
    gamma_imag = np.imag(gamma_rect(Re))
    return 0.1906 * rho * b**2 * L * Q * gamma_imag * omega**2

def Stark_Chi_force_constant(
    b, L, d, A1, fR1, Q1, Tc, RH, medium, cantType, k0=None,
    CorrFact=None, beta=None, Chi=None, invOLSscaling=None,
    username="", password="", selectedCantCode=""
):
    """
    Computes the spring constant (k) in N/m and the deflection sensitivity (invOLS) in m/V 
    of the cantilever using the Sader general method and the Sader GCI method.

            Parameters:
                    b (float): Width of the cantilever in m
                    L (str): Lenght of the cantilever in m
                    d (float): Width cantilever legs in m
                    A1 (float): in m^2/Hz
                    fR1 (float): Resonance frequency of the cantilever in Hz
                    Q1 (float): Quality factor of the cantilever
                    Tc (float): Medium temperature in celcius
                    RH (float): Relative humidity of the medium (0-100%)
                    medium (float): air or water
                    cantType (float): Rectangular or V Shaped
                    k0 (float): Spring constant to use in N/m
                    CorrFact (float): Correction factor to use (CorrFact = beta / Chi^2) (optional)
                    beta (float): Beta to compute correction factor (beta = k / k1) (optional)
                    Chi (float): Chi to compute correction factor (Chi = InvOLSfree / InvOLS) (optional)
                    username (float): Username for GCI web app (optional)
                    password (float): Password for GCI web app (optional)
                    selectedCantCode (float): Valid cantilever code for GCI web app (optional)
            
            Returns:
                    k0 (float): General Sader method spring constant
                    k_GCI (float): Spring constant from GCI (only if username, password and cantilever code are given)
                    invOLS_SHO (float): SHO invOLS
                    invOLS_H (float): Higgins invOLS
            
             References:
                    Stark et al. (2001) https://doi.org/10.1016/S0304-3991(00)00077-2  
                    Sader et al. (2005) https://doi.org/10.1063/1.1935133
                    Higgins (2006) https://doi.org/10.1063/1.2162455
                    Pirzer & Hugel (2009) https://doi.org/10.1063/1.3100258
                    Sader et al. (2016) https://doi.org/10.1063/1.4962866 
                    Sumbul et al. (2020) https://doi.org/10.3389/fphy.2020.00301
    """
    
    # Constants
    kB = 1.380649e-23 # in Nm/K
    T = C_to_kelvin(Tc) # in K
    omegaR1 = fR1 * 2 * np.pi # Hz --> Rad/s

    # Compute correction factor using:
    # beta = k / k1
    # Chi = InvOLSfree / InvOLS
    # References:
    # Sumbul et al. (2020) https://doi.org/10.3389/fphy.2020.00301
    if Chi is not None and beta is not None:
        CorrFact = beta / Chi**2

    # Default Chi values from:
    # Pirzer & Hugel (2009) https://doi.org/10.1063/1.3100258
    # Stark et al. (2001) https://doi.org/10.1016/S0304-3991(00)00077-2  
    if cantType == 'Rectangular' and CorrFact is None: CorrFact = 0.8174
    elif cantType == 'V Shape' and CorrFact is None: CorrFact: CorrFact = 0.764

    # Get properties of the medium
    if medium == 'air':
        rho, eta = air_properties(Tc, RH)
    elif medium == 'water':
        rho = 1000
        eta = 0.9e-3
    
    # Get spring constant using general Sader method:
    # Sader et al. (2005) https://doi.org/10.1063/1.1935133
    if k0 is None:
        k0 = force_constant(rho, eta, b, L, d, Q1, omegaR1, cantType)

    # Get deflection sensitivity using SHO:
    # Sumbul et al. (2020) https://doi.org/10.3389/fphy.2020.00301
    # InvOLSliq = sqrt((beta * kb * T * 2Q1) / (Chi^2 * k1 * pi * A1^2 * fR1))
    invOLS_SHO = np.sqrt(CorrFact * (kB * T * 2 * Q1) / (k0 * np.pi * A1**2 * fR1))

    # Get deflection sensitivity using Higgins:
    # Higgins (2006) https://doi.org/10.1063/1.2162455
    # InvOLS = sqrt((2 * Kb * T) / (pi * k1 * fR1 * A1 * Q1))
    invOLS_H = np.sqrt(2 * kB * T / (np.pi * k0 * A1**2 / Q1 * fR1)) * np.sqrt(CorrFact)

    # Scale the invOLS results by the sensitivity used when loading the thermal
    if invOLSscaling is not None:
        invOLS_SHO *= invOLSscaling
        invOLS_H *= invOLSscaling

    # Call to the GCI API:
    # GCI Webapp: https://www.sadermethod.org/
    # GCI API code: https://github.com/SaderMethod/API/tree/master/1_1/Python
    # GCI Ref.: https://doi.org/10.1063/1.4757398
    # GCI Webtool ref: https://doi.org/10.1063/1.4962866 
    if medium == "air" and username != "" and password != "" and selectedCantCode != "":
        fR1_khz = fR1 * 1e-3 # Hz --> kHz
        k_GCI=SaderGCI_CalculateK(username, password, selectedCantCode, fR1_khz, Q1)
    else:
        k_GCI=np.NaN
    
    # Return results
    return k0, k_GCI, invOLS_SHO, invOLS_H

def test_k_calibration():
    # http://dx.doi.org/10.1063/1.1150021
    # http://experimentationlab.berkeley.edu/sites/default/files/AFMImages/Sader.pdf
    # force_constant(rho, eta, b, L, d, Q, omega
    print(force_constant(1.18, 1.86e-5, 29e-6, 397e-6, 0, 55.5, 17.36*10**3*2*3.1415, 'Rectangular'))
    print(force_constant(1.18, 1.86e-5, 29e-6, 197e-6, 0, 136.0, 69.87*10**3*2*3.1415, 'Rectangular'))
    print(force_constant(1.18, 1.86e-5, 29e-6,  97e-6, 0, 309.0, 278.7*10**3*2*3.1415, 'Rectangular'))

    print(force_constant(1.18, 1.86e-5, 20e-6, 203e-6, 0, 17.6, 10.31*10**3*2*3.1415, 'Rectangular'))
    print(force_constant(1.18, 1.86e-5, 20e-6, 160e-6, 0, 22.7, 15.61*10**3*2*3.1415, 'Rectangular'))
    print(force_constant(1.18, 1.86e-5, 20e-6, 128e-6, 0, 30.9, 24.03*10**3*2*3.1415, 'Rectangular'))
    print(force_constant(1.18, 1.86e-5, 20e-6, 105e-6, 0, 41.7, 36.85*10**3*2*3.1415, 'Rectangular'))
    print(force_constant(1.18, 1.86e-5, 20e-6,  77e-6, 0, 60.3, 64.26*10**3*2*3.1415, 'Rectangular'))