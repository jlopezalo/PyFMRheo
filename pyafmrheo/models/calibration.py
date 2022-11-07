import numpy as np
from scipy.special import kv

from .sader import SaderGCI_CalculateK

BoltzmannConst = 1.380649e-23 # J⋅K−1

def qsat(Ta,Pa=None):
    P_default = 1020     # default air pressure for Kinneret [mbars]
    if not Pa:
        Pa=P_default # pressure in mb
    ew = 6.1121*(1.0007+3.46e-6*Pa)*np.exp((17.502*Ta)/(240.97+Ta)) # in mb
    return 0.62197*(ew/(Pa-0.378*ew));                         # mb -> kg/kg

def air_dens(Ta, RH, Pa=None):
    eps_air = 0.62197    # molecular weight ratio (water/air)
    P_default = 1020     # default air pressure for Kinneret [mbars]
    CtoK = 273.16        # conversion factor for [C] to [K]
    gas_const_R = 287.04 # gas constant for dry air [J/kg/K]
    if not Pa:
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

def Stark_Chi_force_constant(b, L, d, A1, fR1, Q1, Tc, RH, medium, cantType, username="", pwd="", selectedCantCode=""):
    """
    Computes the spring constant (k) in N/m and the deflection sensitivity (invOLS) in m/V 
    of the cantilever using the Sader general method and the Sader GCI method.

            References:
                    Sader 1
                    Sader 2
                    Higgins
                    Fidane
                    GCI 

            Parameters:
                    b (float): in m
                    L (str): in m
                    d (float): in m
                    A1 (float): in m^2/Hz
                    fR1 (float): in Hz
                    Q1 (float): No units
                    Tc (float): in celcius
                    RH (float): in %
                    medium (float): air or water
                    cantType (float): Rectangular or V Shaped
                    username (float): Username
                    pwd (float): Password
                    selectedCantCode (float): Valid cantilever code
            
            Returns:
                    k0
                    GCI_cant_springConst
                    involsValue
                    invOLS_H
    """
    # Reference invOLS
    invOLS= 20*1e-9 # in m/V
    kB = BoltzmannConst # in J⋅K−1
    T=C_to_kelvin(Tc) # in K
    xsqrA1=np.pi*A1**2*fR1/2/Q1
    print('X2')
    print(xsqrA1)
    if cantType == 'Rectangular':
        Chi1= 0.8174
    elif cantType == 'V Shape':
        Chi1= 0.764
    kcantiA=Chi1*kB*T/xsqrA1
    print('K canti A')
    print(kcantiA)
    if medium == 'air':
        rho, eta = air_properties(Tc, RH)
    elif medium == 'water':
        rho = 1000
        eta = 0.9e-3
    print('Params for force constant')
    print(rho, eta, b, L, d, Q1, fR1*2*np.pi, cantType)
    k0 = force_constant(rho, eta, b, L, d, Q1, fR1*2*np.pi, cantType)
    if username != "" and pwd != "" and selectedCantCode != "":
        GCI_cant_springConst=SaderGCI_CalculateK(username, pwd, selectedCantCode, fR1/1e3, Q1)
    else:
        GCI_cant_springConst=np.NaN
    involsValue=invOLS*np.sqrt(kcantiA/k0)/1e9
    invOLS_H=np.sqrt(2*kB*T/(np.pi*k0*(A1)**2/Q1*fR1))*invOLS/1e9*np.sqrt(Chi1)

    return k0, GCI_cant_springConst, involsValue, invOLS_H