from math import gamma
import numpy as np
from sympy import beta
from .hertz import hertz_model_params


def ting_analytical_cone(
    time, betaE, E0, slope, f0, tm, t0, v0, ind_shape, half_angle, poisson_ratio
):

    if ind_shape not in ("pyramid", "cone"):
        raise Exception(
            "The Ting Analytical Cone model is only suitable for the pyramid and cone geometries."
        )

    coeff_func, _ = hertz_model_params[ind_shape]
    Cc = 1 / coeff_func(half_angle, poisson_ratio)
    
    # Compute index of tmax
    tm_indx = (np.abs(time - tm)).argmin()

    # Split time vector on trace and retrace
    ttc = time[:tm_indx+1]
    trc = time[tm_indx+1:]

    # Compute t1 for retrace segment
    t1=trc-(1+v0)**(1/(1-betaE))*(trc-tm)
    t1_end_indx = (np.abs(t1 - 0)).argmin()
    trc_end = t1_end_indx + tm_indx + 1

    # Get retrace time based on t1
    trc = trc[:t1_end_indx]

    Ftc=v0**2/Cc*E0*(t0**betaE*gamma(2)*gamma(1-betaE)/gamma(3-betaE))*ttc**(2-betaE)

    Frc=v0**2*E0*t0**betaE/Cc*(gamma(2)*gamma(1-betaE)/gamma(3-betaE))*2*\
        (t0**(2-betaE)-2*(ttc*(2-betaE)+2**(1/(1-betaE))*(1-betaE)*(ttc-tm))*(ttc-tm)**(1-betaE))
    
    # Output array
    force = np.empty(time.shape)

    # Assign force to output array
    force[:tm_indx+1] = Ftc + f0
    force[tm_indx+1:trc_end] = Frc + f0
    force[trc_end:].fill(Frc[-1] + f0)
    
    return force


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    poisson_ratio = 0.5
    tip_angle = 35.0
    v0 = 4.9999999999999996e-06
    t0 = 1
    E0 = 603
    betaE = 0.2
    f0 = 0
    slope = 0

    ttc = np.linspace(0, 1, 100)
    trc = np.linspace(1.0, 2, 100)
    tm = 1
    tc=0.2

    time = np.r_[ttc, trc]

    f = ting_analytical_cone(time, betaE, E0, slope, f0, tm, t0, v0, "pyramid", tip_angle, poisson_ratio)

    plt.plot(time, f)
    plt.show()