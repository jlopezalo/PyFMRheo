import numpy as np
from .hertz import hertz_model_params
from mpmath import gamma, hyper


def ting_analytical_cone(
    time, betaE, E0, slope, f0, tm, t0, v0t, v0r, ind_shape, half_angle, poisson_ratio
):

    if ind_shape not in ("paraboloid", "pyramid", "cone"):
        raise Exception(
            "The Ting Analytical Cone model is only suitable for the pyramid and cone geometries."
        )
    
    v0=(v0r+v0t)/2

    # Compute index of tmax
    tm_indx = (np.abs(time - tm)).argmin()

    # Split time vector on trace and retrace
    ttc = time[:tm_indx+1]
    trc = time[tm_indx+1:]

    # Compute t1 for retrace segment
    t1_full=trc-(1+v0r/v0t)**(1/(1-betaE))*(trc-tm)
    t1_end_indx = (np.abs(t1_full - 0)).argmin()
    trc_end = t1_end_indx + tm_indx 

    # Get retrace time based on t1
    t1 = t1_full[t1_full>0]
    trc = trc[t1_full>0]

    if ind_shape == "paraboloid":

        coeff_func, _ = hertz_model_params[ind_shape]
        Cp = 1 / coeff_func(half_angle, poisson_ratio)

        Ftp=3/2*v0t**(3/2)*E0*t0**betaE*np.sqrt(np.pi)*gamma(1-betaE)/(Cp*2*gamma(5/2-betaE))*ttc**(3/2-betaE)

        if np.abs(v0r-v0t)/v0t < 0.01:
            Frp=3/2*v0r**(3/2)*E0*t0^betaE*np.sqrt(np.pi)*gamma(1-betaE)/(Cp*2*gamma(5/2-betaE))*t1**(3/2-betaE)

        else:
            a = [hyper([1, 1/2-betaE], [1/2], t1[i]/trc[i]) for i in range(len(trc))]
            Frp=3/Cp*E0*v0t**(3/2)*t0**betaE/(3+4*(betaE-2)*betaE)*t1**(-1/2)*(trc-t1)**(1-betaE)*\
                (-trc+(2*betaE-1)*t1+trc*a)
        FJ = np.r_[Ftp, Frp]
    
    elif ind_shape in ("pyramid", "cone"):

        coeff_func, _ = hertz_model_params[ind_shape]
        Cc = 1 / coeff_func(half_angle, poisson_ratio)

        if np.abs(v0r-v0t)/v0t < 0.01:
            Ftc=2*v0**2*E0*t0**betaE/Cc/(2-3*betaE+betaE**2)*ttc**(2-betaE)
        else:
            Ftc=2*v0t**2.*E0*t0**betaE/Cc/(2-3*betaE+betaE**2)*ttc**(2-betaE)

        if np.abs(v0r-v0t)/v0t < 0.01:
            Frc=-2*v0**2*E0*t0**betaE/Cc/(2-3*betaE+betaE**2)*((trc-t1)**(1-betaE)*(trc+(1-betaE)*t1)-\
                trc**(1-betaE)*(trc))
        else:
            Frc=-2*v0t**2*E0*t0**betaE/Cc/(2-3*betaE+betaE**2)*((trc-t1)**(1-betaE)*(trc+(1-betaE)*t1)-\
                trc**(1-betaE)*(trc))
        FJ = np.r_[Ftc, Frc]
    
    ncr = len(time) - len(FJ)
    
    return np.r_[FJ, np.ones(ncr)*f0]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    poisson_ratio = 0.5
    tip_angle = 75 / 1e9
    v0t = 4.9999999999999996e-06
    v0r = 6e-06
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

    f = ting_analytical_cone(time, betaE, E0, slope, f0, tm, t0, v0r, v0t, "paraboloid", tip_angle, poisson_ratio)

    plt.plot(time, f)
    plt.show()