import numpy as np


# Pyramid params coefficient
Cc_pyr = lambda half_angle, poisson_ratio: np.pi * (1 - poisson_ratio**2) / (2 * np.tan(np.radians(half_angle)))


# Cone geometry coefficient
Cc_cone = lambda half_angle, poisson_ratio: 2/np.pi * np.tan(np.radians(half_angle)) * 1/(1-poisson_ratio**2)


ting_analytical_params = {"pyramid": Cc_pyr, "cone": Cc_cone}


def ting_analytical_cone(
    time, betaE, E0, slope, f0, tm, t0, v0r, v0t, ind_shape, half_angle, poisson_ratio
):

    if ind_shape not in ("pyramid", "cone"):
        raise Exception(
            "The Ting Analytical Cone model is only suitable for the pyramid and cone geometries."
        )

    Cc_func = ting_analytical_params.get(ind_shape)
    Cc = Cc_func(half_angle, poisson_ratio)
    
    # Compute index of tmax
    tm_indx = (np.abs(time - tm)).argmin()

    # Split time vector on trace and retrace
    ttc = time[:tm_indx+1]
    trc = time[tm_indx+1:]

    # Compute t1 for retrace segment
    t1=trc-(1+v0r/v0t)**(1/(1-betaE))*(trc-tm)
    t1_end_indx = (np.abs(t1 - 0)).argmin()
    trc_end = t1_end_indx + tm_indx + 1

    # Get retrace time based on t1
    trc = trc[:t1_end_indx]

    if np.abs(v0r-v0t)/v0t < 0.01:
        Ftc=2*v0t**2*E0*t0**betaE/Cc/(2-3*betaE+betaE**2)*ttc**(2-betaE)
    else:
        Ftc=2*v0t**2*E0*t0**betaE/Cc/(2-3*betaE+betaE**2)*ttc**(2-betaE)

    if np.abs(v0r-v0t)/v0t < 0.01:
        Frc=2*v0r**2*E0*t0**betaE/Cc/(2-3*betaE+betaE**2)*(trc**(2-betaE)-2*(trc-tm)**(1-betaE)*(trc*(2-betaE)-2**(1/(1-betaE))*(1-betaE)*(trc-tm)))
    else:
        Frc=2*E0*t0**betaE/Cc/(2-3*betaE+betaE**2)*v0r*(trc**(1-betaE)*(trc*v0r+(betaE-2)*tm*(v0r-v0t))-\
            (trc-tm)**(1-betaE)*(1+v0r/v0t)*(trc*v0r*(2-(1+v0r/v0t)**(1/(1-betaE))+betaE*((1+v0r/v0t)**(1/(1-betaE))-1))-\
            tm*((betaE-2)*v0t+v0r*(2-(1+v0r/v0t)**(1/(1-betaE))+betaE*((1+v0r/v0t)**(1/(1-betaE))-1)))))
    
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
    v0t = 4.9999999999999996e-06
    v0r = 4.9999999999999996e-06
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

    f = ting_analytical_cone(time, betaE, E0, slope, f0, tm, t0, v0r, v0t, "pyramid", tip_angle, poisson_ratio)

    plt.plot(time, f)
    plt.show()