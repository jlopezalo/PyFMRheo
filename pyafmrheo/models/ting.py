import numpy as np
from mpmath import gamma, hyper
import matplotlib.pyplot as plt

def numdiff(y):
    diffy = np.zeros(len(y))
    idx = np.arange(2, len(y)-3)
    diffy[idx] = (-y[idx+2]+8*y[idx+1]-8*y[idx-1]+y[idx-2])/12
    diffy[:2]=diffy[2]
    diffy[len(diffy)-3:]=diffy[len(diffy)-4]
    return diffy

def smoothM(d, parS):
    y = d
    DL = len(d)-1
    for ij in range(len(d)-1):
        if np.isnan(y[ij]):
            k = 0
            while np.isnan(y[ij]) and ij+k < DL:
                k += 1
                y[ij] = y[ij+k]
    if parS > 1:
        y[1] = (d[1] + d[2] + d[3])/3
        y[-2] = (d[DL-2] + d[DL-1] + d[DL])/3
    if parS in [2, 3]: #for 2 and 3
        for ij in range(2, DL-2):
            y[ij] = (d[ij-1] + d[ij] + d[ij+1])/3
    if parS >= 4:  # :for 4 and 5 and any more
        for n in range(2, DL-2):
            y[n] = (d[n-2] + d[n-1] + d[n] + d[n+1] + d[n+2])/5
    return y

def SolveAnalytical(ttc, trc, t1, model_probe, geom_coeff, v0t, v0r, v0, E0, betaE, t0, F0, vdrag):
    # TO DO: ADD REFERENCE!!!
    if model_probe == 'paraboloid':
        Cp=1/geom_coeff
        Ftp=3/2*v0t**(3/2)*E0*t0**betaE*np.sqrt(np.pi)*np.array(gamma(1-betaE), dtype=float)/(Cp*2*np.array(gamma(5/2-betaE), dtype=float))*ttc**(3/2-betaE)
        if np.abs(v0r-v0t)/v0t<0.01:
            Frp=3/2*v0r**(3/2)*E0*t0**betaE*np.sqrt(np.pi)*np.array(gamma(1-betaE), dtype=float)/(Cp*2*np.array(gamma(5/2-betaE), dtype=float))*t1**(3/2-betaE)
        else:
            A = [hyper([1, 1/2-betaE], [1/2], t1[i]/trc[i]) for i in range(len(trc))]
            Frp=3/Cp*E0*v0t**(3/2)*t0**betaE/(3+4*(betaE-2)*betaE)*t1**(-1/2)*(trc-t1)**(1-betaE)*\
                (-trc+(2*betaE-1)*t1+trc*np.array(A, dtype=float))
        return np.r_[Ftp+v0t*vdrag, Frp-v0r*vdrag]+F0
    elif model_probe in ('cone', 'pyramid'):
        Cc=1/geom_coeff
        if np.abs(v0r-v0t)/v0t<0.01:
            Ftc=2*v0**2*E0*t0**betaE/Cc/(2-3*betaE+betaE**2)*ttc**(2-betaE)
            Frc=-2*v0**2.*E0*t0**betaE/Cc/(2-3*betaE+betaE**2)*((trc-t1)**(1-betaE)*(trc+(1-betaE)*t1)-\
                trc**(1-betaE)*(trc))
        else:
            Ftc=2*v0t**2*E0*t0**betaE/Cc/(2-3*betaE+betaE**2)*ttc**(2-betaE)
            Frc=-2*v0t**2*E0*t0**betaE/Cc/(2-3*betaE+betaE**2)*((trc-t1)**(1-betaE)*(trc+(1-betaE)*t1)-\
                trc**(1-betaE)*(trc))
        return np.r_[Ftc+v0t*vdrag, Frc-v0r*vdrag]+F0

def SolveNumerical(delta, time_, geom_coeff, geom_exp, v0t, v0r, E0, betaE, F0, vdrag, smooth_w, idx_tm, idxCt, idxCr):
    delta0 = delta - delta[idxCt[0]]
    delta_Uto_dot = np.zeros(len(delta0))
    A = smoothM(np.r_[numdiff(delta0[idxCt]**geom_exp), numdiff(delta0[idxCr[0]:]**geom_exp)], smooth_w)
    if len(A) < len(delta_Uto_dot[idxCt[0]:]):
        A = np.append(A, A[-1])
    delta_Uto_dot[idxCt[0]:] = A
    delta_dot = np.zeros(len(delta0))
    B = smoothM(np.r_[numdiff(delta0[idxCt]), numdiff(delta0[idxCr[0]:])], smooth_w)
    if len(B) < len(delta_Uto_dot[idxCt[0]:]):
        B = np.append(B, B[-1])
    delta_dot[idxCt[0]:] = B
    Ftc = np.zeros(len(idxCt))
    for i in range(len(idxCt)):
        idx = idxCt[0] + np.arange(1, i)
        Ftc[i] = geom_coeff * E0 * np.sum(delta_Uto_dot[idx]*np.flipud(time_[idx])**(-betaE))
    idx_min_phi0 = np.zeros(len(idxCt))
    Frc = np.zeros(len(idxCt))
    for j in range(idx_tm+1, idx_tm+len(idxCt)):
        phi0 = np.flipud(np.cumsum(np.flipud(time_[j-1:idxCt[1]-1:-1]**(-betaE)*delta_dot[idxCt[1]+1:j+1]), axis=0))
        phi0 = phi0[:len(idxCt)]
        idx_min_phi0 = np.argmin(np.abs(phi0))
        idxCr0 = np.arange(j+1, j-idx_min_phi0+1, -1)
        t10 = time_[idxCr0]
        idx = np.arange(idxCt[0]+1, idxCt[0]+idx_min_phi0+1)
        Frc[j-idx_tm-1] = geom_coeff * E0 * np.trapz(delta_Uto_dot[idx]*t10**(-betaE))
    return np.r_[Ftc+v0t*vdrag, Frc-v0r*vdrag]+F0

def Ting(time, t0, E0, tc, betaE, F0, F, delta, model_probe, geom_coeff, geom_exp, modelFt, vdrag, idx_tm=None, smooth_w=None):
    # Shift time using t at contact.
    time=time-tc
    # Compute deltat.
    deltat=time[1]-time[0]
    # If no t max index is given search the index of F max.
    if idx_tm is None:
        idx_tm = np.argmax(F)
    # Get t max value.
    tm = time[idx_tm]
    # Determine non contact trace region.
    idxNCt=np.where(time<0)[0]
    # Determine contact trace region
    idxCt=np.where(time>=0)[0]
    # Get indices corresponding to contact trace region.
    # Including t max.
    idxCt = np.arange(idxCt[0], idx_tm + 1)
    # Determine contact time trace.
    ttc=time[idxCt]
    # Define range to compute trace speed.
    # Including t max.
    range_v0t=np.arange((idx_tm-int(len(ttc)*3/4)), idx_tm)
    # Define range to compute retrace speed.
    # Excluding t max.
    range_v0r=np.arange(idx_tm+2, (idx_tm+1+int(len(ttc)*3/4)))
    # Fit 1 degree polynomial (x0 + m) to trace and retrace for determining
    # the corresponding speeds (x0)
    v0t = np.polyfit(time[range_v0t], delta[range_v0t], 1)[0]
    v0r = -1 * np.polyfit(time[range_v0r], delta[range_v0r], 1)[0]
    # Compute mean speed.
    v0=(v0r+v0t)/2
    # Compute retrace contact time.
    # TO DO: ADD REFERENCE TO ARTICLE!!!!
    tcr=(1+v0r/v0t)**(1/(1-betaE))/((1+v0r/v0t)**(1/(1-betaE))-1)*tm
    # If the retrace contact time is smaller than t max,
    # define the end of the contact retrace region as 3 times t max.
    if not tcr<tm:
        idxCr=np.where((time>tm) & (time<=tcr))[0]
    else:
        idxCr=np.where((time>tm) & (time<=3*tm))[0]
    # Define in contact retrace region.
    trc=time[idxCr]
    # Compute t1
    # TO DO: ADD REFERENCE TO ARTICLE!!!!
    t1=trc-(1+v0r/v0t)**(1/(1-betaE))*(trc-tm)
    # Select only the values larger than 0 of t1.
    t1=t1[t1>0]
    # Select the region of retrace time where t1 is larger than 0.
    trc=trc[t1>0]
    # Select the retrace contact indices corresponding to the retrace
    # time region where t1 is larger than 0. 
    idxCr=idxCr[:len(trc)]
    # Assign the value of F0 to the non contact region.
    FtNC=F0*np.ones(idxNCt.size)
    # Compute Force according to the selected mode:
    if modelFt == 'analytical':
        FJ = SolveAnalytical(
            ttc, trc, t1, model_probe, geom_coeff, v0t, v0r, v0, E0, betaE, t0, F0, vdrag
        )
    elif modelFt == 'numerical':
        FJ = SolveNumerical(
            delta, time, geom_coeff, geom_exp, v0t, v0r, E0, betaE, F0, vdrag, smooth_w, idx_tm, idxCt, idxCr
        )
    else:
        print(f'The modelFt {modelFt} is not supported. Current valid modelFt: analytical, numerical.')
    # Determine non contact retrace region.
    idxNCr=np.arange((len(FJ)+len(FtNC)+1),len(delta)+1)
    # Assign the value of F0 to the non contact region.
    FrNC=F0*np.ones(idxNCr.size)
    # Concatenate non contact regions to the contact region. And return.
    return np.r_[FtNC+v0t*vdrag, FJ, FrNC-v0r*vdrag]

if __name__ == "__main__":
    # Set numpy print parameter to show all elements of arrays
    np.set_printoptions(threshold=np.inf)
    # Declare variables
    E0=2500
    F0=0
    betaT0=0.01
    v0t=1000
    v0r=1200
    tm=1
    t0=tm/2
    N=1000
    tp=1
    vdrag=0
    smooth_w=1
    model_probe = 'paraboloid'
    # geom_coeff = np.pi * (1 - 0.5**2) / (2 * np.tan((np.pi * 35 / 180)))
    geom_coeff = 4/3 * np.sqrt(75*1e-9) * 1/(1-0.5**2)
    geom_exp = 3/2
    F0=0
    
    # Define time
    time_=np.linspace(-1/4,tm*2,N)
    
    # Define indentation
    delta = np.concatenate((v0t * time_[:len(time_)//2], v0t*tm+v0r*(tm-time_[(N//2):])), axis=None)

    # Define index of tmax
    idx_tm=np.where(time_>=tm)[0][0]

    # Testing NUMERICAL models #######################################
    tc = tm/2
    t0 = 1
    FT = Ting(time_, t0, E0, tc, betaT0, F0, delta, delta, model_probe, geom_coeff, geom_exp, 'numerical', vdrag, None, smooth_w)

    # Print max force to see if the peak is at the same force as in Felix code.
    # print(FT.max())

    # Testing ANALYTICAL models ######################################
    tc = tm/2
    t0 = 1
    FJ = Ting(time_, t0, E0, tc, betaT0, F0, delta, delta, model_probe, geom_coeff, geom_exp, 'analytical', vdrag, None, smooth_w)
    
    # Print max force to see if the peak is at the same force as in Felix code.
    # print(FJ.max())

    ###################################################################

    # Make plots to see the overlap of Numerical and Analytical solutions
    plt.plot(time_, FT, label='Numerical', alpha=0.7)
    plt.plot(time_, FJ, label='Analytical', alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Force (pN)')
    plt.title('Comparison of Ting models')
    plt.legend()
    plt.grid()
    plt.show()
