import numpy as np
import matplotlib.pyplot as plt
from pyafmrheo.models.ting_numerical import ting_numerical as ttingtest
from pyafmrheo.models.ting_analytical import ting_analytical_cone
from pyafmrheo.models.ting_numerical import simple_power_law

def ting_numerical(E0, alpha, dTp, Poisson, Radius, dT, MaxInd, modelprobe, indentationfull):
    PointN = len(indentationfull)
    time = np.linspace(0, dT*(PointN-1), PointN)

    # Et construction
    Et = E0 * (time/dTp) ** -alpha

    # plt.plot(time, Et)
    # remove zero-time singularity if needed
    if np.isinf(Et[0]):
        Et[0] = 2*Et[1]
    if np.isnan(Et[0]):
        Et[0] = Et[1]+(Et[1]-Et[2])

    BEC = np.ones(len(indentationfull))
    BECspeed = np.ones(len(indentationfull))

    if modelprobe == 'sphere':
        power = 1.5
        K1 = 4*Radius**0.5/3
        K12 = Radius**0.5
    elif modelprobe == 'cone' or modelprobe == 'pyramid':
        power = 2
        if modelprobe == 'cone':
            K1 = 2/np.pi*np.tan(Radius*np.pi/180)
        elif modelprobe == 'pyramid':
            K1 = 1.406/2*np.tan(Radius*np.pi/180)
        K12 = K1
    elif modelprobe == 'cylinder': # cylinder
        power = 1
        K1 = 2*Radius
        K12 = Radius
    K1 = K1/(1-Poisson**2)

    # indentation history
    ind2speed = np.diff(indentationfull**power)/dT
    ind2speed = np.append(ind2speed, ind2speed[-1])
    # ind2speed = smoothM(ind2speed, 5)
    indspeed = np.diff(indentationfull)/dT
    indspeed = np.append(indspeed, indspeed[-1])
    # indspeed = smoothM(indspeed, 5)
    # plt.plot(indspeed)

    ForceR = np.zeros(len(indentationfull))  # Lee_Radok force prediction
    ForceT = np.zeros(len(indentationfull))  # Ting's force prediction

    # for i in range (1, PointN-1):  # MaxInd or PointN-1
    #     ndx = np.asarray(range (0, i+1))  # integration limits, dummy variable
    #     ForceR[i]= K1 * (BEC[i]*np.trapz(Et[i-ndx]*ind2speed[ndx], time[ndx]) + 
    #                      power*eta*indentationfull[i]**(power-1)*indspeed[i]*BECspeed[i])

    for i in range(1, PointN-1):  # MaxInd or PointN-1
        ForceR[i] = K1 * (BEC[i]*np.trapz(Et[i::-1]*ind2speed[:i+1], dx=dT))

    ForceT[:MaxInd]=ForceR[:MaxInd]
    # plt.plot(indentationfull,ForceR)

    cntrad = K12*indentationfull**(power-1)  # contact radius
    cntrad[MaxInd:] = 0

    # retraction part calculation=============================================
    t1_ndx = np.zeros(len(time), dtype=int)
    endofalgorithm2 = len(time)
    b = MaxInd-1  # upper limit for max search

    for i in range(MaxInd, endofalgorithm2-2):  # force for retraction part
        res2 = np.zeros(len(time))
        localend = 0

        for j in range(b, localend-1, -1):
            if localend == 0:
                # ndx = np.asarray(range (j, i+1))
                # res2[j] = np.trapz(Et[i-ndx]*indspeed[ndx], time[ndx]) + eta*indspeed[i]
                res2[j] = np.trapz(Et[i-j::-1]*indspeed[j:i+1], dx=dT)
                if res2[j] > 0:
                    localend = j

        if abs(res2[localend]) <= abs(res2[localend+1]):
            Imin = localend
        else:
            Imin = localend+1

        if Imin > MaxInd+1:
            t1_ndx[i] = Imin-1
            # print("position1")  # check of trigger position
        elif (Imin <= 1):
            t1_ndx[i] = Imin
            t1_ndx[i+1] = 1
            endofalgorithm2 = i
            cntrad[i] = cntrad[t1_ndx[i]]
            cntrad[i+1] = cntrad[t1_ndx[i+1]]
            # print("position2")
            # print(i)
            break
        else:
            b = Imin
            t1_ndx[i] = Imin
            endofalgorithm2 = PointN-1
            # print("position3")

        cntrad[i] = cntrad[t1_ndx[i]]  # a is recalculated
        # indentationfull2[i] = indentationfull[t1_ndx[i]]  # effective indentation

        # ndx = np.asarray(range (0, t1_ndx[i]+1))
        ijk = t1_ndx[i]
        if ijk == i:
            ijk = i-1
        # ForceT[i] = K1 * (BEC[ijk]*np.trapz(Et[i-ndx]*ind2speed[ndx], time[ndx]))
        # ForceT[i] = K1 * (BEC[ijk]*np.trapz(Et[i:i-ijk-1:-1]*ind2speed[:ijk+1], dx=dT))
        ForceT[i] = K1 * (BEC[ijk]*np.trapz(Et[i:i-ijk-1:-1]*ind2speed[:ijk+1], dx=dT))
    cntrad = cntrad[0:len(indentationfull)]
    contact_time = endofalgorithm2*dT
    # plt.plot(indentationfull,Force2)  # linestyle=':'
    return ForceT, cntrad, contact_time, t1_ndx, Et, ForceR

poisson_ratio = 0.5
tip_angle = 35.0
v0t = 1e-06
v0r = 1e-06
t0 = 1
E0 = 2500
betaE = 0.19
alpha = 0.19
f0 = 0
slope = 0

tm = 1
tf = 2
ttc = np.linspace(0, tm, 100)
trc = np.linspace(tm, tf, 100)

time = np.r_[ttc, trc]

d0 = 0
dTp = 1
dT = time[1] - time[0]
indentation = np.piecewise(time, [time <= tm, time >= tm], [lambda t: t * v0t, lambda t: (tf-t) * v0r])

MaxInd = indentation.argmax() + 1

ForceT, cntrad, contact_time, t1_ndx, Et, ForceR =\
    ting_numerical(E0, alpha, dTp, poisson_ratio, tip_angle, dT, MaxInd, "pyramid", indentation)

f_num = ttingtest(indentation, d0, f0, slope, E0, alpha, dT, 0, t0, time, simple_power_law, "pyramid", tip_angle, poisson_ratio)

f_anal = ting_analytical_cone(time, betaE, E0, slope, f0, tm, t0, v0r, v0t, "pyramid", tip_angle, poisson_ratio)

plt.plot(indentation, ForceT)
plt.plot(indentation, f_num)
plt.plot(indentation, f_anal)
plt.show()