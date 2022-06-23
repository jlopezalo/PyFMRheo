import matplotlib.pyplot as plt
import numpy as np
from pyafmrheo.models.ting_numerical import ting_numerical
from pyafmrheo.models.ting_analytical import *

def compute(ind, time, half_angle, E0, t0, fluidity_exp):
    # print(f"Params: {[ind, time, half_angle, E0, t0, fluidity_exp, height, bec]}")
    # coeff = (8 * np.tan(half_angle)) / (3 * np.pi)
    poisson_ratio = 0.5
    coeff = np.pi * (1 - poisson_ratio ** 2) / (2 * np.tan(half_angle))
    MaxIndIdx = int(np.round(len(ind)/2)) - 1
    v0 = ind[MaxIndIdx] / time[MaxIndIdx]
    eta = 0
    dT = time[1] - time[0]
    print(f"Coef Garcia analytical: {coeff}")
    # print(ind == v0 * time)
    # print(ind)
    # print(v0 * time)
    plt.plot(time, ind, label="Indentation Simulated")
    ind_computed = np.r_[v0 * time[:MaxIndIdx], v0 * (time[MaxIndIdx:] - time[MaxIndIdx])[::-1]]
    plt.plot(time, ind_computed, label="Indentation Computed")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Indentation [m]")
    plt.show()
    # A = coeff * E0 * v0 * time ** (2 - fluidity_exp)
    # B = garcia_viscoelastic_bec_factors(ind, half_angle, height, fluidity_exp)
    
    tmax = time[MaxIndIdx]

    # Janshoff approach / Leda's implementation
    Ajans_app = v0 ** 2 / coeff * t0 ** fluidity_exp
    Bjans_app = 1 / (2 - 3 * fluidity_exp + fluidity_exp ** 2)
    Fjans_app = Ajans_app * E0 * Bjans_app * 2 * (time[:MaxIndIdx] ** (2 - fluidity_exp))
    # Fjans_app_2 = janshoff_approach_cone(time[:MaxIndIdx], poisson_ratio, half_angle, v0, t0, E0, fluidity_exp, tmax)
    # print(Fjans_app.all() == Fjans_app_2.all())

    # Janshoff retract / Screenshot
    # tmax, tret, beta, t1, v0, coeff, t0, E0 = symbols("tmax tret beta t1 v0 coeff t0 E0")
    tmax = time[MaxIndIdx]
    tret = time[MaxIndIdx:]
    beta = fluidity_exp
    t1 = time[MaxIndIdx:] - 2 ** (1 / (1 - beta)) * (time[MaxIndIdx:] - tmax)
    # t1 = time[MaxIndInd:] - np.power(1 + (v0 / v0), (1/(1-beta))) * (time[MaxIndInd:] - tmax)
    Ajans_ret = v0 ** 2 / coeff * t0 ** beta
    A = (tret ** (1-beta) * (tret + 2 * (-2 + beta) * tmax) - (tret - t1) ** (1 - beta) * (tret + t1 - beta * t1 + 2 * (-2 + beta) * tmax))
    B = (-2 + beta) * (-1 + beta)
    Bjans_ret = A/B
    Fjans_ret = Ajans_ret * E0 * Bjans_ret * 2 * ((tret - tmax) ** (2 - fluidity_exp))
    # Fjans_ret_2 = janshoff_retract_not_simplified_cone(time[MaxIndIdx:], poisson_ratio, half_angle, v0, t0, E0, beta, tmax)
    # print(Fjans_ret.all() == Fjans_ret_2.all())
    # Fjans = np.r_[Fjans_app_2, Fjans_ret_2[::-1]]
    # Fjans2 = np.r_[Fjans_app_2, Fjans_ret[::-1]]

    # Leda's janshodd approach
    a = v0 ** 2 * E0 * t0 ** fluidity_exp / coeff
    b = 1 / (2 - 3 * fluidity_exp + fluidity_exp ** 2)
    Fleda_app = a * b * 2 * time[:MaxIndIdx] ** (2 - fluidity_exp)
    # Fleda_app_2 = janshoff_approach_cone(time[:MaxIndIdx], poisson_ratio, half_angle, v0, t0, E0, beta, tmax)
    # print(Fleda_app.all() == Fleda_app_2.all())

    # Leda's janshoff retract
    a = v0 ** 2 * E0 * t0 ** fluidity_exp / coeff
    b = 1 / (2 - 3 * fluidity_exp + fluidity_exp ** 2)
    Fleda_ret = a * b * 2 * (time[MaxIndIdx:] - time[MaxIndIdx]) ** (2 - fluidity_exp)
    # Fleda_ret_2 = janshoff_retract_cone(time[MaxIndIdx:], poisson_ratio, half_angle, v0, t0, E0, beta, tmax)
    # print(Fleda_ret.all() == Fleda_ret_2.all())
    # Fleda = np.r_[Fleda_app_2, Fleda_ret_2]

    coeff = (8 * np.tan(half_angle)) / (3 * np.pi)
    print(f"Coeff Garcia: {coeff}")
    AGar = coeff * 2 * E0 * ind ** 2 * (time/t0) ** -fluidity_exp
    BGar = 1 / (2 - 3 * fluidity_exp + fluidity_exp ** 2)
    FGar = AGar * BGar
    # FGar2 = garcia_viscoelastic_cone(ind, time, half_angle, v0, v0, t0, E0, beta, tmax)

    # print(FGar.all() == FGar2.all())

    # Ting numerical
    FTing, _, _, _, _, _ = ting_numerical(E0, fluidity_exp, t0, eta, poisson_ratio, half_angle, dT, MaxIndIdx, "cone", ind)

    # plt.plot(Fjans, label="Jans")
    # plt.plot(Fleda, label=f"Leda Analytical: E0={E0}, beta={fluidity_exp}")
    # plt.plot(Fjans2, label=f"Janshoff Analytical t-tmax: E0={E0}, beta={fluidity_exp}")
    # plt.plot(Fjans,label=f"Janshoff Analytical t: E0={E0}, beta={fluidity_exp}")
    plt.plot(FTing,label=f"Efremov Numerical: E0={E0}, beta={fluidity_exp}")
    plt.legend()
    plt.show()

    # plt.plot(ind, Fleda, label=f"Leda Analytical: E0={E0}, beta={fluidity_exp}")
    # plt.plot(ind, Fjans, label=f"Janshoff Analytical: E0={E0}, beta={fluidity_exp}")
    plt.plot(ind, FGar,label=f"Garcia Analytical: E0={E0}, beta={fluidity_exp}")
    plt.plot(ind, FTing,label=f"Efremov Numerical: E0={E0}, beta={fluidity_exp}")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Parameters for simulation based on article
    # Reference: https://doi.org/10.1039/D1NR03894J
    half_angle = np.radians(35)
    E0 = 2500
    t0 = 1
    fluidity_exp = 0.19
    eta = 0
    height = 4.5 * 1e-6
    poisson_ratio = 0.5

    # Define V0
    v0 = 1 * 1e-6 # m/s

    # Define indentation
    indapp = v0 * np.linspace(0, 1, 100)
    indret = v0 * np.linspace(1, 0, 100)
    indentationfull = np.r_[indapp, indret[1:]]

    # Get time
    t = np.linspace(0, 2, len(indentationfull))

    # Get delta t
    dT = t[1] - t[0]

    # Get index maximum indentation
    MaxIndIdx = int(np.round(len(indentationfull)/2)) - 1

    # Get tmax t2=t1
    tmax = t[MaxIndIdx]
    
    # Simulate
    compute(indentationfull, t, half_angle, E0, t0, fluidity_exp)