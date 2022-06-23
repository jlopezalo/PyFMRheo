import numpy as np
import matplotlib.pyplot as plt
from pyafmrheo.models.ting_analytical import ting_analytical_cone
from pyafmrheo.models.ting_numerical import ting_numerical, simple_power_law

# Parameters for simulation based on article
# Reference: https://doi.org/10.1039/D1NR03894J
half_angle = 35
E0 = 2500
t0 = 1
fluidity_exp = 0.19
eta = 0
height = 4.5 * 1e-6
poisson_ratio = 0.5
slope=0
f0=0
d0 = 0

# Define V0
v0t = 4.9999999999999996e-06 * 1e-6 # m/s
v0r = 4.9999999999999996e-06 * 1e-6 # m/s

# Define indentation
indapp = v0t * np.linspace(0, 1, 100)
indret = v0r * np.linspace(1, 0, 100)
indentationfull = np.r_[indapp, indret[1:]]

# Get time
t = np.linspace(0, 1.1996000000000002, len(indentationfull))

# Get delta t
dT = t[1] - t[0]

# Get index maximum indentation
MaxIndIdx = int(np.round(len(indentationfull)/2)) - 1

# Get tmax t2=t1
tmax = t[MaxIndIdx]

tmax = 0.2774

plt.plot(t, indentationfull)
plt.show()

f_anal = ting_analytical_cone(t, 0.16891065, 249.143680, slope, f0, tmax, t0, v0r, v0t, "pyramid", half_angle, poisson_ratio)

f_num = ting_numerical(indentationfull, d0, f0, slope, 142.590142, 0.29753852, dT, 0, t0, t, simple_power_law, "pyramid", half_angle, poisson_ratio)

plt.plot(t, f_anal, label="ting_analytical")
plt.plot(t, f_num, label="ting_numerical")
plt.legend()
plt.show()