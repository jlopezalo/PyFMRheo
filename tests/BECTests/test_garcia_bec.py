# Imports
import sys
sys.path.insert(0, '../../src')
import matplotlib.pyplot as plt
import numpy as np
from analysis.models.bec import bec_garcia_garcia

# Simulation params in IS
# Parameters extracted from Figure 3.
# Ref: https://doi.org/10.1016/j.bpj.2018.05.012
x = np.arange(0, 1000, 10)  # n.u
indentation = x * 1e-9      # m
E = 4 * 1e3                 # Pa
h1 = 2.5 * 1e-6             # m
h2 = 5 * 1e-6               # m
h3 = 10 * 1e-6              # m
r = 5 * 1e-6                # m

def garcia_model(indentation, E, h, r):
    # In the paper they use another expression for F0, assuming v = 0.5
    # Ref: https://doi.org/10.1016/j.bpj.2018.05.012
    f0 = 16/9 * E * np.sqrt(r) * np.power(indentation, 3/2)
    # Compute coefficients using the library method
    coeffs = bec_garcia_garcia(h, indentation, "paraboloid", r)
    return f0 * coeffs

# Compute forces at different heights
force1 = garcia_model(indentation, E, h1, r)
force2 = garcia_model(indentation, E, h2, r)
force3 = garcia_model(indentation, E, h3, r)

# Plot at different heights
plt.plot(indentation * 1e6, force1*1e9, label=f"{h1 * 1e6} um")
plt.plot(indentation * 1e6, force2*1e9, label=f"{h2 * 1e6} um")
plt.plot(indentation * 1e6, force3*1e9, label=f"{h3 * 1e6} um")
plt.xlabel("Indentation [um]")
plt.ylabel("Force [nN]")
plt.title("BEC Simulations at different heights")
plt.legend()
plt.show()