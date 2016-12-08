# Hysteresis module by Anderson Ang
# Inspired by the Jiles-Atherton Model, 1984

import numpy as np
from matplotlib import pyplot as plt

# Permittivity of free space
mu0 = 4 * np.pi * 1e-7 # H/m

# Scale of mag field strength
a = 470 # A/m
# Mean field param
alpha = 9.38e-4
# Weighting of anhysteric vs irreversible magnetization (1 removes Mrev)
c = 0.0889
# Sizing of hysteresis
k = 483 # A/m
# Saturation magnetization
Ms = 1.48e6 # A/m

H = [0]
delta = [0]
Man = [0]
dMirrdH = [0]
Mirr = [0]
M = [0]

# Tracks change of ext field H (permeance) to magnetization
DeltaH = 10
# Values below 20 reflect a hard ferromagnet - (high coercivity, lower saturation mag)
# Values above 40 reflect a soft ferromagnet - (low coercivity, higher sat mag)
Nfirst = 125 # initial magnetization curve range (DO NOT CHANGE as it is basically a vertical axis offset)
Ndown = 250
Nup = 250
val = 250 # sample value of H applied field (A/m)

for i in range(Nfirst):
    H.append(H[i] + DeltaH)

for i in range(Ndown):
    H.append(H[-1] - DeltaH)

for i in range(Nup):
    H.append(H[-1] + DeltaH)

delta = [0]
# For x > 1, L(x) = 1
# For x < -1, L(x) = -1
for i in range(len(H) - 1):
    if H[i + 1] > H[i]:
        delta.append(1)
    else:
        delta.append(-1)

# From Wikipedia - Langevin Function
# x/3 effectively for all values absolute (-1 <= x <= 1)
def L(x):
    return (np.cosh(x) / np.sinh(x)) - (1 / x)

for i in range(Nfirst + Ndown + Nup):
    Man.append(Ms * (1 / np.tanh((H[i + 1] + alpha * M[i]) / a) - a / (H[i + 1] + alpha * M[i])))
    dMirrdH.append((Man[i+1] - M[i]) / (k * delta[i+1] - alpha * (Man[i + 1] - M[i])))
    Mirr.append(Mirr[i] + dMirrdH[i + 1] * (H[i+1] - H[i]))
    M.append(c * Man[i + 1] + (1 - c) * Mirr[i + 1])
    if (H[i] == val):
        B_field = H[i]*mu0*(1000.0)
        print str(B_field) + ' Teslas'

plt.xlabel('Applied magnetic field H (A/m)')
plt.ylabel('Magnetization M (MA/m)')
plt.plot(H, M)
plt.show()
