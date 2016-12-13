"""
Hysteresis module by Anderson Ang
V1.2
Changelog
- Refined delta function to have a step value of 1
- Added plot points to find specific M value for a given H (on all curves)
- Added anhysteric array splices and added polyfit (numpy) for higher M accuracy
Inspired by the Jiles-Atherton Model, 1984
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

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
DeltaH = 1
# Values below 20 reflect a hard ferromagnet - (high coercivity, lower saturation mag)
# Values above 40 reflect a soft ferromagnet - (low coercivity, higher sat mag)
Nfirst = 1250 # initial magnetization curve range (DO NOT CHANGE as it is basically a vertical axis offset)
Ndown = 2500
Nup = 2500
val = 523 # sample value of H applied field (A/m)

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
        data_x = [float(H[i])]
        data_y = [float(M[i])]
        print data_y
        print str(B_field) + ' Teslas'
        plt.plot(data_x, data_y, 'or')

plt.xlabel('Applied magnetic field H (A/m)')
plt.ylabel('Magnetization M (MA/m)')
plt.plot(H, M)

# reducing anhysteric magnetization range to upper curve values
startAn = Nfirst + Ndown
endAn = Nfirst + Nup
end = startAn + Ndown
M_up = M[Nfirst:endAn]
H_an = H[Nfirst:endAn]

# Polyfit curve - added in v1.2
polynomial = np.polyfit(H_an,M_up, 4)
p = np.poly1d(polynomial)

# Interpolation curve - added in v1.3
H_an2 = H[startAn:end] #FLIPPED IT!
M_up2 = M_up[::-1] #flipped it twice! woohoo!
#print 'max',  max(H_an2)
#print 'H_AN2', H_an2
print 'H_AN', H_an
polation = interp1d(H_an2, M_up2)

plt.plot(H_an, p(H_an),'o', H_an2, polation(H_an2),'--')
plt.show()
