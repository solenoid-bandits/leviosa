# Units : SI Units
import numpy as np
import scipy
from scipy.integrate import quad as integrate
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

pi = np.pi
# Permittivity of free space
mu0 = 4e-7 * pi # H/m
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

def vec(*args):
    return np.atleast_2d(args).T

def R(x,y,z):
    # Rotation Matrix
    np.matrix()

class Pose(object):
    def __init__(self):
        self.position = vec(0,0,0)
        self.rotation = vec(0,0,0)

class Magnet(object):
    def __init__(self, pose=None):
        if pose:
            self.pose = pose
        else:
            self.pose = Pose()
    def field(self, current, position):
        pass

class Levitron(Magnet):
    # The object to levitate
    # Assumed to be neodymium, cylinder
    def __init__(self, pose=None):
        super(Levitron, self).__init__(pose)
        pass
    # def field(self, current, position):
    #     # depends on H-Field
    #     pass
    def hysteresis(self, magFieldStr): 
        '''
        given H (magnetic field strength, returns M, which I think is magnetic saturation)
        '''
        delta = [0]
        Man = [0]
        dMirrdH = [0]
        Mirr = [0]
        M = [0]
        H = [0]

        # Tracks change of ext field H (permeance) to magnetization
        DeltaH = 1
        # Values below 2 reflect a hard ferromagnet - (high coercivity, lower saturation mag)
        # Values above 4 reflect a soft ferromagnet - (low coercivity, higher sat mag)
        Nfirst = 1250 # initial magnetization curve range (DO NOT CHANGE as it is basically a vertical axis offset)
        Ndown = 2500
        Nup = 2500
        # switch to magField when complete
        val = 523 # sample value of H applied field (kA/m) (actual x 10^(-3))

        for i in range(Nfirst):
            H.append(H[i] + DeltaH)

        for i in range(Ndown):
            H.append(H[-1] - DeltaH)

        for i in range(Nup):
            H.append(H[-1] + DeltaH)

        for i in range(len(H) -1):
            if H[i + 1] > H[i]: # determines the direction of movement
                delta.append(1)
            else:
                delta.append(-1)

        def L(x):
            return (np.cosh(x) / np.sinh(x)) - (1 / x)

        for i in range(Nfirst + Ndown + Nup):
            Man.append(Ms * (1 / np.tanh((H[i + 1] + alpha * M[i]) / a) - a / (H[i + 1] + alpha * M[i])))
            dMirrdH.append((Man[i+1] - M[i]) / (k * delta[i+1] - alpha * (Man[i + 1] - M[i])))
            Mirr.append(Mirr[i] + dMirrdH[i + 1] * (H[i+1] - H[i]))
            M.append(c * Man[i + 1] + (1 - c) * Mirr[i + 1])
            if (H[i] == val): # only detects integers/whole numbers
                B_field = H[i]*mu0*(1000.0)
                data_x = [float(H[i])]
                data_y = [float(M[i])]
                # print data_y
                # print str(B_field) + ' Teslas'
                # plt.plot(data_x, data_y, 'or')
        # Commenting this section out because I know it works
        # plt.xlabel('Applied magnetic field H (A/m)')
        # plt.ylabel('Magnetization M (MA/m)')
        # plt.plot(H, M)
        mag_saturation =  max(M)/pow(10,6)
        

        # reducing anhysteric magnetization range to upper/lower curve values
        startAn = Nfirst + Ndown
        endAn = Nfirst + Nup
        end = startAn + Ndown
        M_up = M[Nfirst:endAn]
        # for polyfitting ()
        H_an = H[Nfirst:endAn]

        # Copied from hysteresis.py:
        polynomial = np.polyfit(H_an,M_up, 4)
        p = np.poly1d(polynomial)

        # Interpolation curve - added in v1.3
        H_an2 = H[startAn:end] #FLIPPED IT!
        M_up2 = M_up[::-1]
        polation = interp1d(H_an2, M_up2)
        # plt.plot(H_an, p(H_an),'o', H_an2, polation(H_an2),'--')
        # plt.show()
        return mag_saturation #just for now
    def force(self, solenoid, position):
        # B_values = solenoid.field(position)
        # B = B_values[2]
        # H = B/mu0
        # M = hysteresis(H)
        # BdotM = B*M
        B_beforevalues = solenoid.field(position - vec(0,0,.005))
        B_before = B_beforevalues[2]
        print "B before: " + str(B_before)
        H_before = B_before/mu0
        M_before = self.hysteresis(H_before)
        BdotM_before =B_before * M_before

        B_aftervalues = solenoid.field(position - vec(0,0,.005))
        B_after = B_aftervalues[2]
        print "B after: " + str(B_after)
        H_after = B_after/mu0
        M_after = self.hysteresis(H_after)
        BdotM_after = B_after * M_after

        return (BdotM_after - BdotM_before)/.01

class Solenoid(Magnet):
    def __init__(self, radius, length, loops, pose=None):
        # Solenoid with origin at its center
        super(Solenoid, self).__init__(pose)
        self.radius = radius
        self.length = length
        self.loops = loops
        self.current = 0.0

    def pos(self, t):
        # Parametrized position of the wire
        # t = 0.0 ~ 1.0
        # at t = 1.0, theta = 2*pi*self.loops
        x = self.radius * np.cos(2 * pi * self.loops * t)
        y = self.radius * np.sin(2 * pi * self.loops * t)
        z = self.length * t
        # returns column vector
        if len(t) > 1:
            return np.vstack((x,y,z))
        else:
            return vec(x, y, z)

    def d_pos(self,t):
        dx = - self.radius * 2 * pi * np.sin(2*pi * self.loops * t)
        dy = self.radius * 2 * pi * np.cos(2*pi*self.loops*t)
        dz = self.length * np.ones(t.shape)
        if len(t) > 1:
            return np.vstack((dx,dy,dz))
        else:
            return vec(dx, dy, dz)

    def set_current(self, current):
        self.current = current

    def field(self, position):
        # Biot-Savart Law
        pos_local = position

        # pos_local = translate(rotate(position))
        # doesn't matter since solenoid definition is in global coordinates

        def integrand(t):
            pos = self.pos(t)
            dp = np.subtract(pos_local, pos) # difference in position
            dpt = self.d_pos(t)
            c = np.cross(dpt, dp, axis=0)
            m = np.linalg.norm(dp, axis=0)
            res = c / (m**3)
            return res

        t = np.linspace(0.0, 1.0, 10000)
        y = integrand(t)
        Bi = np.trapz(y, t, axis=1)
        #Bi = integrate(integrand, 0.0, 1.0)
        B = mu0/4 * pi * self.current * Bi
        return B



class Model(object):
    def __init__(self,solenoid,magnet):
        self.solenoid = solenoid
        self.magnet = magnet

        self.reset()

        # params['s_r'] # solenoid radius
        # params['s_l'] # solenoid length

        # params['m_r'] # magnet radius
        # params['m_l'] # magnet length

    def reset(self):
        pass

if __name__ == "__main__":
    magnet = Levitron()
    solenoid = Solenoid(1.0,0.01,100.0)
    solenoid.set_current(1.0)

    Bs = []
    forces = []
    for i in range(100):
        z = i * 0.01
        force = magnet.force(solenoid, vec(0,0,z))
        print "force = " + str(force)
        forces.append(force)
        B = solenoid.field(vec(0,0,z))
        Bs.append(B[2])
    print Bs
    # plt.plot(Bs)
    # plt.show()
    print forces
    # plt.plot(Bs)
    # plt.show()
    m = Model(solenoid, magnet)
