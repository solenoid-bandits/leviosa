# Units : SI Units
import numpy as np
import scipy
from scipy.integrate import quad as integrate
from matplotlib import pyplot as plt

pi = np.pi
mu0 = 4e-7 * pi

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
        # magnetic susceptibility of magnet
        self.x_M = 1
        pass
    def field(self, current, position):
        # depends on H-Field
        # NOTHING HERE omg
        # can't proceed without defining field for permanent magnet
        pass
    # Magnetic Field Strength, H:
    def fieldStrength(self, current, position):
    	return self.field(current, position) / self.x_M
    def magMoment(self, fieldStrength):
    	return self.x_M * self.fieldStrength()

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

	def force(self, position, magMoment_val):
		# Taking field from the above method
		field = self.field()
		magMoment = Levitron.magMoment()
		return np.gradient(field * magMoment)


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
    solenoid = Solenoid(1.0,0.01,1.0)
    solenoid.set_current(1.0)

    Bs = []
    Fs = []
    for i in range(100):
        z = i * 0.01
        B = solenoid.field(vec(0,0,z))
        fstr = magnet.fieldStrength(1.0,vec(0,0,z))
        magMoment_val = magnet.magMoment(fstr)
        F = Solenoid.force(vec(0,0,z), magMoment_val)
        Bs.append(B[2])
        Fs.append(F[2])
    print Bs
    print Fs
    plt.plot(Bs)
    plt.show()

    m = Model(solenoid, magnet)

    plt.plot(Fs)
    plt.show()
