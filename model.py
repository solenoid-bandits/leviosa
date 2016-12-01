# Units : SI Units
import numpy as np
from scipy.integrate import ode

mu0 = 4e-7 * np.pi

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
    def field(self, current, position):
        # depends on H-Field
        pass

class Solenoid(Magnet):
    def __init__(self, radius, length, loops, pose=None):
        # Solenoid with origin at its center
        super(Solenoid, self).__init__(pose)
        self.radius = radius
        self.length = length
        self.loops = loops
        self.current = 0

    def pos(self, t):
        # Parametrized position of the wire
        # t = 0.0 ~ 1.0
        # at t = 1.0, theta = 2*pi*self.loops
        x = self.radius * sin(2 * np.pi * self.loops * t)
        y = self.radius * cos(2 * np.pi * self.loops * t)
        z = self.length / self.loops * l
        return vec(x,y,z)

    def set_current(self, current):
        self.current = current

    def field(self, position):
        # Biot-Savart Law
        pos_local = position
        # TODO : actually perform conversion
        # pos_local = translate(rotate(position))

        # B(pos)
        # dp = pos_local - pos(t)
        # B = mu0/4*pi * self.current * Integrate(Cross(differentiate(pos(t), t), dp) / abs(dp)**3, t)
        # return B

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
    solenoid = Solenoid(0.025, 0.5)
    magnet = Levitron()
    m = Model(solenoid, magnet)
