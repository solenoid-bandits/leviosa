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

class Orientation(object):
    # Disabled Angular Params for now
    def __init__(self):
        self.lin_pos = vec(0,0,0)
        self.lin_vel = vec(0,0,0)
        self.lin_acc = vec(0,0,0)
        #self.ang_pos = vec(0,0,0)
        #self.ang_vel = vec(0,0,0)
        #self.ang_acc = vec(0,0,0)
    def update(self, dt):
        self.lin_pos += self.lin_vel * dt
        self.lin_vel += self.lin_acc * dt
        #self.ang_pos += self.ang_vel * dt
        #self.ang_vel += self.ang_acc * dt

class Geometry(object):
    def __init__(self):
        pass
    def volume(self):
        return 0

class CylinderGeometry(Geometry):
    def __init__(self, r, h):
        self.r = r
        self.h = h
        self._volume = pi*r*r*h
    def volume(self):
        return self._volume

class Object(object):
    # Generic Physical Object Class
    # Composed of Uniform Material
    def __init__(self, geometry, density):
        self.orientation = Orientation()
        self.geometry = geometry
        self.density = density
        self.mass = density * geometry.volume()
    def apply_force(self, f):
        self.orientation.lin_acc += (f / self.mass)
        #self.oritentation.ang_acc += ...
    def update(self, dt):
        self.orientation.update(dt)

class Magnet(object):
    def __init__(self, geometry, density):
        super(Magnet,self).__init__(geometry, density)
        pass
    def field(self, current, position):
        pass

class Levitron(Magnet):
    # The object to levitate
    # Assumed to be neodymium, cylindrical
    def __init__(self, geometry, density):
        super(Levitron, self).__init__()
        self.density = density
        self.mass = geometry.volume() * density
        pass
    def field(self, current, position):
        # depends on H-Field
        pass
    def apply_force(self, f):
        pass
    def update(self,dt):
        pass

class Solenoid(Magnet):
    def __init__(self, radius, length, loops):
        # Solenoid with origin at its center
        super(Solenoid, self).__init__(Geometry(), 0)
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

#def apply_force(o, f, dt):
#    # f = ma
#    #o.position += o.velocity * dt
#    #o.velocity += (f/o.mass) * dt

if __name__ == "__main__":
    geom = CylinderGeometry(0.1, 0.1)
    magnet = Levitron(geom, 0.5)
    solenoid = Solenoid(1.0,0.01,1.0)
    solenoid.set_current(1.0)

    Bs = []
    for i in range(100):
        z = i * 0.01
        B = solenoid.field(vec(0,0,z))
        Bs.append(B[2])
    print Bs
    plt.plot(Bs)
    plt.show()

    m = Model(solenoid, magnet)
