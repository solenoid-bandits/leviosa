# Units : SI Units
import numpy as np
import scipy
from scipy.integrate import quad as integrate
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from controller import PIDController

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
    return np.atleast_2d(args).T.astype(np.float32)

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
    def __init__(self):
        self.orientation = Orientation()
        #self.geometry = geometry
        #self.density = density
        #self.mass = density * geometry.volume()
    def apply_force(self, f):
        pass
        #self.orientation.lin_acc += (f / self.mass)
        #self.oritentation.ang_acc += ...
    def update(self, dt):
        pass
        #self.orientation.update(dt)

class Magnet(object):
    def __init__(self):
        super(Magnet,self).__init__()
        pass
    def field(self, current, position):
        pass

class Hysteresis(object):
    def __init__(self):
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
        DeltaH = 0.7
        # Values below 2 reflect a hard ferromagnet - (high coercivity, lower saturation mag)
        # Values above 4 reflect a soft ferromagnet - (low coercivity, higher sat mag)
        Nfirst = 2500 # initial magnetization curve range (DO NOT CHANGE as it is basically a vertical axis offset)
        Ndown = 5000
        Nup = 5000
        # Value saves the magnetic field strength into the right magnitude
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

        # For plot debugging - disabled by default
        #plt.xlabel('Applied magnetic field H (A/m)')
        #plt.ylabel('Magnetization M (MA/m)')
        #plt.plot(H, M)

        # reducing anhysteric magnetization range to upper/lower curve values
        startAn = Nfirst + Ndown
        endAn = Nfirst + Nup
        end = startAn + Ndown
        M_up = M[Nfirst:endAn]

        # Interpolation curve - added in v1.3
        H_an2 = H[startAn:end] #FLIPPED IT!
        M_up2 = M_up[::-1]
        self.polation = interp1d(H_an2, M_up2)
        self.max_M = max(M_up2)

    def M(self, H):
        #if polation(val) > self.max_H:
        #    returnMag = max(polation(M_up2))/pow(10,6)
        #else:
        
        # Squash
        if H >= 1700:
            H = 1700
        elif H <= -1700:
            H = -1700

        m = self.polation(H)
        if m > self.max_M:
            return self.max_M
        else:
            return m
        #print returnMag, "MA/m" #shows you the magnetization value being returned
        #plt.plot(val, polation(val),'or',H_an2, polation(H_an2),'-')
        #plt.show()

class Levitron(Magnet):
    # The object to levitate
    # Assumed to be neodymium, cylindrical
    def __init__(self, geometry, density):
        super(Levitron, self).__init__()
        self.geom = geometry
        self.density = density
        self.volume = geometry.volume()
        self.mass = self.volume * density
        self.hysteresis_prop = Hysteresis()
    def apply_force(self, f):
        pass
    def update(self,dt):
        pass
    def hysteresis(self, H):
        m = self.hysteresis_prop.M(H)
        m *= self.volume
        return m

    def force(self, solenoid, position):

        def BM(pos):
            B = solenoid.field(pos)[2] # Z component only
            H = B/mu0
            M = self.hysteresis(H)
            return B*M

        delta = .0005

        cur = BM(position)
        prv = BM(position - vec(0,0,delta))
        #nxt = BM(position + vec(0,0,delta))

        d1 = (cur - prv) / delta 
        #d2 = (nxt - cur) / delta
        return d1

class Solenoid(Magnet):
    def __init__(self, radius, length, loops):
        # Solenoid with origin at its center
        super(Solenoid, self).__init__()
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
        dx = - self.radius * 2 * pi * self.loops * np.sin(2*pi * self.loops * t)
        dy = self.radius * 2 * pi * self.loops * np.cos(2*pi*self.loops*t)
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
        B = mu0/(4*pi) * self.current * Bi
        return B

class Model(object):
    def __init__(self):

        geom = CylinderGeometry(0.015, 0.02) # r 10cm, h 10cm
        self.levitron = Levitron(geom, 2000) # density in kg/m^3 of neodymium is 7000, but reduced to 2000

        self.solenoid = Solenoid(.1,0.15,300.0) # radius, length, loops
        self.solenoid.set_current(5.0) #5 Amps

        self.reset()

        self.position = vec(0,0,-0.02)
        self.target = -0.03

        self.velocity = vec(0,0,0)

        self.ctrl = PIDController(70,0.3,2.0)

    def update(self, dt):
        g_f = -9.8 * self.levitron.mass
        current = self.ctrl.current(self.target - self.position[2], dt)
        self.solenoid.set_current(current)
        m_f = self.levitron.force(self.solenoid,self.position)
        f = vec(0, 0, m_f + g_f)

        self.position += self.velocity * dt
        self.velocity += (f / self.levitron.mass) * dt

        # params['s_r'] # solenoid radius
        # params['s_l'] # solenoid length

        # params['m_r'] # levitron radius
        # params['m_l'] # levitron length

    def reset(self):
        pass

#def apply_force(o, f, dt):
#    # f = ma
#    #o.position += o.velocity * dt
#    #o.velocity += (f/o.mass) * dt

if __name__ == "__main__":
    geom = CylinderGeometry(0.015, 0.02) # r 10cm, h 10cm
    magnet = Levitron(geom, 2000) # density in kg/m^3 of neodymium is 7000, but reduced to 2000

    solenoid = Solenoid(.1,0.15,300.0) # radius, length, loops
    solenoid.set_current(5.0) #5 Amps

    Bs = []
    forces = []

    initial_position = vec(0, 0, -0.02) # arbitrary starting position, 2cm below solenoid
    position = initial_position.copy()
    target_pos = -0.03
    velocity = vec(0,0,0)

    gravity = -9.8 * magnet.mass

    print(magnet.force(solenoid,position))
    print 'gravity', gravity

    ctrl = PIDController(70,0.3,2.0) #k_p, k_i, k_d

    T_START = 0.0
    T_END = 5.0
    T_N = 1000
    ts = np.linspace(T_START, T_END, T_N)
    
    cs = []
    fs = []
    ps = []
    vs = []

    for i in range(T_N):
        dt = (ts[i] - ts[i-1]) if i>0 else 0.0
        #current = ctrl.current(-velocity[2],dt) #counteract velocity
        current = ctrl.current(target_pos - position[2],dt) #counteract velocity
        solenoid.set_current(current)
        magnetic_force = magnet.force(solenoid,position)
        net_force = vec(0, 0, magnetic_force + gravity)
        print 'net force', net_force[2][0]
        print 'magnetic force', magnetic_force
        print 'velocity', velocity[2][0]
        print 'current', current

        position += velocity * dt 
        velocity += (net_force / magnet.mass) * dt # f=ma, a=f/m

        fs.append(net_force[2][0])
        cs.append(current)
        vs.append(velocity[2][0])
        ps.append(position[2][0])

    plt.suptitle('PID Control of Magnetic Levitation System')

    plt.subplot(2,2,1)
    plt.plot(ts, fs)
    plt.title('Force (N)')
    plt.subplot(2,2,2)
    plt.plot(ts, cs)
    plt.title('Current (A)')
    plt.subplot(2,2,3)
    plt.plot(ts, vs)
    plt.title('Velocity (m/s)')
    plt.subplot(2,2,4)
    plt.plot(ts, ps)
    plt.title('Position (m)')

    plt.show()
    m = Model(solenoid, magnet)
