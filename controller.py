import numpy as np
from matplotlib import pyplot as plt
import Net from net

class Controller(object):
    def __init__(self):
        pass
    def current(self, pos, target, dt):
        # t = time
        return (target - pos) * 1.0 # proportional-ish.

class PIDController(Controller):
    def __init__(self, k_p=1.0, k_i=0.0, k_d=0.0, t=0.0):
        super(PIDController,self).__init__()
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d

        self.e_i = 0
        self.e_d = 0
        self.res = 0.0

    def current(self, pos, target, dt):
        # t = time
        if dt == 0:
            return self.res

        k_p, k_i, k_d = self.k_p, self.k_i, self.k_d

        err = target - pos
        self.e_i += err * dt
        self.res = k_p * err + k_i * self.e_i + k_d * (err - self.e_d) / dt;
        self.e_d = err # remember last error

        return self.res

class GradientController(Controller):
    # Controller Based on Gradient Descent
    def __init__(self):
        self.net = NeuralNet()
        pass
    def current(self, pos, target, t):
        # backprop
        # output
        pass

if __name__ == "__main__":
    T_START = 0.0
    T_END = 200 * np.pi
    T_N = 10000

    ctrl = PIDController(1.0, 0.1, 0.15)
    target_pos = 1.0
    pos = 0.0

    ts = np.linspace(T_START, T_END, T_N)
    cs = []
    ps = []

    for i in range(T_N):
        dt = ts[i] - ts[i-1] if i>0 else 0.0
        c = ctrl.current(pos,target_pos,ts[i] - ts[i-1])
        pos -= np.sin(c)
        ps.append(pos)
        cs.append(c)

    plt.plot(ts, ps)
    plt.plot(ts, [target_pos for _ in ts])
    plt.plot(ts, cs)

    plt.legend(['position','target','current'])
    ax = plt.gca()
    plt.show()