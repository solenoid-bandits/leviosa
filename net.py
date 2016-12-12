#import tensorflow as tf
import numpy as np

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))
def d_sigmoid(x):
    s = sigmoid(x)
    return s*(1.0-s)

transfer = sigmoid
d_transfer = d_sigmoid 

# Learning on-the-fly
class Layer(object):
    def __init__(self):
        pass
    def apply(self, X):
        pass

class DenseLayer(Layer):
    def __init__(self, i, o):
        super(DenseLayer,self).__init__()
        self.W = np.random.randn((i,o))
        self.B = np.random.randn((1,o))

        self.O = np.zeros((1,o))
        self.G = np.zeros((1,o))

    def setup(self, prv, nxt):
        # aware of previous and next layers
        self.prv = prv
        selv.nxt = nxt

    def apply(self, X):
        #self.I = X.copy()
        self.I = X
        self.O = np.dot(X, self.W) + self.B
        return self.O

    def compute_gradient(self, T):
        if nxt is None:
            G = T - self.O
        else:
            G = self.nxt.G

        self.G = np.dot(G, self.W.T) * d_transfer(self.prv.O)

    def apply_gradient(self):
        dW = np.dot(self.I.T, self.G)
        self.W += 0.6 * dW
        self.B += self.G

class OutputLayer(DenseLayer):
    def __init__(self, i, o):
        super(OutputLayer,self).__init__(i,o)

    def compute_gradient(self, T, G):
        # specialize on computing gradient
        self.G = T - self.O

    def apply_gradient(self):
        dW = np.dot(self.I.T, self.G)
        self.W += 0.6 * dW
        self.B += self.G

class Net(object):
    def __init__(self, dims):
        self.dims = dims

    def train(self):
        pass
    def predict(self):
        pass
