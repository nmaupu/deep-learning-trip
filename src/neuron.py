import numpy as np


class Neuron:
    """A simple neuron"""

    def __init__(self, W, b):
        self.W = W
        self.b = b

    def predict(self, X):
        Z = X.dot(self.W) + self.b
        A = 1 / (1 + np.exp(-Z))
        return A >= 0.5, A
