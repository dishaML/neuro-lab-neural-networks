import numpy as np

class HopfieldNetwork:
    def __init__(self, n_units):
        self.n_units = n_units
        self.weights = np.zeros((n_units, n_units))

    def train(self, patterns):
        for p in patterns:
            p = p.reshape(self.n_units, 1)
            self.weights += p @ p.T
        np.fill_diagonal(self.weights, 0)

    def recall(self, pattern, steps=5):
        s = pattern.copy()
        for _ in range(steps):
            for i in range(self.n_units):
                raw = np.dot(self.weights[i], s)
                s[i] = 1 if raw >= 0 else -1
        return s
