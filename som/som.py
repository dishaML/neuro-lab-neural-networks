import numpy as np

class SOM:
    def __init__(self, m, n, dim, learning_rate=0.5, sigma=None, n_iter=1000):
        self.m = m  # rows
        self.n = n  # cols
        self.dim = dim  # input dimension
        self.lr = learning_rate
        self.sigma = sigma if sigma else max(m, n) / 2
        self.n_iter = n_iter

        # Initialize weights randomly
        self.weights = np.random.rand(m * n, dim)

        # Grid positions
        self.locations = np.array(list(np.ndindex(m, n)))

    def _find_bmu(self, x):
        # Best Matching Unit
        distances = np.linalg.norm(self.weights - x, axis=1)
        return np.argmin(distances)

    def train(self, X):
        for t in range(self.n_iter):
            lr_t = self.lr * (1 - t / self.n_iter)
            sigma_t = self.sigma * (1 - t / self.n_iter)

            for x in X:
                bmu_idx = self._find_bmu(x)
                bmu_loc = self.locations[bmu_idx]

                # Update neighbors
                for i, loc in enumerate(self.locations):
                    dist = np.linalg.norm(loc - bmu_loc)
                    h = np.exp(-dist**2 / (2 * sigma_t**2))
                    self.weights[i] += lr_t * h * (x - self.weights[i])

    def map(self, X):
        return [self._find_bmu(x) for x in X]
