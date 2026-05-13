import numpy as np

class RBFNetwork:
    def __init__(self, n_hidden, learning_rate=0.01, epochs=1000):
        self.n_hidden = n_hidden
        self.lr = learning_rate
        self.epochs = epochs
        self.centers = None
        self.weights = None

    def _rbf(self, x, c, sigma=1.0):
        return np.exp(-np.linalg.norm(x-c)**2 / (2*sigma**2))

    def _compute_activations(self, X):
        G = np.zeros((X.shape[0], self.n_hidden))
        for i, x in enumerate(X):
            for j, c in enumerate(self.centers):
                G[i, j] = self._rbf(x, c)
        return G

    def train(self, X, y):
        # Choose random centers
        random_idx = np.random.choice(X.shape[0], self.n_hidden, replace=False)
        self.centers = X[random_idx]

        # Initialize weights
        self.weights = np.random.randn(self.n_hidden)

        for _ in range(self.epochs):
            G = self._compute_activations(X)
            y_pred = G @ self.weights
            error = y - y_pred
            self.weights += self.lr * G.T @ error

    def predict(self, X):
        G = self._compute_activations(X)
        return G @ self.weights
