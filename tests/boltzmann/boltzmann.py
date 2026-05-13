import numpy as np

class BoltzmannMachine:
    def __init__(self, n_visible, n_hidden, learning_rate=0.1, epochs=1000):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.lr = learning_rate
        self.epochs = epochs

        # Initialize weights
        self.W = np.random.randn(n_visible, n_hidden) * 0.01
        self.b_visible = np.zeros(n_visible)
        self.b_hidden = np.zeros(n_hidden)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, X):
        for epoch in range(self.epochs):
            for x in X:
                # Positive phase
                h_prob = self._sigmoid(np.dot(x, self.W) + self.b_hidden)
                h_state = (h_prob > np.random.rand(self.n_hidden)).astype(int)

                # Negative phase (reconstruction)
                v_prob = self._sigmoid(np.dot(h_state, self.W.T) + self.b_visible)
                v_state = (v_prob > np.random.rand(self.n_visible)).astype(int)

                h_prob_neg = self._sigmoid(np.dot(v_state, self.W) + self.b_hidden)

                # Weight updates
                self.W += self.lr * (np.outer(x, h_prob) - np.outer(v_state, h_prob_neg))
                self.b_visible += self.lr * (x - v_state)
                self.b_hidden += self.lr * (h_prob - h_prob_neg)

    def reconstruct(self, x):
        h_prob = self._sigmoid(np.dot(x, self.W) + self.b_hidden)
        h_state = (h_prob > np.random.rand(self.n_hidden)).astype(int)
        v_prob = self._sigmoid(np.dot(h_state, self.W.T) + self.b_visible)
        return v_prob

