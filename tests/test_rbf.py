import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from rbf.rbf import RBFNetwork

# Example: XOR-like dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])

rbf = RBFNetwork(n_hidden=2, learning_rate=0.1, epochs=500)
rbf.train(X, y)

print("Predictions:", rbf.predict(X).round(3))
