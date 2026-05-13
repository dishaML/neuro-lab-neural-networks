import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from som.som import SOM

# Example: clustering 2D points
X = np.array([
    [0.1, 0.2],
    [0.2, 0.1],
    [0.8, 0.9],
    [0.9, 0.8]
])

som = SOM(m=2, n=2, dim=2, learning_rate=0.5, n_iter=100)
som.train(X)

print("Mapped nodes:", som.map(X))
