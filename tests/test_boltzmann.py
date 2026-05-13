import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from boltzmann.boltzmann import BoltzmannMachine

# Simple binary dataset
X = np.array([
    [1,0,1,0],
    [0,1,0,1],
    [1,1,0,0],
    [0,0,1,1]
])

bm = BoltzmannMachine(n_visible=4, n_hidden=2, learning_rate=0.1, epochs=500)
bm.train(X)

test = np.array([1,0,0,0])
print("Input:", test)
print("Reconstruction:", bm.reconstruct(test).round(3))
