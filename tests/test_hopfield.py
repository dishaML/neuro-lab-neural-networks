import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from hopfield.hopfield import HopfieldNetwork

# Store two patterns
patterns = np.array([
    [1, -1, 1, -1],
    [-1, 1, -1, 1]
])

hopfield = HopfieldNetwork(n_units=4)
hopfield.train(patterns)

# Recall from noisy input
test_pattern = np.array([1, -1, -1, -1])
recalled = hopfield.recall(test_pattern, steps=10)

print("Input:", test_pattern)
print("Recalled:", recalled)
