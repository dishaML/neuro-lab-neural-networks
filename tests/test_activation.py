import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from activation_functions.activation import sigmoid, relu, tanh

import numpy as np
from activation_functions.activation import sigmoid, relu, tanh

X = np.array([-2, -1, 0, 1, 2])
print("Sigmoid:", sigmoid(X))
print("ReLU:", relu(X))
print("Tanh:", tanh(X))

