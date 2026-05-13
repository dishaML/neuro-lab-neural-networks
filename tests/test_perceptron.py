import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from perceptron.perceptron import Perceptron
import numpy as np

# Example: AND gate training
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])

p = Perceptron(learning_rate=0.1, n_iters=10)
p.fit(X, y)

print("Predictions:", p.predict(X))

