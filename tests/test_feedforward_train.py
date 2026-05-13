import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from feedforward_network.feedforward import FeedForwardNetwork

# XOR problem
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])

ffn = FeedForwardNetwork(input_size=2, hidden_size=2, output_size=1, learning_rate=0.1)
ffn.train(X, y, epochs=5000)

print("Predictions:", ffn.forward(X).round(3))
