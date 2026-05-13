import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from feedforward_network.feedforward import FeedForwardNetwork

X = np.array([[0,0],[0,1],[1,0],[1,1]])

ffn = FeedForwardNetwork(input_size=2, hidden_size=2, output_size=1)
output = ffn.forward(X)

print("Feedforward outputs:", output)

