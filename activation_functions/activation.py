import numpy as np

def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))

def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)

def tanh(x):
    """Tanh activation function"""
    return np.tanh(x)
