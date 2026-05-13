import numpy as np
import matplotlib.pyplot as plt

def normalize_data(X):
    return (X - np.min(X)) / (np.max(X) - np.min(X))

def one_hot_encode(y, num_classes):
    encoded = np.zeros((y.size, num_classes))
    encoded[np.arange(y.size), y] = 1
    return encoded

def plot_samples(X, y, n=10):
    plt.figure(figsize=(10,2))
    for i in range(n):
        plt.subplot(1,n,i+1)
        plt.imshow(X[i].reshape(28,28), cmap='gray')
        plt.title(str(y[i]))
        plt.axis('off')
    plt.show()
