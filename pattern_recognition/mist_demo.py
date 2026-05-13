import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from sklearn.datasets import fetch_openml
from utils.helpers import normalize_data, one_hot_encode, plot_samples
from feedforward_network.feedforward import FeedForwardNetwork

# Load MNIST dataset (force NumPy arrays, avoid pandas)
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)

# Normalize and one-hot encode
X = normalize_data(X)
y_encoded = one_hot_encode(y, 10)

# Use a small subset for demo (1000 samples)
X_train, y_train = X[:1000], y_encoded[:1000]

# Build and train feedforward network
ffn = FeedForwardNetwork(input_size=784, hidden_size=64, output_size=10)
ffn.train(X_train, y_train, epochs=100)

# Predictions on first 10 samples
preds = ffn.forward(X_train[:10])
pred_labels = np.argmax(preds, axis=1)

print("Predicted labels:", pred_labels)
print("True labels:", y[:10])

# Accuracy on training subset
train_preds = np.argmax(ffn.forward(X_train), axis=1)
train_true = np.argmax(y_train, axis=1)
accuracy = np.mean(train_preds == train_true)
print(f"Training accuracy on 1000 samples: {accuracy:.2f}")

# Visualize first 10 samples with labels
plot_samples(X_train, y[:1000], n=10)

