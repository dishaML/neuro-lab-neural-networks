import numpy as np
from perceptron.perceptron import Perceptron

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])

model = Perceptron(lr=0.1, epochs=10)
model.fit(X, y)
print("Predictions:", model.predict(X))
