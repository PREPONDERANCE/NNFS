import numpy as np
import nnfs

from nnfs.datasets import spiral_data

nnfs.init()

# Adding Layers

# inputs = [[1, 2.1, 3, 2.4], [4.2, 1.1, 6.1, -1.2], [3.3, 0.5, 2.4, 6.1]]
# weights1 = [[4, 2.1, 2.6, 1.1], [6.1, -0.6, 1.6, 7.2], [5.1, 9.1, 0.7, -4.5]]
# bias1 = [1.2, 5.2, -4.1]
# weights2 = [[4.4, 2.5, 8.1], [-0.7, 1.5, 5.2], [4.1, -9.3, 5.2]]
# bias2 = [6.4, 1.2, 5.2]

# layer1 = np.dot(inputs, np.array(weights1).T) + bias1
# output = np.dot(layer1, np.array(weights2).T) + bias2

# print(output)


# A Dense Layer


class DenseLayer:
    def __init__(self, n_inputs: int, n_neurons: int):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs: np.ndarray):
        self.output = np.dot(inputs, self.weights) + self.biases


# Inputs: 100x2
x, _ = spiral_data(samples=100, classes=3)
dense1 = DenseLayer(2, 3)
dense1.forward(x)
print(dense1.output[:5])
