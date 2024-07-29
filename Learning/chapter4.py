import numpy as np
import nnfs

from nnfs.datasets import spiral_data

nnfs.init()


class DenseLayer:
    def __init__(self, n_inputs: int, n_neurons: int):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs: np.ndarray):
        self.output = np.dot(inputs, self.weights) + self.biases


class ActivationReLU:
    def forward(self, inputs: np.ndarray):
        # Broadcast happens here
        self.output = np.maximum(0, inputs)


class ActivationSoftMax:
    def forward(self, inputs: np.ndarray):
        inputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = inputs / np.sum(inputs, axis=1, keepdims=True)


x, _ = spiral_data(samples=100, classes=3)

layer1 = DenseLayer(2, 3)
layer2 = DenseLayer(3, 3)
inter_activation = ActivationReLU()
final_activation = ActivationSoftMax()

layer1.forward(x)
inter_activation.forward(layer1.output)
layer2.forward(inter_activation.output)
final_activation.forward(layer2.output)

print(final_activation.output[:5])
