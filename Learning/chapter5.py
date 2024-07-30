import numpy as np
import nnfs

from abc import ABC, abstractmethod
from nnfs.datasets import spiral_data

nnfs.init()


class DenseLayer:
    def __init__(self, n_inputs: int, n_neurons: int):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs: np.ndarray):
        self.output = np.dot(inputs, self.weights) + self.biases


class ActivationBase(ABC):
    @abstractmethod
    def forward(self, inputs: np.ndarray):
        pass


class ActivationReLU(ActivationBase):
    def forward(self, inputs: np.ndarray):
        self.output = np.maximum(0, inputs)


class ActivationSoftMax(ActivationBase):
    def forward(self, inputs: np.ndarray):
        inputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = inputs / np.sum(inputs, axis=1, keepdims=True)


class Loss(ABC):
    @abstractmethod
    def forward(self, predicted: np.ndarray, desired: np.ndarray) -> np.ndarray:
        pass

    def _calculate_acc(self, predicted: np.ndarray, desired: np.ndarray):
        _pred = np.argmax(predicted, axis=1)

        if desired.ndim == 2:
            desired = np.argmax(desired, axis=1)

        # Comparison is performed element-wise
        # `_pred == desired` -> a list of boolean
        self.acc = np.mean(_pred == desired)

    def calculate(self, predicted: np.ndarray, desired: np.ndarray) -> np.ndarray:
        self._calculate_acc(predicted, desired)
        return np.mean(self.forward(predicted, desired))


class CrossEntropyLoss(Loss):
    def forward(self, predicted: np.ndarray, desired: np.ndarray) -> np.ndarray:
        _pred = (
            predicted[range(len(predicted)), desired]
            if desired.ndim == 1
            else np.sum(predicted * desired, axis=1)
        )

        return -np.log(np.clip(_pred, 1e-7, 1 - 1e-7))


x, y = spiral_data(samples=100, classes=3)

dense1 = DenseLayer(2, 3)
activate1 = ActivationReLU()
dense2 = DenseLayer(3, 3)
activate2 = ActivationSoftMax()
loss = CrossEntropyLoss()

dense1.forward(x)
activate1.forward(dense1.output)
dense2.forward(activate1.output)
activate2.forward(dense2.output)

print(activate2.output[:5])
print(loss.calculate(activate2.output, y))
print(loss.acc)
