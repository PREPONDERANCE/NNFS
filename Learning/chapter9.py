import numpy as np
import nnfs

from abc import ABC, abstractmethod
from nnfs.datasets import spiral_data

nnfs.init()


class LayerDense:
    def __init__(self, n_inputs: int, n_neurons: int):
        self.weight = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs: np.ndarray):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weight) + self.biases

    def backward(self, dvalues: np.ndarray):
        self.dinputs = np.dot(dvalues, self.weight.T)
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)


class ActivationReLU:
    def forward(self, inputs: np.ndarray):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues: np.ndarray):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class ActivationSoftMax:
    def forward(self, inputs: np.ndarray):
        inputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = inputs / np.sum(inputs, axis=1, keepdims=True)

    def backward(self, dvalues: np.ndarray):
        self.dinputs = np.empty_like(dvalues)

        for idx, (output, dvalue) in enumerate(zip(self.output, dvalues)):
            output = output.reshape(-1, 1)
            jacob = np.diagflat(output) - np.dot(output, output.T)
            self.dinputs[idx] = np.dot(dvalue, jacob)


class Loss(ABC):
    @abstractmethod
    def forward(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        pass

    def calculate(self, predict: np.ndarray, target: np.ndarray):
        return np.mean(self.forward(predict, target))


class LossCategoricalCrossEntropy(Loss):
    def forward(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        predict = (
            predict[range(len(predict)), target]
            if target.ndim == 1
            else np.sum(predict * target, axis=1)
        )

        return -np.log(np.clip(predict, 1e-7, 1 - 1e-7))

    def backward(self, dvalues: np.ndarray, target: np.ndarray):
        batch, sample = dvalues.shape

        if target.ndim == 1:
            target = np.eye(sample)[target]

        self.dinputs = (-target / dvalues) / batch


class ActivationLoss:
    def __init__(self):
        self._loss = LossCategoricalCrossEntropy()
        self._activate = ActivationSoftMax()

    def forward(self, inputs: np.ndarray, target: np.ndarray):
        self._activate.forward(inputs)
        self.output = self._activate.output
        return self._loss.calculate(self.output, target)

    def backward(self, dvalues: np.ndarray, target: np.ndarray):
        batch, _ = dvalues.shape

        if target.ndim == 2:
            target = np.argmax(target, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(batch), target] -= 1
        self.dinputs /= batch


if __name__ == "__main__":
    x, y = spiral_data(100, 3)

    d1 = LayerDense(2, 3)
    ac = ActivationReLU()
    d2 = LayerDense(3, 3)
    lo = ActivationLoss()

    d1.forward(x)
    ac.forward(d1.output)
    d2.forward(ac.output)
    loss = lo.forward(d2.output, y)
    print("loss:", loss)

    lo.backward(lo.output, y)
    d2.backward(lo.dinputs)
    ac.backward(d2.dinputs)
    d1.backward(ac.dinputs)

    print(d1.dweights)
    print(d1.dbiases)
    print(d2.dweights)
    print(d2.dbiases)
