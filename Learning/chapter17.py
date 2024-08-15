import numpy as np
import nnfs

from abc import ABC, abstractmethod
from nnfs.datasets import sine_data

nnfs.init()


class LayerDense:
    def __init__(
        self,
        n_inputs: int,
        n_neurons: int,
        weight_regularize_l1: float = 0.0,
        weight_regularize_l2: float = 0.0,
        bias_regularize_l1: float = 0.0,
        bias_regularize_l2: float = 0.0,
    ):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        self.wr1 = weight_regularize_l1
        self.wr2 = weight_regularize_l2
        self.br1 = bias_regularize_l1
        self.br2 = bias_regularize_l2

    def forward(self, inputs: np.ndarray):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues: np.ndarray):
        self.dinputs = np.dot(dvalues, self.weights.T)
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        if self.wr1 > 0:
            d = np.ones_like(self.weights)
            d[self.weights < 0] = -1
            self.dweights += self.wr1 * d
        if self.wr2 > 0:
            self.dweights += 2 * self.wr2 * self.weights
        if self.br1 > 0:
            d = np.ones_like(self.biases)
            d[self.biases < 0] = -1
            self.dbiases += self.br1 * d
        if self.br2 > 0:
            self.dbiases += 2 * self.br2 * self.biases


class LayerDropout:
    def __init__(self, drop_rate: float):
        self.rate = 1 - drop_rate

    def forward(self, inputs: np.ndarray):
        self.binary_masks = np.random.binomial(1, self.rate, inputs.shape) / self.rate
        self.output = inputs * self.binary_masks

    def backward(self, dvalues: np.ndarray):
        self.dinputs = dvalues * self.binary_masks


class ActivationLinear:
    def forward(self, inputs: np.ndarray):
        self.output = inputs

    def backward(self, dvalues: np.ndarray):
        self.dinputs = dvalues.copy()


class ActivationReLU:
    def forward(self, inputs: np.ndarray):
        self.inputs = inputs
        self.output = np.maximum(0, self.inputs)

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


class ActivationSigmoid:
    def forward(self, inputs: np.ndarray):
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues: np.ndarray):
        self.dinputs = dvalues * self.output * (1 - self.output)


class Loss(ABC):
    @abstractmethod
    def forward(self, predict: np.ndarray, target: np.ndarray):
        pass

    @abstractmethod
    def backward(self, dvalues: np.ndarray, target: np.ndarray):
        pass

    def calculate(self, predict: np.ndarray, target: np.ndarray):
        return np.mean(self.forward(predict, target))

    def regularize(self, layer: LayerDense):
        rl = 0

        if layer.wr1 > 0:
            rl += layer.wr1 * np.sum(np.abs(layer.weights))
        if layer.wr2 > 0:
            rl += layer.wr2 * np.sum(layer.weights * layer.weights)
        if layer.br1 > 0:
            rl += layer.br1 * np.sum(np.abs(layer.biases))
        if layer.br2 > 0:
            rl += layer.br2 * np.sum(layer.biases * layer.biases)

        return rl


class LossCategoricalCrossEntropy(Loss):
    def forward(self, predict: np.ndarray, target: np.ndarray):
        predict = (
            predict[range(len(predict)), target]
            if target.ndim == 1
            else np.sum(predict * target, axis=1)
        )

        return -np.log(np.clip(predict, 1e-7, 1 - 1e-7))

    def backward(self, dvalues: np.ndarray, target: np.ndarray):
        batch, labels = dvalues.shape

        if target.ndim == 1:
            target = np.eye(labels)[target]

        self.dinputs = (-target / dvalues) / batch


class LossBinaryCrossEntropy(Loss):
    def forward(self, predict: np.ndarray, target: np.ndarray):
        predict = np.clip(predict, 1e-7, 1 - 1e-7)
        return np.mean(
            -(target * np.log(predict)) - (1 - target) * np.log(1 - predict), axis=1
        )

    def backward(self, dvalues: np.ndarray, target: np.ndarray):
        batch, output = dvalues.shape
        dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
        self.dinputs - (target / dvalues - (1 - target) / (1 - dvalues)) / (
            batch * output
        )


class LossMeanSquaredError(Loss):
    def forward(self, predict: np.ndarray, target: np.ndarray):
        return np.mean((target - predict) ** 2, axis=-1)

    def backward(self, dvalues: np.ndarray, target: np.ndarray):
        batch, output = dvalues.shape
        self.dinputs = -2 * (target - dvalues) / (batch * output)


class LossMeanAbsoluteError(Loss):
    def forward(self, predict: np.ndarray, target: np.ndarray):
        return np.mean(np.abs(target - predict), axis=-1)

    def backward(self, dvalues: np.ndarray, target: np.ndarray):
        batch, output = dvalues.shape
        self.dinputs = np.sign(target - dvalues) / (batch * output)


class ActivationCategorical:
    """
    The combination of Softmax and Categorical Cross Entropy.
    """

    def __init__(self):
        self._loss = LossCategoricalCrossEntropy()
        self._activate = ActivationSoftMax()

    def forward(self, inputs: np.ndarray, target: np.ndarray):
        self._activate.forward(inputs)
        self.output = self._activate.output
        return self._loss.calculate(self.output, target)

    def regularize(self, layer: LayerDense):
        """
        A wrapper of `regularize` in `Loss` since the loss
        function is considered protected in this class.
        """
        return self._loss.regularize(layer)

    def backward(self, dvalues: np.ndarray, target: np.ndarray):
        batch, _ = dvalues.shape

        if target.ndim == 2:
            target = np.argmax(target, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(batch), target] -= 1
        self.dinputs /= batch


class Optimizer(ABC):
    def __init__(self, learning_rate: float, decay: float):
        self.lr = learning_rate
        self.curr_lr = self.lr
        self.iter = 0
        self.decay = decay

    def pre_update_params(self):
        self.curr_lr = self.lr * (1 / (1 + self.decay * self.iter))

    @abstractmethod
    def update_params(self, layer: LayerDense):
        pass

    def post_update_params(self):
        self.iter += 1


class OptimizerSGD(Optimizer):
    def __init__(self, learning_rate: float, decay: float, momentum: float):
        super().__init__(learning_rate, decay)
        self.momentum = momentum

    def update_params(self, layer: LayerDense):
        if not hasattr(layer, "weight_momentum"):
            layer.weight_momentum = np.zeros_like(layer.weights)
            layer.biases_momentum = np.zeros_like(layer.biases)

        weight_update = (
            -self.curr_lr * layer.dweights + self.momentum * layer.weight_momentum
        )
        biases_update = (
            -self.curr_lr * layer.dbiases + self.momentum * layer.biases_momentum
        )

        layer.weight_momentum = weight_update
        layer.biases_momentum = biases_update

        layer.weights += weight_update
        layer.biases += biases_update


class OptimizerAdaGrad(Optimizer):
    def __init__(self, learning_rate: float, decay: float, epsilon: float):
        super().__init__(learning_rate, decay)
        self.epsilon = epsilon

    def update_params(self, layer: LayerDense):
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.biases_cache = np.zeros_like(layer.biases)

        layer.weight_cache += layer.dweights**2
        layer.biases_cache += layer.dbiases**2

        layer.weights += (
            -self.curr_lr
            * layer.dweights
            / (np.sqrt(layer.weight_cache) + self.epsilon)
        )
        layer.biases += (
            -self.curr_lr * layer.dbiases / (np.sqrt(layer.biases_cache) + self.epsilon)
        )


class OptimizerRMSProp(Optimizer):
    def __init__(self, learning_rate=1.0, decay=0.0, epsilon=0.0, rho=0.0):
        super().__init__(learning_rate, decay)
        self.epsilon = epsilon
        self.rho = rho

    def update_params(self, layer: LayerDense):
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.biases_cache = np.zeros_like(layer.biases)

        layer.weight_cache = (
            self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights**2
        )
        layer.biases_cache = (
            self.rho * layer.biases_cache + (1 - self.rho) * layer.dbiases**2
        )

        layer.weights += (
            -self.curr_lr
            * layer.dweights
            / (np.sqrt(layer.weight_cache) + self.epsilon)
        )
        layer.biases += (
            -self.curr_lr * layer.dbiases / (np.sqrt(layer.biases_cache) + self.epsilon)
        )


class OptimizerAdam(Optimizer):
    def __init__(self, learning_rate=1.0, decay=0.0, epsilon=0.0, beta1=0.0, beta2=0.0):
        super().__init__(learning_rate, decay)
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2

    def update_params(self, layer: LayerDense):
        if not hasattr(layer, "weight_cache"):
            layer.weight_momentum = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.biases_momentum = np.zeros_like(layer.biases)
            layer.biases_cache = np.zeros_like(layer.biases)

        layer.weight_momentum = (
            self.beta1 * layer.weight_momentum + (1 - self.beta1) * layer.dweights
        )
        layer.biases_momentum = (
            self.beta1 * layer.biases_momentum + (1 - self.beta1) * layer.dbiases
        )

        layer.weight_cache = (
            self.beta2 * layer.weight_cache + (1 - self.beta2) * layer.dweights**2
        )
        layer.biases_cache = (
            self.beta2 * layer.biases_cache + (1 - self.beta2) * layer.dbiases**2
        )

        _corrected_wm = layer.weight_momentum / (1 - self.beta1 ** (self.iter + 1))
        _corrected_bm = layer.biases_momentum / (1 - self.beta1 ** (self.iter + 1))

        _corrected_wc = layer.weight_cache / (1 - self.beta2 ** (self.iter + 1))
        _corrected_bc = layer.biases_cache / (1 - self.beta2 ** (self.iter + 1))

        layer.weights += (
            -self.curr_lr * _corrected_wm / (np.sqrt(_corrected_wc) + self.epsilon)
        )
        layer.biases += (
            -self.curr_lr * _corrected_bm / (np.sqrt(_corrected_bc) + self.epsilon)
        )


X, y = sine_data()

dense1 = LayerDense(1, 64)
activation1 = ActivationReLU()
dense2 = LayerDense(64, 64)
activation2 = ActivationReLU()
dense3 = LayerDense(64, 1)
activation3 = ActivationLinear()
loss_function = LossMeanSquaredError()
optimizer = OptimizerAdam(0.005, 1e-3, 1e-7, 0.9, 0.99)
accuracy_precision = np.std(y) / 250

for epoch in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)

    data_loss = loss_function.calculate(activation3.output, y)
    regularization_loss = (
        loss_function.regularize(dense1)
        + loss_function.regularize(dense2)
        + loss_function.regularize(dense3)
    )
    loss = data_loss + regularization_loss

    predictions = activation3.output
    accuracy = np.mean(np.abs(predictions - y) < accuracy_precision)

    if not epoch % 100:
        print(
            f"epoch: {epoch}, "
            + f"acc: {accuracy:.3f}, "
            + f"loss: {loss:.3f}, "
            + f"data_loss: {data_loss:.3f}, "
            + f"reg_loss: {regularization_loss:.3f}, "
            + f"lr: {optimizer.curr_lr}"
        )

    loss_function.backward(activation3.output, y)
    activation3.backward(loss_function.dinputs)
    dense3.backward(activation3.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.post_update_params()
