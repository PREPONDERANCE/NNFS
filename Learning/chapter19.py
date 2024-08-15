import os
import cv2
import math
import nnfs
import numpy as np

nnfs.init()


class Layer:
    def __init__(self):
        self.next = self.prev = None

    def connect(self, other: "Layer"):
        self.next = other
        other.prev = self

    def __str__(self):
        return self.__class__.__name__


class LayerInput(Layer):
    def forward(self, inputs: np.ndarray):
        self.output = inputs


class LayerDense(Layer):
    def __init__(
        self,
        n_inputs: int,
        n_neurons: int,
        weight_regularize_l1: float = 0.0,
        weight_regularize_l2: float = 0.0,
        bias_regularize_l1: float = 0.0,
        bias_regularize_l2: float = 0.0,
    ):
        super().__init__()
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        self.wr1 = weight_regularize_l1
        self.wr2 = weight_regularize_l2
        self.br1 = bias_regularize_l1
        self.br2 = bias_regularize_l2

    def forward(self, inputs: np.ndarray, training: bool):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues: np.ndarray):
        self.dinputs = np.dot(dvalues, self.weights.T)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dweights = np.dot(self.inputs.T, dvalues)

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


class LayerDropout(Layer):
    def __init__(self, drop_rate: float = 0.0):
        super().__init__()
        self.rate = 1 - drop_rate

    def forward(self, inputs: np.ndarray, training: bool):
        self.binary_mask = np.random.binomial(1, self.rate, inputs.shape) / self.rate
        self.output = inputs * self.binary_mask if training else inputs.copy()

    def backward(self, dvalues: np.ndarray):
        self.dinputs = dvalues * self.binary_mask


class ActivationReLU(Layer):
    def forward(self, inputs: np.ndarray, training):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues: np.ndarray):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class ActivationLinear(Layer):
    def forward(self, inputs: np.ndarray, training):
        self.output = inputs

    def backward(self, dvalues: np.ndarray):
        self.dinputs = dvalues.copy()

    def predict(self, output: np.ndarray):
        return output


class ActivationSoftMax(Layer):
    def forward(self, inputs: np.ndarray, training):
        inputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = inputs / np.sum(inputs, axis=1, keepdims=True)

    def backward(self, dvalues: np.ndarray):
        self.dinputs = np.empty_like(dvalues)

        for idx, (output, dvalue) in enumerate(zip(self.output, dvalues)):
            output = output.reshape(-1, 1)
            jacob = np.diagflat(output) - np.dot(output, output.T)
            self.dinputs[idx] = np.dot(dvalue, jacob)

    def predict(self, output: np.ndarray):
        return np.argmax(output, axis=1)


class ActivationSigmoid(Layer):
    def forward(self, inputs: np.ndarray, training):
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues: np.ndarray):
        self.dinputs = dvalues * self.output * (1 - self.output)

    def predict(self, output: np.ndarray):
        return (output > 0.5) * 1


class Loss(Layer):
    def __init__(self):
        super().__init__()
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def forward(self, predict: np.ndarray, target: np.ndarray):
        raise NotImplementedError

    def backward(self, dvalues: np.ndarray, target: np.ndarray):
        raise NotImplementedError

    def set_trainable(self, layers: list[Layer]):
        self.trainable = layers

    def calculate(
        self,
        predict: np.ndarray,
        target: np.ndarray,
        *,
        include_regularization: bool = True,
    ):
        sample_losses = self.forward(predict, target)
        data_loss = np.mean(sample_losses)

        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        if not include_regularization:
            return data_loss

        return data_loss, self.regularize()

    def calculate_accumulated(self, *, include_regularization: bool = True):
        data_loss = self.accumulated_sum / self.accumulated_count

        if not include_regularization:
            return data_loss

        return data_loss, self.regularize()

    def new_pass(self):
        self.accumulated_count = 0
        self.accumulated_sum = 0

    def regularize(self):
        rl = 0

        for layer in self.trainable:
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
            -(target * np.log(predict) + (1 - target) * np.log(1 - predict)), axis=-1
        )

    def backward(self, dvalues: np.ndarray, target: np.ndarray):
        batch, output = dvalues.shape
        dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
        self.dinputs = -(target / dvalues - (1 - target) / (1 - dvalues)) / (
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


class ActivationCategoricalLoss:
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


class Accuracy(Layer):
    def __init__(self):
        super().__init__()
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def init(self, target: np.ndarray):
        pass

    def compare(self, output: np.ndarray, target: np.ndarray):
        pass

    def calculate(self, output: np.ndarray, target: np.ndarray):
        comparisons = self.compare(output, target)

        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)

        return np.mean(comparisons)

    def calculate_accumulated(self):
        return self.accumulated_sum / self.accumulated_count

    def new_pass(self):
        self.accumulated_count = 0
        self.accumulated_sum = 0


class AccuracyRegression(Accuracy):
    def __init__(self):
        super().__init__()
        self.precision = None

    def init(self, target: np.ndarray, *, re_init: bool = True):
        if not self.precision or re_init:
            self.precision = np.std(target) / 250

    def compare(self, output: np.ndarray, target: np.ndarray):
        return np.abs(output - target) < self.precision


class AccuracyCategorical(Accuracy):
    def compare(self, output: np.ndarray, target: np.ndarray):
        if target.ndim == 2:
            target = np.argmax(target, axis=1)

        return output == target


class Optimizer(Layer):
    def __init__(self, learning_rate: float, decay: float):
        super().__init__()
        self.lr = learning_rate
        self.curr_lr = self.lr
        self.iter = 0
        self.decay = decay

    def set_trainable(self, layers: list[Layer]):
        self.trainable = layers

    def pre_update_params(self):
        self.curr_lr = self.lr * (1 / (1 + self.decay * self.iter))

    def update_params(self, layer: LayerDense):
        pass

    def update(self):
        for layer in self.trainable:
            self.update_params(layer)

    def post_update_params(self):
        self.iter += 1


class OptimizerSGD(Optimizer):
    def __init__(self, learning_rate: float, decay: float, momentum: float):
        super().__init__(learning_rate, decay)
        self.momentum = momentum

    def update_params(self, layer: LayerDense):
        if not hasattr(layer, "weight_momentum"):
            layer.weight_momentum = np.zeros_like(layer.weights)
            layer.bias_momentum = np.zeros_like(layer.biases)

        weight_update = (
            -self.curr_lr * layer.dweights + self.momentum * layer.weight_momentum
        )
        bias_update = (
            -self.curr_lr * layer.dbiases + self.momentum * layer.bias_momentum
        )

        layer.weights += weight_update
        layer.biases += bias_update

        layer.weight_momentum = weight_update
        layer.bias_momentum = bias_update


class OptimizerAdaGrad(Optimizer):
    def __init__(self, learning_rate: float, decay: float, epsilon: float):
        super().__init__(learning_rate, decay)
        self.epsilon = epsilon

    def update_params(self, layer: LayerDense):
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        layer.weights += (
            -self.curr_lr
            * layer.dweights
            / (np.sqrt(layer.weight_cache) + self.epsilon)
        )
        layer.biases += (
            -self.curr_lr * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
        )


class OptimizerRMSProp(Optimizer):
    def __init__(self, learning_rate: float, decay: float, epsilon: float, rho: float):
        super().__init__(learning_rate, decay)
        self.epsilon = epsilon
        self.rho = rho

    def update_params(self, layer: LayerDense):
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache = (
            self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights**2
        )
        layer.bias_cache = (
            self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases**2
        )

        layer.weights += (
            -self.curr_lr
            * layer.dweights
            / (np.sqrt(layer.weight_cache) + self.epsilon)
        )
        layer.biases += (
            -self.curr_lr * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
        )


class OptimizerAdam(Optimizer):
    def __init__(
        self,
        learning_rate: float,
        decay: float,
        epsilon: float,
        beta1: float,
        beta2: float,
    ):
        super().__init__(learning_rate, decay)
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2

    def update_params(self, layer: LayerDense):
        if not hasattr(layer, "weight_cache"):
            layer.weight_momentum = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentum = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentum = (
            self.beta1 * layer.weight_momentum + (1 - self.beta1) * layer.dweights
        )
        layer.bias_momentum = (
            self.beta1 * layer.bias_momentum + (1 - self.beta1) * layer.dbiases
        )

        layer.weight_cache = (
            self.beta2 * layer.weight_cache + (1 - self.beta2) * layer.dweights**2
        )
        layer.bias_cache = (
            self.beta2 * layer.bias_cache + (1 - self.beta2) * layer.dbiases**2
        )

        corrected_wm = layer.weight_momentum / (1 - self.beta1 ** (self.iter + 1))
        corrected_bm = layer.bias_momentum / (1 - self.beta1 ** (self.iter + 1))
        corrected_wc = layer.weight_cache / (1 - self.beta2 ** (self.iter + 1))
        corrected_bc = layer.bias_cache / (1 - self.beta2 ** (self.iter + 1))

        layer.weights += (
            -self.curr_lr * corrected_wm / (np.sqrt(corrected_wc) + self.epsilon)
        )
        layer.biases += (
            -self.curr_lr * corrected_bm / (np.sqrt(corrected_bc) + self.epsilon)
        )


class Model:
    def __init__(self):
        self.layers, self.trainable = [], []
        self.input_layer = self.output_layer = None

    def add(self, layer: Layer):
        if self.layers:
            self.layers[-1].connect(layer)
        self.layers.append(layer)

    def set(self, *, loss: Loss, optimizer: Optimizer, accuracy: Accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def finalize(self):
        self.input_layer = LayerInput()

        if self.layers and self.loss:
            self.output_layer = self.layers[-1]

            self.input_layer.connect(self.layers[0])
            self.layers[-1].connect(self.loss)

            self.trainable = [_l for _l in self.layers if hasattr(_l, "weights")]
            self.loss.set_trainable(self.trainable)
            self.optimizer.set_trainable(self.trainable)

            if isinstance(self.output_layer, ActivationSoftMax) and isinstance(
                self.loss, LossCategoricalCrossEntropy
            ):
                self.softmax_loss = ActivationCategoricalLoss()

    def forward(self, data: np.ndarray, *, training: bool = False):
        self.input_layer.forward(data)

        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        return layer.output

    def backward(self, dvalues: np.ndarray, target: np.ndarray):
        if self.softmax_loss:
            self.softmax_loss.backward(dvalues, target)
            self.layers[-1].dinputs = self.softmax_loss.dinputs

            for layer in self.layers[-2::-1]:
                layer.backward(layer.next.dinputs)
            return

        self.loss.backward(dvalues, target)

        for layer in self.layers[::-1]:
            layer.backward(layer.next.dinputs)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        batch_size: int | None = None,
        validation: tuple[np.ndarray, np.ndarray] = None,
        epochs: int = 100,
        print_every: int = 10,
    ):
        self.accuracy.init(y)

        train_steps = 1 if not batch_size else math.ceil(len(X) / batch_size)
        if validation is not None:
            X_test, y_test = validation
            validation_steps = (
                1 if not batch_size else math.ceil(len(X_test) / batch_size)
            )

        for epoch in range(1, 1 + epochs):
            print(f"epoch: {epoch}")

            self.loss.new_pass()
            self.accuracy.new_pass()

            for step in range(train_steps):
                if batch_size is None:
                    batch_x = X
                    batch_y = y
                else:
                    batch_x = X[step * batch_size : (step + 1) * batch_size]
                    batch_y = y[step * batch_size : (step + 1) * batch_size]

                output = self.forward(batch_x)

                data_loss, reg_loss = self.loss.calculate(output, batch_y)
                loss = data_loss + reg_loss

                predictions = self.output_layer.predict(output)
                acc = self.accuracy.calculate(predictions, batch_y)

                self.backward(output, batch_y)

                self.optimizer.pre_update_params()
                self.optimizer.update()
                self.optimizer.post_update_params()

                if not step % print_every or step == train_steps - 1:
                    print(
                        f"step: {step}, "
                        + f"acc: {acc:.3f}, "
                        + f"loss: {loss:.3f}, "
                        + f"data_loss: {data_loss:.3f}, "
                        + f"reg_loss: {reg_loss:.3f}, "
                        + f"lr: {self.optimizer.curr_lr}"
                    )

            epoch_data_loss, epoch_reg_loss = self.loss.calculate_accumulated()
            epoch_loss = epoch_data_loss + epoch_reg_loss
            epoch_acc = self.accuracy.calculate_accumulated()

            print(
                "training, "
                + f"acc: {epoch_acc:.3f}, "
                + f"loss: {epoch_loss:.3f}, "
                + f"data_loss: {epoch_data_loss:.3f}, "
                + f"reg_loss: {epoch_reg_loss:.3f}, "
                + f"lr: {self.optimizer.curr_lr}"
            )

        if validation:
            self.loss.new_pass()
            self.accuracy.new_pass()

            for step in range(validation_steps):
                if batch_size is None:
                    batch_x_test = X_test
                    batch_y_test = y_test
                else:
                    batch_x_test = X_test[step * batch_size : (step + 1) * batch_size]
                    batch_y_test = y_test[step * batch_size : (step + 1) * batch_size]

                output = self.forward(batch_x_test, training=False)
                self.loss.calculate(output, batch_y_test, include_regularization=False)
                predictions = self.output_layer.predict(output)
                self.accuracy.calculate(predictions, batch_y_test)

            loss = self.loss.calculate_accumulated(include_regularization=False)
            acc = self.accuracy.calculate_accumulated()

            print(f"Validation, acc: {acc:.3f}, loss: {loss:.3f}")

    def dump(self):
        if self.layers:
            print("Layers:", end=" ")
            head = self.input_layer
            while head:
                print(head, end="")
                head = head.next
                print(" <-> ", end="") if head else print()
        else:
            print("No layers added yet.")


def load_dataset(dataset: str, path: str):
    labels = os.listdir(os.path.join(path, dataset))

    X, y = [], []

    for label in labels:
        for file in os.listdir(os.path.join(path, dataset, label)):
            image = cv2.imread(
                os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED
            )
            X.append(image)
            y.append(label)

    return np.array(X), np.array(y).astype("uint8")


def create_data(path: str):
    X_train, y_train = load_dataset("train", path)
    X_test, y_test = load_dataset("test", path)

    return (X_train, y_train, X_test, y_test)


X, y, X_test, y_test = create_data("fashion_mnist_images")

# Shuffle the data
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X, y = X[keys], y[keys]

# Reshape and Scale the data
X = (X.astype("float32") - 127.5) / 127.5
X_test = (X_test.astype("float32") - 127.5) / 127.5

X = X.reshape(X.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

model = Model()

model.add(LayerDense(X.shape[1], 64))
model.add(ActivationReLU())
model.add(LayerDense(64, 64))
model.add(ActivationReLU())
model.add(LayerDense(64, 10))
model.add(ActivationSoftMax())

model.set(
    loss=LossCategoricalCrossEntropy(),
    optimizer=OptimizerAdam(
        learning_rate=0.005, decay=5e-5, epsilon=1e-7, beta1=0.9, beta2=0.999
    ),
    accuracy=AccuracyCategorical(),
)

model.finalize()
model.train(
    X, y, batch_size=128, validation=(X_test, y_test), epochs=5, print_every=100
)
