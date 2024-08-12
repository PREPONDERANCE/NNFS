# Regression

## Linear Activation

Since our goal now shifted into predicting a specific scalar value,
all we need is a linear activation function in the output layer.
The hidden layers though, remain using ReLU activation.

From the coding aspects, we'd like similar operations on different
activation functions, i.e. same APIs. So we construct a `ActivationLinear`
class to implement linear activation.

## Mean Squared Error Loss

### Forward

$$
L_{i,j} = \sum_{j=1}^{n} (y_{i,j} - \hat{y_{i,j}})^2
$$

$$
L_i = \dfrac{1}{J} \sum_{j=1}^{n} (y_{i,j} - \hat{y_{i,j}})^2
$$

### Backward

$$
\begin{equation}
\dfrac{\partial L_i}{\partial \hat{y_{i,j}}} = \\
-\dfrac{2}{J} (y_{i,j} - \hat{y_{i,j}})
\end{equation}
$$

## Mean Absolute Error Loss

### Forward

$$
L_{i,j} = \sum_{j=1}^{n} |y_{i,j} - \hat{y_{i,j}}|
$$

$$
L_i = \dfrac{1}{J} \sum_{j=1}^{n} |y_{i,j} - \hat{y_{i,j}}|
$$

### Backward

$$
\dfrac{\partial L_i}{\partial \hat{y_{i,j}}} =
\begin{cases}
\dfrac{1}{J} & \text{if }\ y_{i,j} - \hat{y_{i,j}} > 0 \\
-\dfrac{1}{J} & \text{if }\ y_{i,j} - \hat{y_{i,j}} < 0
\end{cases}
$$

### `np.sign`

`np.sign` takes in an array and return an array with the same
shape. The value at each slot depends on that in the original
array, $1$ if $original > 0$, $0$ if $original = 0$, $-1$ otherwise.
