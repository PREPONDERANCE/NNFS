# Binary Logistic Regression

## Introduction

So far, we've seen the categorical prediction where we end up with a
list of probabilities on each category. **Binary** prediction, on the
other hand, pegs each neuron as a separate class. The more a neuron
consider it close to a class, the closer its value is to 1. Thus, we
introduce the sigmoid activation function and binary cross entropy.

## Sigmoid Activation Function

### Forward

$$
\sigma(x) = \dfrac{1}{1 + e^{-x}}
$$

or

$$
\sigma(i,j) = \dfrac{1}{1 + e^{-z_{i,j}}}
$$

The result of this function is confined within the range $(0, 1)$. The
higher the inputs value is, the closer its result is to 1. With NumPy,
implementing this function is easy.

```Python
output = 1 / (1 + np.exp(-inputs))
```

### Backward

$$
\begin{equation}
\frac{d \sigma(i,j)}{dz_{i,j}} = \frac{e^{-z_{i,j}}}{(1 + e^{-z_{i,j}})^2} = \\
\frac{1}{1 + e^{-z_{i,j}}} \times \frac{e^{-z_{i,j}}}{1 + e^{-z_{i,j}}} = \\
\sigma(i,j) \times (1-\sigma(i,j))
\end{equation}
$$

## Binary Cross Entropy

One difference between binary cross entropy and categorical cross
entropy is that we calculate the loss by taking the sum of $y\_true
 \times \log(\hat{y})$ and negate it in the latter one but in binary
cross entropy, each neuron represents a class, leading to calculating
the loss neuron by neuron.

### Forward

$$
L_{i,j} = - y_{i,j} \times \log(\hat{y_{i,j}}) - (1-y_{i,j}) \times \log(1-\hat{y_{i,j}})
$$

With NumPy:

```Python
loss = - (target * np.log(predict) + (1-target) * np.log(1-predict))
loss = np.mean(loss, axis=-1)
```

But, it's worth noticing that before we carry out the calculation, we
need to clip the `predict` to $[10^{-7}, 1-10^{-7}]$ as we've done in
categorical cross entropy since a $log(0)$ will result in an `inf` value
, making it meaningless to calculate the mean value.

We've also brought the mean value calculation to loss in binary cross
entropy. This is because the `loss` is a list of vectors and we need
to operate on a set of samples.

$$
L_j = \dfrac{1}{J} \times \sum_{j=1}^{n} L_{i,j}
$$

### Backward

$$
\begin{equation}
\frac{\partial L_{i,j}}{\partial \hat{y_{i,j}}} = \\
-(\frac{y_{i,j}}{\hat{y_{i,j}}} - \frac{1-y_{i,j}}{1-\hat{y_{i,j}}})
\end{equation}
$$

This seems quite enough except that eventually we're going to calculate
the sample's loss with respect to each input.

$$
\begin{equation}
\frac{\partial L_j}{\partial \hat{y_{i,j}}} = \\
\frac{\partial L_j}{\partial L_{i,j}} \times \frac{\partial L_{i,j}}{\partial \hat{y_{i,j}}} = \\
-\dfrac{1}{J} \times (\frac{y_{i,j}}{\hat{y_{i,j}}} - \frac{1-y_{i,j}}{1-\hat{y_{i,j}}})
\end{equation}
$$
