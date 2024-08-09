# L1 & L2 Regularization

## Definition

L1 and L2 regularization are used to prevent over-fitting in neural
network. The results of L1 and L2 regularization are called **penalty**,
indicating penalizing a neuron for attempting to memorize / over-fit
the network. If a neuron's weight value is way too big, it usually
means the underlying over-fitting problem.

## Forward Pass

- L1 Weight Regularization

$$
L_1 = \lambda \sum_{m=1}^{n} | w_m |
$$

- L1 Bias Regularization

$$
L_1 = \lambda \sum_{m=1}^{n} | b_m |
$$

- L2 Weight Regularization

$$
L_2 = \lambda \sum_{m=1}^{n} w_m^2
$$

- L2 Bias Regularization

$$
L_2 = \lambda \sum_{m=1}^{n} b_m^2
$$

The overall loss value will be:

$$
loss = data\_loss + regularization\_loss
$$

## Backward Pass

Since we're adding the regularization loss to the overall loss value,
we need to address it in the backward pass as well.

As we've mentioned above:

> $$
> loss = data\_loss + regularization\_loss
> $$

To calculate its derivative:

$$
\begin{equation}
\frac{\partial loss}{\partial weights} = \frac{\partial data\_loss}{\partial weights}
+ \frac{\partial regularization\_loss}{\partial weights}
\end{equation}
$$

$$
\begin{equation}
\frac{\partial loss}{\partial biases} = \frac{\partial data\_loss}{\partial biases}
+ \frac{\partial regularization\_loss}{\partial biases}
\end{equation}
$$

Thus, in each dense layer's backward method, we need to calculate its
corresponding regularization loss derivative and add it to the data loss
derivative.
