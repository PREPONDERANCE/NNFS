# Optimizers

## Definition

Before we talk about variant optimization algorithms, it's necessary
to think about what the optimization process intends to do.

We calculate the loss and accuracy values when one training process
is complete. The loss values describes how close we're predicting the
results while the accuracy describes how correct we're in predicting
the right category.

Our goal in optimization remains always decreasing the **loss** values.
The accuracy may rise accordingly when the loss value drops but is not
our area of focus.

## Stochastic Gradient Descent (SGD)

SGD basically subtracts the $step\_size \times derivative$ from weights
and biases and repeats this process in each epoch (An epoch is a full
pass including a forward pass and a backward pass).

This $step\_size$ is also referred to as **learning rate ($lr$)**. A
training with too high a $lr$ value may lead to the increment in loss
value which is really something we do not desire. A rather low $lr$
value may lead to a local minimum.

We can dynamically decrease the $lr$ value as we train the model, that's
where **learning rate decay** comes into play. In our case, we shall use
$\dfrac{1}{t}$ learning rate decay.

$$
curr\_lr = lr \times \dfrac{1}{1 + iterations \times decay}
$$

Besides the above, **momentum** is introduced to speed up the training
process. See [this link](https://machinelearningmastery.com/gradient-descent-with-momentum-from-scratch/) for more in-detail explanation
of momentum.

Basically, momentum keeps record of history weight & bias update and
influence the current weight & bias update by adding a fraction of the
history update.

$$
weight\_update = -curr\_lr \times dweights + momentum \times weight\_momentum
$$

To see why momentum actually works, here're some helpful links:

- [What is the logic and intuition behind the momentum optimization algorithm, and why is it considered better than gradient descent?](https://www.quora.com/What-is-the-logic-and-intuition-behind-the-momentum-optimization-algorithm-and-why-is-it-considered-better-than-gradient-descent)
- [CS Notes](https://cs231n.github.io/neural-networks-3/#sgd)

## AdaGrad Gradient Descent

AdaGrad stands for **Adaptive Gradient** and basically it will tune the
learning rate for each weight dynamically by recording a `weight_cache`
and a `biases_cache`.

In the real-world training process, some weight value may rise very
swiftly while others may rise rather slowly, in which case, we'd like to
apply different learning rate to each weight value.

```Python
weight_cache += layer.dweights**2
layer.weight += -self.curr_lr * layer.dweights / (np.sqrt(weight_cache) + self.epsilon)
```

The epsilon value here simply serves to prevent division by $0$. The reason
why this algorithm tunes the learning rate dynamically is that it divides
`np.sqrt(weight_cache) + self.epsilon` at the end of the equation. We can
actually peg it as tuning the `curr_lr`.

## RMS Prop Gradient Descent

RMS Prop stands for **Root Mean Square Propagation**, which works pretty
much like that of AdaGrad except that it introduces another hyper parameter
-- $\rho$ when calculating the cache.

```Python
weight_cache = self.rho * weight_cache + (1-self.rho) * layer.dweights**2
```

Other things remain the same. The hyper parameter $\rho$ is the cache decay
rate, meaning how much fraction to keep from the accumulated cache. Since
the accumulated cache may grow with training, even small updates will lead
to significant change in `weight_cache`, thus causing the `weights` to change
dramatically if the initial learning rate is set too high. A typical learning
rate fro RMSProp will be 0.001.

## Adam Gradient Descent

To the original [paper](https://arxiv.org/pdf/1412.6980).

Adam stands for **Adaptive Momentum**. I consider it a combination of SGD and
AdaGrad. It introduces `epsilon`, `beta1`, and `beta2` hyper parameter. It also
adds a bias-correction mechanism. To achieve this correction, both momentums
and caches are divided by $1-beta^{step}$.

For weight/bias momentum calculation:

$$
m(t) = beta_1 * m(t-1) + (1-beta_1) * gradient
$$

For weight/bias cache calculation:

$$
u(t) = beta_1 * u(t-1) + (1-beta_1) * gradient^2
$$

Weight/bias momentum/cache correction:

$$
m_{corrected}(t) = \dfrac{m(t)}{1-beta_2^{step}}
$$

$$
u_{corrected}(t) = \dfrac{u(t)}{1-beta_2^{step}}
$$

```Python
weight = -self.curr_lr * weight_momentum_corrected / \
    (np.sqrt(weight_cache_corrected) + self.epsilon)
```
