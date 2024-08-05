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
