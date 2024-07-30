# Calculating Neural Network With Loss

## Categorical Cross Entropy Loss

$$
L_i = - \sum_{i=1}^{n}y_{i,j}\log{\hat{y_{i,j}}}
$$

$L_i$ represents the _ith_ sample in the resulting output sample sets.
$y$ denotes the desired output and $\hat{y}$ denotes the predicted values.
$i$ and $j$ are simply indices.

## One-hot Array/Vector

An array/vector with only one value being 1 and the rest being 0. In our
example, our desired output belongs to one-hot array/vector. Thus, we may
simplify the calculation of categorical cross entropy loss.

$$
L_i = - \log{\hat{y_{desired}}}
$$

The index of $\hat{y_{desired}}$ in the predicted array is the value 1
in the desired array.

## Clip the Result

Before we actually calculate the $\log$ result, we need to clip the values
within the range $[1e-7, 1-1e-7]$.

Consider an edge case where our predicted result is the exact same as the
desired result -- an one-hot array/vector, but the indices of value 1 in
two arrays are not the same, in which case, we end up calculating $\log{0}$,
which is undefined (will be `-np.inf` using NumPy).

In the above case, we're unable to evaluate the loss rate of this batch inputs
we send in since one of the loss value is negative infinity. No matter how
small other loss values are, it's nothing compared to infinity, making it hard
to tune it the weights and biases based on the batch loss.

## What is Accuracy?

The accuracy basically evaluates how much times our model predict the right
category without considering the how much certainty our model has for the
predicted class.

To calculate the accuracy, find out the index at which the element is the
largest among the array for each input in the batch. We shall call it `predicted`.
The desired output may have already described our indices of interests if it's
a 1D array. Otherwise, we can well reuse the same methodology for calculating
argmax in `predicted`. We shall call it `desired`.

```Python
acc = np.mean(predicted==desired)
```
