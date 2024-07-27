# Coding Our First Neurons

## A Single Neuron

A single neuron consists of inputs, weights, and a bias. For example:

```Python
inputs = [1.2, 4.5, 2.8]
weights = [3, 1, 4]
bias = 0.6
```

From what we've known so far, to calculate the output, we need to perform
a dot product between the `inputs` and the `weights` and add a `bias` to
the result. This can be done with the following code.

```Python
output = sum(itertools.starmap(lambda x, y: x * y, zip(inputs, weights))) + bias
```

In `NumPy`, we have a much easier way for calculation.

```Python
output = np.dot(inputs, weights) + bias
```

## A Layer of Neurons

A layer of neurons means several neurons aligned one by one, each with a
universal inputs, its own weights and biases. For example:

```Python
inputs = [1.2, 4.5, 6.4, 0.4]
weights = [[0.7, 1.2, 0.6, 2.1], [1.2, -0.5, 2.1, 1.4], [5.4, 2.2, 3.3, 2.4]]
bias = [1, -1, 4]
```

The above arbitrary code means there're three output neurons (denoted by the
length of `weights`). Each output neuron is fully connected to four inputs
and hence each `weight` is of length 4.

To calculate the output of each neuron, a simple work-around is to iterate
through each `weight` among the `weights` list and perform the single-neuron
calculation as stated above.

The raw Python code looks like this:

```Python
output = list(
    itertools.starmap(
        lambda x, y: sum(itertools.starmap(lambda p, q: p * q, zip(inputs, x))) + y,
        zip(weights, bias),
    )
)
```

Or with `NumPy`, we can simplify it as:

```Python
output = np.dot(weights, inputs) + bias
```

## NumPy: underlying `np.dot`

Take a closer look at the second `NumPy` code snippet, why should we write
`np.dot(weight, inputs)` instead of `np.dot(inputs, weights)` as in the first
example?

If you give it a shot, a **ValueError: shapes (4,) and (3,4) not aligned:4
(dim 0) != 3 (dim 0)** is raised. What is this supposed to mean?

To start off, every NumPy array has two attributes: `ndim` and `shape`. The
former one (type `int`) denotes the dimension of the array and the second
(type `tuple`) denotes its shape.

For example:

```Python
a = np.array([1, 2, 3])
b = np.array([[1, 2, 3]])

print(f"The dimension of a is {a.ndim} and that of b is {b.ndim}")
print(f"The shape of a is {a.shape} and that of b is {b.shape}")

"""
>>> The dimension of a is 1 and that of b is 2
>>> The shape of a is (3,) and that of b is (1, 3)
"""
```

Coming back to our inputs and weights example, in the first(second) example,
the `inputs` has dimension of 1(1) and shape of `(3,)`(`(4,)`) while the
`weights` has dimension of 1(2) and shape of `(3, 4)`(`(3,)`).

In the first example, all parameters in `np.dot` are 1D arrays. When it comes
to the second example, the `weights` is a 2D array.

Performing `np.dot`, NumPy will check if the shape is compatible, i.e. the
second dimension of the first array must be the same as the first dimension
of the second array. In the wrong example where we attempt to perform dot
product between `inputs` and `weights`, 4 != 3. Thus, a **ValueError** is raised.

## Batch Operations

Peg this term as _parallelism_. In the above examples, we're essentially
processing one set of inputs, also called a **sample** in deep learning. Batch
operations means training multiple samples at the same time.

For example:

```Python
inputs = [[1.2, 2.3, 4.2, 5.3], [0.6, 7.1, 2.6, 9.1], [1.5, 6.4, 7.3, 3.8]]
weights = [[0.7, 1.2, 0.6, 2.1], [1.2, -0.5, 2.1, 1.4], [5.4, 2.2, 3.3, 2.4]]
bias = [1.1, -0.5, 2.3]
```

It's obvious that the `inputs` array has turned into a 2D array, each being a
separate training sample. Since we introduce multiple samples to our neural
network, the outputs are no longer confined to one scalar value or a list of
outputs but a matrix output.

The core logic remains the same: iterate over `weights` for each `input`. But
using pure Python is no longer convenient.

In `NumPy`, we can achieve this easily.

```Python
output = np.dot(inputs, np.array(weights).T) + bias
```

## NumPy: Transposition and Broadcast

### Transposition

The transposition essentially reverses the array's shape. For example:

```Python
# 1D array
a = np.array([1,2,3])
print(a.shape, a.T.shape)

# 2D array
a = np.array([[1,2,3],[2,3,4]])
print(a.shape, a.T.shape)

# 3D array
a = np.array([[[1,2,3],[2,3,4]]])
print(a.shape, a.T.shape)

# Multi-dimension

"""
>>> (3,) (3,)
>>> (2,3) (3,2)
>>> (1,2,3) (3,2,1)
"""
```

It would be easier to grasp transposition if you visualize the matrix and
actually write it out. But it's not well applied to arrays of dimensions
higher than 2 since visualization simply won't work.

I'd rather suggest diving into the definition of transposition itself.
Suppose `a` is a four-dimensional matrix, transposition on `a` means
`a[p][q][r][s] = a[s][r][q][p]` or $a_{pqrs} = a_{srqp}$ in math notation.

### Broadcast

Link to the [NumPy](https://numpy.org/doc/stable/user/basics.broadcasting.html) docs on broadcast.
