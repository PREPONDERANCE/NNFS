# Adding Layers To Neural Network

## Dense Layer Class

```Python
class DenseLayer:
    def __init__(self, n_inputs: int, n_neurons: int):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs: np.ndarray):
        self.output = np.dot(inputs, self.weights) + self.biases
```

### NumPy: random & zeros

- `np.random.randn`: this function takes in `n` integers (d1, d2, d3 ... dn)
  as the inputs and return an array of shape (d1, d2, d3 ... dn), whose values
  center around 0 and own a standard deviation of 1.

- `np.zeros`: this function takes in an integer or a tuple and return an array
  filled with 0. Return a 1D array if an integer is passed in, otherwise an array
  of shape of the input tuple.

### Details into the class

#### Why construct the weight matrix with shape `(n_inputs, n_neurons)`?

Coming from Chapter 2, the example matrix for `weights` seems logical.

```Python
weights = [[0.7, 1.2, 0.6, 2.1], [1.2, -0.5, 2.1, 1.4], [5.4, 2.2, 3.3, 2.4]]
```

The length of `weights` corresponds to the number of output neurons while the
length of each inner list corresponds to the number of input neurons.

Logically, we should construct the `weights` as `(n_neurons, n_inputs)` but do
remember that in our previous examples where we'd like to perform a dot product
to calculate the output neurons, we will transpose the `weights` beforehand.

By switching the order in shape tuple, we avoid transposing the matrix each time
we want to calculate the output.

#### Why times 0.01 before `np.random.randn`?

We’re going to multiply this Gaussian distribution for the weights by ​0.01​ to
generate numbers that are a couple of magnitudes smaller. Otherwise, the model
will take more time to fit the data during the training process as starting
values will be disproportionately large compared to the updates being made during
training.

The idea here is to start a model with non-zero values small enough that they
won’t affect training. This way, we have a bunch of values to begin working with,
but hopefully none too large or as zeros. You can experiment with values other
than ​0.01​ if you like.

#### Why construct the `biases` into a matrix?

As per the writer puts it:

> We’ll initialize the biases with the shape of ​(1, n_neurons),​ as a row vector,
> which will let us easily add it to the result of the dot product later, without
> additional operations like transposition.

We have the chance to construct a 1D `biases`, which won't affect the final output.

### Notice

Running `nnfs.init()` before everything will override part of NumPy functions,
leading to some unexpected errors.

So if you want to test NumPy outputs, please comment out `nnfs.init()` and proceed.
