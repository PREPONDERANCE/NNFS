# Back Propagation

## Definition

It's the often case where we want to find out how a single neuron affects
the entire network's output. In mathematical sense, the whole network can
be generalized into a function that takes in $n$ neurons as inputs and
outputs $m$ values. Hence the following equation:

$$
output = f_1(f_2(f_3(f_4(x_1, x_2, x_3, ..., x_n))))
$$

To calculate how a single neuron impacts the entire output, we may calculate
the partial derivative on the output with respect the a single input. For
example:

$$
\begin{equation}
\frac{\partial output}{\partial x_1} = \frac{\partial output}{\partial f_1} \\
\times \frac{\partial f_1}{\partial f_2} \times\frac{\partial f_2}{\partial f_3} \\
\times \frac{\partial f_3}{\partial f_4} \times\frac{\partial f_4}{\partial x_1}
\end{equation}
$$

This is called **Chain Rule** in math. Applying it to our network, we can
calculate each layer's partial derivative from backward and chain them
together for the final impact.

## Single Layer

For a single neuron to pass the ReLU activation function, it generally
takes the following steps:

1. $input \times weight = output1$
2. $output1 + bias = output2$
3. $output2 \text{ if } output2 > 0 \text{ else } 0$

For a single layer's input to pass the activation function, the following
equation should be obvious based on the above steps.

$$
output = ReLU(sum(mul(x_1, w_1), mul(x_2, w_2), mul(x_3, y_3), biases))
$$

### Calculate the impact of input on output

$$
\begin{equation}
\frac{\partial output}{\partial x_1} = \\
\frac{\partial ReLU}{\partial sum} \times \\
\frac{\partial sum}{\partial mul} \times \\
\frac{\partial mul}{\partial x_1} = \\
dvalues \times drelu \times w_1
\end{equation}
$$

### Calculate the impact of weight on output

$$
\begin{equation}
\frac{\partial output}{\partial x_1} = \\
\frac{\partial ReLU}{\partial sum} \times \\
\frac{\partial sum}{\partial mul} \times \\
\frac{\partial mul}{\partial w_1} = \\
dvalues \times drelu \times x_1
\end{equation}
$$

### Calculate the impact of biases on output

$$
\begin{equation}
\frac{\partial output}{\partial biases} = \\
\frac{\partial ReLU}{\partial sum} \times \\
\frac{\partial sum}{\partial biases} = \\
dvalues \times drelu
\end{equation}
$$

The $dvalues$ are the back propagated derivatives from the previous
layers. According to the chain rule, we need to multiply it to get
the full back propagated results.

## Derivative of ReLU

### ReLU

$$
y = \begin{cases}
    x & \text{if } x > 0 \\
    0 & \text{if } x \leq 0
    \end{cases}
$$

### Derivative

$$
\frac{dy}{dx} = \begin{cases}
                1 & \text{if } x > 0 \\
                0 & \text{if } x \leq 0
                \end{cases}
$$

### Implementation

```Python
class ActivationReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
```

We made a copy of the passed in `dvalues` so that the original
variables won'e be compromised. `self.dinputs[self.inputs <= 0] = 0`
means we're going to modify `self.dinputs` in such a way that when
the corresponding value in `inputs` are $\leq 0$, we change the value
in the current slot to $0$.

## Derivative of Loss

### Loss

$$
L_{i,j} = -\sum_{j}^{n} y_{i,j} \log{\hat{y_{i,j}}}
$$

### Derivative

$$
\begin{equation}
\frac{\partial L_{i,j}}{\partial \hat{y_{i,j}}} = \\
-(\frac{y_{i,1}}{\hat{y_{i,j}}} + \\
  \frac{y_{i,2}}{\hat{y_{i,j}}} + ... \\
  \frac{y_{i,j}}{\hat{y_{i,j}}} + ... \\
  \frac{y_{i,n}}{\hat{y_{i,j}}}) = \\
- \sum_{j}^{n} \frac{y_{i,j}}{\hat{y_{i,j}}} = \\
- \frac{y_{i,j}}{\hat{y_{i,j}}}
\end{equation}
$$

### Implementation

```Python
class Loss(ABC):
    @abstractmethod
    def forward(self, predict, target):
        pass


class LossCrossEntropy(Loss):
    def forward(self, predict, target):
        pass

    def backward(self, dvalues, target):
        batch, labels = dvalues.shape

        if target.ndim == 1:
            target = np.eye(labels)[target]

        self.dinputs = (-target / dvalues) / batch
```

The most mind-boggling part among the `backward` function will be
the use of `np.eye`. As we know, the `target`, also known as the
ground-truth matrix may be a 1D array, in which case, we need to
convert it into a 2D matrix.

Example on 1D `target`:

```Python
target = [0, 1, 2, 1, 0, 2]
```

This array means for example sample in the batch the index of the
desired category. Its corresponding 2D version may be:

```Python
target = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1]
]
```

Suppose there're 3 neurons as the output. It's easy to discover that
the matrix is just **a linear combination of a 3x3 identity matrix**.

Thus, we can create a 3x3 identity matrix and select its rows with the
given 1D target array.

## Derivative of Softmax

### Softmax

$$
S_{i,j} = \frac{e^{z_{i,j}}}{\sum_{l=0}^{n} e^{z_{i,l}}}
$$

### Derivative

Every input impacts the result of softmax in its own way. Thus it's
necessary to calculate the partial derivative of each softmax result
with respect to each input.

$$
\begin{equation}
\frac{\partial S_{i,j}}{\partial Z_{i,k}} =
\frac{\partial \frac{e^{z_{i,j}}}{\sum_{l=0}^{n} e^{z_{i,l}}}}{\partial Z_{i,k}} = \\
\frac{\frac{\partial e^{z_{i,j}}}{\partial Z_{i,k}} \times \sum_{l=0}^{n} e^{z_{i,l}} \\
- e^{z_{i,j}} \times \frac{\partial \sum_{l=0}^{n} e^{z_{i,l}} }{\partial Z_{i,k}} }{(\sum_{l=0}^{n} e^{z_{i,l}})^2} \\
\end{equation}
$$

There're two cases to consider at this point: $j = k$ and $j \neq k$.

**First case: $j = k$**

$$
\begin{equation}
\frac{\partial S_{i,j}}{\partial Z_{i,k}} =
\frac{e^{z_{i,j}} \times \sum_{l=0}^{n} e^{z_{i,l}} - e^{z_{i,j}} \times e^{z_{i,k}} }{(\sum_{l=0}^{n} e^{z_{i,l}})^2} = \\
\frac{e^{z_{i,j}}}{\sum_{l=0}^{n} e^{z_{i,l}}} \times (1 - \frac{e^{z_{i,k}}}{\sum_{l=0}^{n} e^{z_{i,l}}}) \\
\end{equation}
$$

**Second case: $j \neq k$**

$$
\begin{equation}
\frac{\partial S_{i,j}}{\partial Z_{i,k}} =
\frac{- e^{z_{i,j}} \times e^{z_{i,k}} }{(\sum_{l=0}^{n} e^{z_{i,l}})^2} = \\
\frac{e^{z_{i,j}}}{\sum_{l=0}^{n} e^{z_{i,l}}} \times (0 - \frac{e^{z_{i,k}}}{\sum_{l=0}^{n} e^{z_{i,l}}}) \\
\end{equation}
$$

The above can be generalized into the following:

$$
\delta(i,j) = \begin{cases}
              1 & \text{if } i = j \\
              0 & \text{if } i \neq j
              \end{cases}
$$

$$
\frac{\partial S_{i,j}}{\partial Z_{i,k}} = S_{i,j} \times (\delta(j,k) - S_{i,k})
$$

### Implementation

```Python
class ActivationSoftMax:
    def forward(self, inputs):
        pass

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for idx, (output, dvalue) in enumerate(zip(self.output, dvalues)):
            output = output.reshape(-1, 1)
            jacob = np.diagflat(output) - np.dot(output, output.T)
            self.dinputs[idx] = np.dot(jacob, dvalue)
```

Suppose there're three neurons from the `dvalues` in the next layer.
In other words, the shape of `dvalues` is $m$x3. As we've mentioned above,

> Every input impacts the result of softmax in its own way.

we need to calculate the partial derivative of each softmax output
with respect to each input neuron value, eventually leading to the
following matrix.

$$
jacob = \begin{bmatrix}
        \frac{\partial S_{0}}{\partial Z_{0}} & \frac{\partial S_{0}}{\partial Z_{1}} & \frac{\partial S_{0}}{\partial Z_{2}} \\
        \frac{\partial S_{1}}{\partial Z_{0}} & \frac{\partial S_{1}}{\partial Z_{1}} & \frac{\partial S_{1}}{\partial Z_{2}} \\
        \frac{\partial S_{2}}{\partial Z_{0}} & \frac{\partial S_{2}}{\partial Z_{1}} & \frac{\partial S_{2}}{\partial Z_{2}} \\
        \end{bmatrix}
$$

This matrix represents a single input list (it's called **Jacobian Matrix**).
Thus, we need to summarize the overall impact a neuron has on the output,
i.e. accumulate the rows -- perform a dot product between `dvalue` and `jacob`.

As of how to construct the **Jacobian Matrix** with NumPy, let's trace back to
the equation we just conducted.

> $$
> \frac{\partial S_{i,j}}{\partial Z_{i,k}} = \\
> S_{i,j} \times (\delta(j,k) - S_{i,k}) = \\
> S_{i,j} \times \delta(j,k) - S_{i,j} \times S_{i,k}
> $$

The left part before the "$-$" sign essentially represents the diagonal
in the $jacob$ matrix.

To build this diagonal, we first reshape the each single output to a 2D
array with the second dimension being 1. The -1 in the code means let
NumPy decides the appropriate value for the first dimension. We then use
`np.diagflat` to construct a diagonal matrix with the output values being
the diagonal.

The right part after the "$-$" sign can be constructed by performing a dot
product between `output` and `output.T`.

Finally, we summarized the jacobian matrix based on the rows -- another
dot product.

## Derivative of Softmax and Loss

In real network. we tend to use this method since it's faster to
execute and easier to implement.

### Derivative

$$
\begin{equation}
\frac{\partial L_{i,j}}{\partial Z_{i,k}} = \\
\frac{\partial L_{i,j}}{\partial \hat{y_{i,j}}} \times \frac{\partial S_{i,j}}{\partial Z_{i,k}}
\end{equation}
$$

Since $S_{i,j}$ is $\hat{y_{i,j}}$, we can simplify the above into:

$$
\begin{equation}
\frac{\partial L_{i,j}}{\partial Z_{i,k}} = \\
\frac{\partial L_{i,j}}{\partial \hat{y_{i,j}}} \times \frac{\partial \hat{y_{i,j}}}{\partial Z_{i,k}} = \\
-\sum_{j}^{n} \frac{y_{i,j}}{\hat{y_{i,j}}} \times \frac{\partial \hat{y_{i,j}}}{\partial Z_{i,k}} = \\
-\frac{y_{i,k}}{\hat{y_{i,k}}} \times \frac{\partial \hat{y_{i,k}}}{\partial Z_{i,k}} -\sum_{j \neq  k}^{n} \frac{y_{i,j}}{\hat{y_{i,j}}} \times \frac{\partial \hat{y_{i,j}}}{\partial Z_{i,k}}
\end{equation}
$$

Again, we need to separately consider the $j = k$ and $j \neq k$ cases.

When $j = k$, the left part before the "$-$" sign is valid since for
every $\hat{y_{i,j}}$ where $j \neq k$, they will be treated as a constant
while conducting partial derivative. Thus they shall be $0$ in this case.

When $j \neq k$, the right part after the "$-$" sign is valid since
$\hat{y_{i,k}}$ is no longer considered a variable when conducting a partial
derivative.

Combining the equations we've implemented:

> $$
> \frac{\partial S_{i,j}}{\partial Z_{i,k}} = \\
> S_{i,j} \times (\delta(j,k) - S_{i,k})
> $$

we can further simplify our softmax-loss derivative:

$$
\begin{equation}
\begin{split}
\frac{\partial L_{i,j}}{\partial Z_{i,k}} =
-\frac{y_{i,k}}{\hat{y_{i,k}}} \times \hat{y_{i,k}} \times (1 - \hat{y_{i,k}}) = \\
-\sum_{j \neq  k}^{n} \frac{y_{i,j}}{\hat{y_{i,j}}} \times (- \hat{y_{i,j}} \times \hat{y_{i,k}}) = \\
-y_{i,k} + y_{i,k} \times \hat{y_{i,k}} + \sum_{j \neq  k}^{n} y_{i,j} \times \hat{y_{i,k}} = \\
-y_{i,k} + \sum_{j}^{n} y_{i,j} \times \hat{y_{i,k}} = -y_{i,k} + \hat{y_{i,k}}
\end{split}
\end{equation}
$$

### Implementation

```Python
class ActivationLoss:
    def __init__(self):
        self._loss = CrossEntropyLoss()
        self._activate = ActivationSoftMax()

    def forward(self, predict, target):
        pass

    def backward(self, dvalues, target):
        batch, _ = dvalues.shape

        if target.ndim == 2:
            target = np.argmax(target, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(batch), target] -= 1
        self.dinputs /= batch
```

In this case, we're turning 2D target matrix into 1D array. And since
our `target` matrix contains one-hot vectors, we simply subtract 1 at
the index of interest after converting it into its 1D version.
