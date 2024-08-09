# Dropout

## Definition

The dropout layer essentially tries to address one problem:
preventing over-fitting as much as possible, which is understandable
since part of the neurons are "dropped out" randomly during training,
making it hard for some neurons to completely memorize the inputs.

The term **drop out** means to zero out part of the values in the
output layer in random order, which brings us to another term often
in user -- **drop out rate**. This describes how many neurons to
drop in contrast to **rate** describing how many neurons to keep.

## Forward & Backward Pass

### Forward Pass

Suppose the drop rate is $p$. This would mean that for each neuron it
has $(1-p)%$ probability to not be zeroed out. This pattern matches
exactly the _Bernoulli Distribution_. As we know, Bernoulli distribution
can be treated as a special case of Binomial Distribution, we shall
use `np.random.binomial` to handle it.

**`np.random.binomial(n, p, size)`** : in $size$ number of trials, play
$n$ times with a probability of $p$, the total number of successful plays
in each trial.

Thus, the output:

```Python
output = inputs * np.random.binomial(1, 1-dropout, inputs.shape)
```

This seems a plausible solution except that it misses one critical point.
When we apply dropout on the network, the intermediate result of each layer
will shrink due to the zeroing-out values. If we don't do anything about
this, the network will perform very badly in case of validating data since
it's trained for better working with smaller results.

One way to tackle this would be to divide $1-dropout$ in the back of the
equation.

```Python
output = inputs * np.random.binomial(1, 1-dropout, inputs.shape) / (1-dropout)
```

This output definitely vary from the fully connected network where dropout
is not applied. But in case of large quantity of data, the average output
value converges to the same.

### Backward Pass

$$
dinputs = dvalues \times \dfrac{1}{1-dropout}
$$
