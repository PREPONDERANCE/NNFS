import itertools
import numpy as np

# A single neuron
inputs = [1.2, 4.5, 2.8]
weights = [3, 1, 4]
bias = 0.6

output = sum(itertools.starmap(lambda x, y: x * y, zip(inputs, weights))) + bias
output = np.dot(np.array(inputs), np.array(weights).T) + bias

# A layer of neurons
inputs = [1.2, 4.5, 6.4, 0.4]
weights = [[0.7, 1.2, 0.6, 2.1], [1.2, -0.5, 2.1, 1.4], [5.4, 2.2, 3.3, 2.4]]
bias = [1, -1, 4]

output = list(
    itertools.starmap(
        lambda x, y: sum(itertools.starmap(lambda p, q: p * q, zip(inputs, x))) + y,
        zip(weights, bias),
    )
)
output = np.dot(inputs, weights) + bias
output = np.dot(np.array(inputs), np.array(weights).T) + bias


# Batch operations

inputs = [[1.2, 2.3, 4.2, 5.3], [0.6, 7.1, 2.6, 9.1], [1.5, 6.4, 7.3, 3.8]]
weights = [[0.7, 1.2, 0.6, 2.1], [1.2, -0.5, 2.1, 1.4], [5.4, 2.2, 3.3, 2.4]]
bias = [1.1, -0.5, 2.3]

# Broadcast happens here
output = np.dot(inputs, np.array(weights).T) + bias
print(output)
