import mxnet as mx

from mxnet.ndarray import *


def activaton(x: NDArray):
    return 1 / (1 + exp(-x))


mx.random.seed(7)

features = mx.random.randn(1, 5)

weights = mx.random.randn(*features.shape)

bias = mx.random.randn(1, 1)

output = (features * weights).sum() + bias

y = activaton(output)
print(y)
