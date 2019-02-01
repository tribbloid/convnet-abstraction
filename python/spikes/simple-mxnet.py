# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import mxnet as mx
from mxnet import autograd

mx.test_utils.set_default_context(mx.context.gpu())

a = mx.nd.zeros((100, 50))
b = mx.nd.zeros((100, 50))

c = a * b
c *= 1
print(c)

# %%
a = mx.nd.zeros((100, 50), ctx=mx.context.cpu())
b = mx.nd.zeros((100, 50))

c = a * b
c *= 1
print(c)

# %%
with autograd.record():
    
    x = mx.random.randn(2, 2)
    x.attach_grad()

    y = x ** 2
    y.attach_grad()
    
    z = y.mean()
    print(f"{x} - {y}")

print(f"{type(x)} - {type(y)}")

print(f"{x.grad} - {y.grad}")

z.backward()

print(f"{x.grad} - {y.grad}")
print(f"{x / 2}")

# %%

