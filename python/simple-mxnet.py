import mxnet

a = mxnet.nd.zeros((100, 50))
b = mxnet.nd.zeros((100, 50))

c = a * b
c *= 1
print(c)
