from mxnet.gluon.nn import Sequential, HybridSequential
import mxnet as mx


def expand(v: Sequential):
    c = list(v)
    buffer = list()

    for i in c:
        if isinstance(i, Sequential):
            buffer.extend(expand(i))
        if isinstance(i, HybridSequential):
            buffer.extend(expand(i))
        else:
            buffer.append(i)

    return buffer


def debugSeq(v: Sequential, shape):
    input = mx.ndarray.zeros(shape)

    children = expand(v)

    m = input
    print(f"tensor - {m.shape}")
    for e in children.__iter__():
        m = e(m)
        print(f"\t => {0}:{type(e)} >=>")
        print(f"tensor - {m.shape}")

    return
