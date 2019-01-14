import mxnet as mx
import mxnet.gluon as gl

from mxnet.gluon.data.vision import transforms, MNIST

toTensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=(0.5), std=(0.5))
transform = transforms.Compose([
    toTensor, normalize
])

# Download and load the training data
trainSet: MNIST = MNIST('~/.mxnet/MNIST_data/', train=True).transform_first(transform)
trainLoader = gl.data.DataLoader(trainSet, batch_size=64, shuffle=True)


class HConv_SO2(gl.Block):

    def forward(self, *args):
        pass
