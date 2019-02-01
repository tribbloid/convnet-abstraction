import mxnet.gluon as gl

from mxnet import autograd, image
from mxnet.gluon import nn
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import transforms, MNIST
from mxnet.gluon.nn import Sequential
from mxnet.ndarray import NDArray

transform: Sequential = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

# Download and load the training data
trainSet: MNIST = MNIST('~/.mxnet/MNIST_data/', train=True).transform_first(transform)
trainLoader: DataLoader = DataLoader(trainSet, batch_size=64, shuffle=True)

# Build a feed-forward network
model = nn.Sequential()
# with model.name_scope():
model.add(
    nn.Dense(128, activation='relu'),
    # nn.Activation('relu'),
    nn.Dense(64, activation='relu'),
    nn.Dense(10)
)
model.initialize()

criterion = gl.loss.SoftmaxCrossEntropyLoss()
optimizer = gl.Trainer(model.collect_params(), 'sgd', {'learning_rate': 0.01})


def pad(img: NDArray, x: int, y: int) -> NDArray:
    result = image.copyMakeBorder(img, y, y, x, x)
    return result









epochs = 5
for e in range(epochs):
    running_loss = 0
    for images, labels in trainLoader:
        # Flatten MNIST images into a 784 long vector
        images = images.reshape(images.shape[0], -1)

        # print(images.shape)

        with autograd.record():
            output = model.forward(images)
            loss = criterion(output, labels)

        loss.backward()
        optimizer.step(images.shape[0] / 2)

        running_loss += loss.mean().asscalar()
    else:
        print(f"Training loss: {running_loss / len(trainLoader)}")
