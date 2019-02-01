from mxnet.gluon.nn import Sequential
from mxnet.io import NDArrayIter

from aug2conv import *

transform: Sequential = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

# Download and load the training data
trainSet: MNIST = MNIST('~/.mxnet/MNIST_data/', train=True).transform_first(transform)
trainLoader: DataLoader = DataLoader(trainSet, batch_size=4, shuffle=True)

imgs: NDArray
imgs, labels = next(trainLoader.__iter__())
imgs = imgs.squeeze(axis=(1,))
assert imgs.shape.__len__() == 3
img = imgs[0, :, :].squeeze()

assert type(img) == NDArray


def testInPlace():
    r1 = shiftX(img.copy(), 10)
    shiftX(img, 10)

    assert np.array_equal(r1.asnumpy(), img.asnumpy())


def testIter():
    print(imgs.shape)
    itr = NDArrayIter(imgs)
    for batch in itr:
        print(type(batch))
        print(len(batch.data))
        print(batch.data[0].shape)


def testAugBatch():
    augmenter = Augmenter(img)
    r = augmenter.augBatch(imgs, labels)
    print(r[0].shape)
    print(r[1].shape)
    assert (r[0].shape[0] == r[1].shape[0])


def testGetData():
    p = preprocessor
    from utils import debugSeq
    debugSeq(p, (28, 28, 3))

# def getSub():
#     global subI
#     v = plt.subplot(1, 3, subI)
#     subI += 1
#     return v
#
# def testRef():
#     getSub().imshow(img.asnumpy())
#     # plt.show()
#
# def testRollX():
#
#     copy = img.copy()
#     shifted = rollX(copy, 10)
#
#     getSub().imshow(shifted.asnumpy())
#     # plt.show()
#
# @pytest.fixture(scope='session', autouse=True)
# def db_conn():
#     # Will be executed before the first test
#     yield conn
#     # Will be executed after the last test
#     conn.disconnect()
