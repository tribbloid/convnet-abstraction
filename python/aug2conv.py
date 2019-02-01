import logging
from itertools import tee

import numpy as np
from mxnet import nd
from mxnet.gluon import nn
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import transforms, MNIST
from mxnet.image import copyMakeBorder
from mxnet.ndarray import NDArray

from const import *

logger = logging.getLogger(__name__)


def shiftY(img: NDArray, offset: int) -> NDArray:
    # xRange = range(0, img.shape[0])
    xRange = range(0, img.shape[1])

    for x in xRange:
        img[:, x] = np.roll(img[:, x].asnumpy(), offset)

    return img


def shiftX(img: NDArray, offset: int) -> NDArray:
    yRange = range(0, img.shape[0])
    # yRange = range(0, img.shape[1])

    for y in yRange:
        img[y] = np.roll(img[y].asnumpy(), offset)

    return img


def pad(img: NDArray, size: int) -> NDArray:
    result = copyMakeBorder(img, size, size, size, size)
    return result


# accelerated impl using
class Augmenter(object):

    def __init__(self, proto: NDArray):
        self.shape = proto.shape
        ctx = proto.context

        h = self.shape[0]
        w = self.shape[1]

        leftEye = nd.eye(h, ctx=ctx)
        rightEye = nd.eye(w, ctx=ctx)

        leftShifts = nd.zeros((h, h, h), ctx=ctx)
        for i in range(0, h):
            leftShifts[i] = leftEye
            shiftY(leftShifts[i], i)

        rightShifts = nd.zeros((w, w, w))
        for i in range(0, w):
            rightShifts[i] = rightEye
            shiftX(rightShifts[i], i)

        self.leftShifts = leftShifts
        self.rightShifts = rightShifts

    def aug1(self, img: NDArray) -> NDArray:

        result = mx.nd.zeros((self.shape[0] * self.shape[1], self.shape[1], self.shape[0]))

        k = 0
        for i in range(0, self.shape[0]):
            for j in range(0, self.shape[1]):
                result[k] = mx.nd.dot(mx.nd.dot(self.leftShifts[i], img), self.rightShifts[j])
                k += 1

        return result

    def augBatch(self, imgs: NDArray, labels: NDArray):

        assert imgs.shape.__len__() == 3, f"length = {imgs.shape.__len__()}"
        assert labels.shape.__len__() == 1

        def _batchItr():
            length = imgs.shape[0]
            for i in range(0, length):
                img = imgs[i].squeeze()
                label = labels[i]

                # print(f"batch! {img.shape}")
                imgAug: NDArray = self.aug1(img)
                labelAug: NDArray = mx.ndarray.ones(imgAug.shape[0], dtype=np.int32) * label[0]
                assert imgAug.shape[0] == labelAug.shape[0]
                # r = augged.reshape(1, *augged.shape)
                yield (imgAug, labelAug)

        batchItr = _batchItr()
        batchItr, batchItr2 = tee(batchItr)

        imgConcat = mx.nd.concat(*[i[0] for i in batchItr], dim=0)
        labelConcat = mx.nd.concat(*[i[1] for i in batchItr2], dim=0)

        return imgConcat, labelConcat

    def augTuples(self, tt):
        return self.augBatch(tt[0], tt[1])

    def augFirstTuple(self, tt):
        return self.augBatch(tt[0][[0]], tt[1][[0]])


# def aug1(img: NDArray) -> NDArray:
#     ySize, xSize = img.shape
#
#     allAugged = mx.ndarray.zeros((ySize * xSize, ySize, xSize))
#     k = 0
#     for i in range(0, ySize):
#         for j in range(0, xSize):
#             frame = img.copy()
#             shiftY(frame, i)
#             shiftX(frame, j)
#
#             allAugged[k] = frame
#             k += 1
#
#     return allAugged
#
#
# def aug(imgs: NDArray, labels: NDArray) -> (NDArray, NDArray):
#     s = f"len = {imgs.shape}"
#     logger.info(s)
#
#     assert imgs.shape.__len__() == 3
#     assert labels.shape.__len__() == 1
#
#     def _batchItr():
#         length = imgs.shape[0]
#         for i in range(0, length):
#             img = imgs[i].squeeze()
#             label = labels[i]
#
#             # print(f"batch! {img.shape}")
#             imgAug: NDArray = aug1(img)
#             labelAug: NDArray = mx.ndarray.ones(imgAug.shape[0], dtype=np.int32) * label[0]
#             assert imgAug.shape[0] == labelAug.shape[0]
#             # r = augged.reshape(1, *augged.shape)
#             yield (imgAug, labelAug)
#
#     batchItr = _batchItr()
#     batchItr, batchItr2 = tee(batchItr)
#
#     imgConcat = mx.nd.concat(*[i[0] for i in batchItr], dim=0)
#     labelConcat = mx.nd.concat(*[i[1] for i in batchItr2], dim=0)
#
#     return imgConcat, labelConcat
#
#
# def augTupleFirst(tt):
#     return aug(tt[0][0], tt[1][0])


preprocessor: nn.Sequential = transforms.Compose([
    transforms.Resize(size=(10, 10)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])


def getData():
    # Download and load the training data
    trainSet: MNIST = MNIST('~/.mxnet/MNIST_data/', train=True).transform_first(preprocessor)
    trainLoader: DataLoader = DataLoader(trainSet, batch_size=64, shuffle=False, last_batch='keep')
    return trainLoader
