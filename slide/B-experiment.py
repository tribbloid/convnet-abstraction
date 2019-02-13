# -*- coding: utf-8 -*-
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

# %% {"slideshow": {"slide_type": "skip"}}
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

# %% {"slideshow": {"slide_type": "skip"}}
import os
import sys
from typing import Tuple

from dataclasses import dataclass

if '' in sys.path:
    sys.path.remove('')

module_path = os.path.abspath(os.path.join('../python'))
if module_path not in sys.path:
    sys.path.append(module_path)

import networkx as nx

from graphPlot import drawGraph, setCanvas
from const import *

setCanvas()

# %% {"slideshow": {"slide_type": "skip"}}
import aug2conv
from mxnet.ndarray import NDArray
from mxnet.gluon.data import DataLoader
from mxnet.gluon.nn import Sequential
import utils.helper

data = aug2conv.getData()

imgs: NDArray
imgs, ls = next(data.__iter__())

# %% {"slideshow": {"slide_type": "slide"}}
print("Let's do a hello-world experiment on MNIST dataset:")

slice = imgs[0:64]
utils.helper.viewWeights(slice)

# %% {"slideshow": {"slide_type": "slide"}}
print("... for which each image can be augmented until all cases are covered")

img1 = imgs[0].squeeze(axis=(0,))
augmenter = aug2conv.Augmenter(img1)
auggedUp = augmenter.aug1(img1)
utils.helper.viewWeights(auggedUp)

# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Start Learning!
#
# ### **2 layers only**
#
# 1. Highway only bypassing Linear/FC/~~Dense/Perceptron~~
#     
#     - Designed to break symmetry at saddle points*
#
#     - WITHOUT initialisation (all weights start with 0)
#     
#     - 10x10 => 10x10 => ReLU
#
# 2. Plain old Linear/FC
#
#     - 10x10 => one-hot 1~10 => ReLU
#    
# ---
#
# [*] Y. Li and Y. Yuan, “Convergence Analysis of Two-layer Neural Networks with ReLU Activation” NIPS 2017, pp. 1–11.

# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Start Learning!
#
# <img src="assets/skip1Net.png" width="400">
#
# ---
#
# [*] Y. Li and Y. Yuan, “Convergence Analysis of Two-layer Neural Networks with ReLU Activation” NIPS 2017, pp. 1–11.

# %% {"slideshow": {"slide_type": "skip"}}

from mxnet import autograd, initializer
import mxnet.gluon as glu
from pathlib import Path

lossFn = glu.loss.SoftmaxCrossEntropyLoss()


# %% {"slideshow": {"slide_type": "skip"}}

@dataclass
class HWY(glu.HybridBlock):

    def __init__(self, layers: Tuple):
        super(HWY, self).__init__()
        self.delegate = glu.nn.HybridSequential()
        self.delegate.add(*layers)

    def getLayers(self):
        return list(self.delegate)

    def hybrid_forward(self, F, x, *args, **kwargs):
        r = self.delegate.forward(x, *args)
        return r + x.reshape(r.shape)

    def __hash__(self):
        return hash(self.delegate)


# %% {"slideshow": {"slide_type": "skip"}}
# Build a feed-forward network
# this goofy-looking skip architecture is from:
# [1] Y. Li and Y. Yuan, “Convergence Analysis of Two-layer Neural Networks with ReLU Activation,” no. Nips, pp. 1–11, 2017.
# designed to break symmetry

def newModel() -> glu.nn.HybridSequential:
    model = glu.nn.HybridSequential()
    # with model.name_scope():
    model.add(
        HWY((
            glu.nn.Dense(100),
        )),
        glu.nn.Activation('relu'),
        # Skip((
        #     glu.nn.Dense(100),
        # )),
        # glu.nn.Activation('relu'),
        glu.nn.Dense(10)
    )

    # init = initializer.Uniform()
    init = initializer.Zero()
    model.initialize(ctx=CTX, init=init)
    return model


# %% {"slideshow": {"slide_type": "slide"}}
print("Weight map before training")

# Remember all weights start with 0
model = newModel()
model.forward(imgs.as_in_context(CTX))

fc1 = model[0].getLayers()[0]
utils.helper.viewFCWeights(fc1)


# %% {"slideshow": {"slide_type": "skip"}}
def train(
        name: str,
        loader: DataLoader = data,
        lossTarget=0.15,
        maxEpochs=100,
        aug=lambda v: v
) -> Sequential:
    model = newModel()

    filePath = f"{os.getcwd()}/{MODEL_CHKPNT}/{name}.model"

    try:
        model.load_parameters(filePath)
        print(f">> model loaded from: {filePath}")
    except Exception as ee:
        print(f">> model being learned from scratch: {filePath}")

        optimizer = glu.Trainer(model.collect_params(), 'sgd', {'learning_rate': 0.01})

        # cc = 0
        for epoch in range(maxEpochs):
            sumLoss = 0
            for imgs, labels in loader:
                # print(f"loading batch {cc} - of {imgs.shape[0]}")
                # cc += 1
                imgs = imgs.squeeze(axis=(1,))

                imgs, labels = aug((imgs, labels))
                imgs = imgs.as_in_context(CTX)
                labels = labels.as_in_context(CTX)

                with autograd.record():
                    output = model.forward(imgs)
                    loss = lossFn(output, labels)

                loss.backward()
                sumLoss += loss.mean().asscalar()

                optimizer.step(imgs.shape[0] / 2)

            else:
                print(f"Training loss: {sumLoss / len(loader)}")

            if sumLoss / len(loader) <= lossTarget:
                break

        os.makedirs(Path(filePath).parent, exist_ok=True)
        model.save_parameters(filePath)
        print(f">>> model saved to: {filePath}")

    return model


# %% {"slideshow": {"slide_type": "skip"}}
model = train("raw")

logits = model.forward(auggedUp.as_in_context(CTX))
ps = mx.ndarray.softmax(logits, axis=1)

# %% {"slideshow": {"slide_type": "slide"}}
print("Trained on raw MNIST dataset")

utils.helper.view_classify(auggedUp[0], ps[0])
utils.helper.view_classify(auggedUp[88], ps[88])

# %% {"slideshow": {"slide_type": "slide"}}
print("Weight map after training on raw MNIST dataset")

fc1 = model[0].getLayers()[0]
utils.helper.viewFCWeights(fc1)

# %% {"slideshow": {"slide_type": "skip"}}

# now let's enable augmentation

augModel = train("aug", aug=augmenter.augFirstTuple)

# %% {"slideshow": {"slide_type": "slide"}}
print("Weight map after training on AUGMENTED MNIST dataset")

fc1 = augModel[0].getLayers()[0]
utils.helper.viewFCWeights(fc1)

# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Data Augmentation
#
# - How did this happen? (you probably want to try this on neural-ODE)
#
# <img src="assets/jon-batiste-reaction.gif">
