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
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

# %%
import os
import sys
from typing import Tuple

from dataclasses import dataclass

if '' in sys.path:
    sys.path.remove('')

module_path = os.path.abspath(os.path.join('../python'))
if module_path not in sys.path:
    sys.path.append(module_path)

# print(sys.path)

import networkx as nx

from graphPlot import drawGraph, SIZE
from const import *

plt.rcParams['figure.figsize'] = SIZE
# print(plt.rcParams['figure.figsize'])

# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# # **Tensor Field Network** (and other ConvNet Generalisations)
#
# ## *TDLS - Feb 11. 2019*
#
# ### **Chris Dryden**
#
# - [neeed your email and company]
#
# - github.com/chrisdryden
#
# ### **Peng Cheng**
#
# - pc175@uowmail.edu.au
#
# - datapassports.com
#
# - github.com/tribbloid

# %% [markdown]
# ## **Overview**

# %%

g = nx.DiGraph(directed=True)

schNet = "schNet\n(Nature\nCommunication\n2016)"
groupInv = "Group Equivariant ConvNet\n(ICML 2016)"
steerable = "Steerable CNNs\n(ICLR 2017)"
harmonic = "Harmonic Net\n(CVPR 2017)"
spherical = "Spherical CNNs\n(ICLR 2018 best paper)"
tensorField = "*Tensor Field Net*\n(not peer-reviewed!)"
cgNet = "Clebsch-Gordan Net\n(NIPS 2018)"
threeDSteerable = "3D Steerable CNNs\n(NIPS 2018)"

g.add_edge(schNet, tensorField)
g.add_edge(groupInv, steerable)
g.add_edge(harmonic, spherical)
g.add_edge(steerable, spherical)
g.add_edge(spherical, tensorField)
g.add_edge(tensorField, cgNet)
g.add_edge(tensorField, threeDSteerable)

drawGraph(g)

plt.show()

# %% [markdown]
# ## Pre-ConvNet (1960-1987)
#
# <img src="assets/winterIsComing.jpg">
#

# %% [markdown]
# ## Pre-ConvNet - Linear/Fully Connected/~~Dense/Perceptron~~ Layer
#
# In pursuing of unbounded representation/approximation power
#
# \begin{align}
# & \text{($w$ are weight of neurons)} & f_+(y) = \Phi \Big( f(x) \Big) &= \phi \Big( \sum_{x \in \text{domain}} f(x) w(x, y) \Big) \\
# & \text{(pardon the abusing of notation)} & &= \phi \Big( \bbox[yellow]{< f(x), w(x, y) >_x} \Big)
# \end{align}

# %%


g = nx.DiGraph(directed=True)

g.add_edge("$f(x)$", "== layer ==")
g.add_edge("== layer ==", "$f_+(y)$")
g.add_edge("$f_+(y)$", " == layer ==")
g.add_edge(" == layer ==", "$f_{++}(.)$")
g.add_edge("$f_{++}(.)$", "...")

dot = "$f(.)$\ninput signal"
fc = "$<f(x), w(x, y)> d x$\nlinear"
nl = "$\phi(<f(x), w(x, y)> d x)$\nactivation"
dot2 = "$f_+(y)$\nhigh-level signal"
# hw = "highway?"

g.add_edge(dot, fc)
g.add_edge(fc, nl)
g.add_edge(nl, dot2)

g2 = g.copy()

g2.add_edge(dot, "$f(x)$", wedge=True)
g2.add_edge(dot2, "$f_+(y)$", wedge=True)

drawGraph(g2, g, font_family='humor sans')

plt.show()

# %% [markdown]
# ## Pre-ConvNet - Linear Layer
#
# <img src="assets/classify-raw.jpg"><img src="assets/classify-shifted.jpg">
#
# ---
#  
# (you'll see this later in experiment)

# %% [markdown]
# ## Invariant Layer / Bag-of-words?
#
# - Don't do this
#
# <img src="assets/picassoEffect.jpg">
#

# %% [markdown]
# ## Data Augmentation
#
# - Good catch
#
# <img src="assets/data-aug.png">
#

# %% [markdown]
# ## Data Augmentation
#
# - Too slow in practice
#     - In **convex case** SGD "theoretically probably" converges equally fast
#     - otherwise it "kind of works" but with much less efficiency
#
# --- 
#
# 2D translation
#
# <img src="assets/image-pan.gif" width="600px">
#

# %% [markdown]
# ## Data Augmentation
#
# - Time & space complexity increase exponentially with the dimensionality of augmentation
#
# ---
#
# 2D translation x 1D rotation, you'll see this fairly often on some cameras
#
# <img src="assets/human-0g.jpg"><img src="assets/drone-overhead.png">
#

# %% [markdown]
# ## Data Augmentation
#
# - Time & space complexity increase exponentially with the dimensionality of augmentation
#
# ---
#
# 3D rotation
#
# <img src="assets/fisheye-pan.gif">
#
#
#
#



# %% [markdown]
# ## Data Augmentation
#
# - Time & space complexity increase exponentially with the dimensionality of augmentation
#
# ---
#
# 3D translation x 3D rotation
#
# <img src="assets/point-cloud-6d.gif" width="800px">
#

# %% [markdown]
# ## Data Augmentation
#
# - Time & space complexity increase exponentially with the dimensionality of augmentation
#
# ---
#
# How about air pressure depending on translation?
#
# <img src="assets/IAS-vs-TAS.jpg">
#

# %% [markdown]
# ## Data Augmentation
#
# How about a better idea?
#
# - Instead of augmenting, we hard-bake such prior knowledge into the network that yield identical result!
#
#
# <img src="assets/aerial-g-conv.jpg" width="500">
#
# ---
#
# # **Answer**
#
# augmentation types | solution
#  --- | --- 
# 2d translation | ConvNet
# **others** | **G-ConvNet**
# - 2d translation + 90$^{\circ}$ rotation | Group Equivariant CNNs
# - 2d translation + rotation | Harmonic Net
# - 3d rotation | Spherical CNNs
# - 3d translation + rotation | Tensor Field Net
#

# %%


