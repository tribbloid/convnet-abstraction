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

from graphPlot import drawGraph
from const import *

plt.rcParams['figure.figsize'] = [10, 10]
# print(plt.rcParams['figure.figsize'])

# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# # **ConvNet Abstraction**
#
# ## *TDLS Feb 11. 2019*
#
# ## Chris Dryden
#
# - [missing]
#
# ## Peng Cheng
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
harmonic = "Harmonic CNNs\n(CVPR 2017)"
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
# ## Pre-ConvNet
#
# 1960-1987
#
# [Eternal winter covers the land]

# %% [markdown]
# ## Pre-ConvNet - Linear/Fully Connected/~~Dense/Perceptron~~ Layer
#
# In pursuing of unbounded representation power
#
#
# \begin{align}
# & \text{(let $w$ be the weight function)} & f_+(x) &= \phi \circ \Big( \int\limits_{x \in \text{Manifold}} f(x) w(x, y) dx \Big) \\
# & \text{(pardon me for abusing notation a bit)} & &= \phi \circ \Big( < f(x), w(x, y) > dx \Big)
# \end{align}

# %%


g = nx.DiGraph(directed=True)

g.add_edge("$f(.)$", "== layer ==")
g.add_edge("== layer ==", "$f_+(.)$")
g.add_edge("$f_+(.)$", " == layer ==")
g.add_edge(" == layer ==", "$f_{++}(.)$")
g.add_edge("$f_{++}(.)$", "...")

dot = "$f(.)$\nsignal"
fc = "$<f(x), w(x, .)> d x$\nFC"
nl = "$\phi(<f(x), w(x, .)> d x)$\nactivation"
dot2 = "$f_+(.)$\nhigh-level signal"
# hw = "highway?"

g.add_edge(dot, fc)
g.add_edge(fc, nl)
g.add_edge(nl, dot2)

g2 = g.copy()

g2.add_edge(dot, "$f(.)$", wedge=True)
g2.add_edge(dot2, "$f_+(.)$", wedge=True)

# g.add_edge(dot, hw)
# g.add_edge(hw, dot2)

drawGraph(g2, font='humor sans', layoutG=g)

plt.show()

# %% [markdown]
# ## Pre-ConvNet - Linear/Fully Connected/~~Dense/Perceptron~~ Layer is Weak
#
# [insert example that shows it failing to generalise beyond changing of POV]

# %% [markdown]
# ## Invariant Layer / Bag of Features? (DON'T DO THIS)
#
# [picasso effect]

# %% [markdown]
# ## Data Augmentation
#
# [insert example pictures]
#
# - Sounds like the right direction
#     - a bit slow in practice
#     - even in "convex case" SGD "theoretically probably" converges equally fast
# - How about a better idea?

# %%


