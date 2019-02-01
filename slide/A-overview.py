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

# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# # ConvNet Abstraction

# %% [markdown]
# ## Overview

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import os
import sys

if '' in sys.path:
    sys.path.remove('')

module_path = os.path.abspath(os.path.join('../python'))
if module_path not in sys.path:
    sys.path.append(module_path)

print(sys.path)

import networkx as nx

from graphPlot import drawGraph
from const import *

plt.rcParams['figure.figsize'] = [10, 10]

# %%

g = nx.DiGraph(directed=True)

schNet = "schNet\n(Nature Communication 2016)"
groupInv = "Group Invariant CNN\n(ICML 2016)"
steerable = "Steerable CNNs\n(ICLR 2017)"
harmonic = "Harmonic CNNs\n(CVPR 2017)"
spherical = "Spherical CNNs\n(ICLR 2018 best paper)"
tensorField = "*Tensor Field Net*\n(NOT peer-reviewed!)"
clebNet = "Clebsch-Gordan Net\n(NIPS 2018)"

g.add_edge(schNet, tensorField)
g.add_edge(groupInv, steerable)
g.add_edge(harmonic, spherical)
g.add_edge(steerable, spherical)
g.add_edge(spherical, tensorField)
g.add_edge(tensorField, clebNet)

drawGraph(g)

plt.show()

# %% [markdown]
# ## Pre-ConvNet
#
# 1960-1987
#
# [Eternal winter covers the land]

# %% [markdown]
# ## Pre-ConvNet - Linear Fully Connected Dense (FC) Layer
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
fc = "$<f(x), w(x, .)> d x$\nlinear fully-connected dense"
nl = "$\phi(<f(x), w(x, .)> d x)$\nnon-linear activation"
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
# ## Pre-ConvNet - Fully Connected LayersLinear Fully Connected Dense (FC) Layer
#
# Considerably weak in today's standard
# - e.g. easily confused by turning its head

# %% [markdown]
# ## Invariant Layer / Bag of Features?
#
# (DON'T DO THIS)
#
# [picasso effect]

# %% [markdown]
# ## Data Augmentation
#
# [insert example pictures]
#
# - Sounds like in the right direction
#     - maybe a bit slow in practice
#     - even in convex case SGD "theoretically probably" converges equally fast
#     - but it works fines
# - Wait a second, how about a better idea?
