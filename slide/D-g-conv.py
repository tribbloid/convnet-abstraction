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

# %% [markdown]
# ## Conventional CNNs
#
# $$f_+(y) = <A_{ug} \circ f(x), w_0(x)> _x$$
#
# - this implies a bijection $y \longleftrightarrow A_{ug}$
#
# | - | Input $f(x)$ | High-level $f_+(y)$, $f_{++}(z)$, ... | Augmentation $A_{ug}$, $U_{ga}$, ...
# | ---|---------------|----------------|-------------------------------
# | domain | $R^2$         | $R^2$          | $R^2$ (translation only)
#
# ---
#
# - That's how it happens! ConvNet layers are just augmented FC layers!
# - First of its kind but not the last
# - A rare case when domain $X = Y$, in all other cases $X \subset Y$

# %% [markdown]
# ## Conventional CNNs
#
# | - | Input $f(x)$ | High-level $f_+(y)$, $f_{++}(z)$, ... | Augmentation $A_{ug}$, $U_{ga}$, ...
# | ---|---------------|----------------|-------------------------------
# | domain | $R^2$         | $R^2$          | $R^2$ (translation only)
#
# ---

# %%
g = nx.DiGraph(directed=True)

nodes = [
    "$f: R^2$",
    "$f_+: R^2$",
    "$f_{++}: R^2$",
    "$f_{+++}: R^2$"
]

for i in range(0, len(nodes) - 1):
    g.add_edge(nodes[i], nodes[i + 1], text="||")

drawGraph(g)

plt.show()

# %% [markdown]
# ## Group Equivariant CNNs (ICML 2016*)
#
# | - | Input $f(x)$ | High-level $f_+(y)$, $f_{++}(z)$, ... | Augmentation $A_{ug}$, $U_{ga}$, ...
# | --- |---|---|---
# | domain | $R^2$ | $R^2 \times p4$ | $R^2 \times p4$ (translation, rotation $\pm 90^{\circ}$)
#
# ---
#
# - change looks trivial
#
# <img src="assets/r2p4.png" width="500">
#
# ---
#
# [*] T. S. Cohen and M. Welling, “Group Equivariant Convolutional Networks,” ICML 2016.

# %% [markdown]
# ## Group Equivariant CNNs (ICML 2016*)
#
# | - | Input $f(x)$ | High-level $f_+(y)$, $f_{++}(z)$, ... | Augmentation $A_{ug}$, $U_{ga}$, ...
# | --- |---|---|---
# | domain | $R^2$ | $R^2 \times p4$ | $R^2 \times p4$ (translation, rotation $\pm 90^{\circ}$)
#
# ---

# %%
g = nx.DiGraph(directed=True)

tail = "$f: R^2$"

angles = [-90, 0, 90, 180]


def regularize(v: int) -> int:
    if v > 180:
        return regularize(v - 360)
    elif v <= -180:
        return regularize(v + 360)
    else:
        return v


def repr(v: int) -> str:
    r = regularize(v)
    if r > 0:
        return f"+{str(r)}^{{\circ}}"
    else:
        return f"{str(r)}^{{\circ}}"


sub = "+"
subPlus = ""

for i in angles:
    node = f"$f_{{{sub}}} | {repr(i)}$"
    g.add_edge(tail, node, text=f"${repr(i)}$")

for epoch in range(1, 3):
    subPlus = f"{sub}+"
    for i in angles:
        for j in angles:
            prev = f"$f_{{{sub}}} | {repr(i)}$"
            node = f"$f_{{{subPlus}}} | {repr(j)}$"
            g.add_edge(prev, node, text=f"${repr(j - i)}$")
    sub = subPlus

drawGraph(g, font='humor sans', label_pos=0.8)

plt.show()

# %% [markdown]
# ## Group Equivariant CNNs (ICML 2016*) - Alternatively
#
# | - | Input $f(x)$ | High-level $f_+(y)$, $f_{++}(z)$, ... | Augmentation $A_{ug}$, $U_{ga}$, ...
# | --- |---|---|---
# | domain | $R^2$ | $R^2 \times p4m$ | $R^2 \times p4m$ (translation, rotation $\pm 90^{\circ}$, flipping)
#
# ---
#
# - Size of filter bank start to become annoying, but still acceptable.
#
# <img src="assets/r2p4m.png" width="500">
#
# ---
#
# [*] T. S. Cohen and M. Welling, “Group Equivariant Convolutional Networks,” ICML 2016.

# %% [markdown]
# ## Steerable CNNs (ICLR 2017*)
#
# ---
#
# ---
#
# [*] T. S. Cohen and M. Welling, “Steerable CNNs,” ICLR 2017.


# %%



