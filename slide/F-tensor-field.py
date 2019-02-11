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
# ## Harmonic Net (CVPR 2017*)
#
# | - | Input $f(x)$ | High-level $f_+(y)$, $f_{++}(z)$, ... | Augmentation $A_{ug}$, $U_{ga}$, ...
# | --- |---|---|---
# | domain | $R^2$ | $O(2)$ | $O(2) \cong R^2 \times SO(2)$ (translation, arbitrary rotation)
#
# ---
#
# - Size of filter bank become really annoying, can't take it any more
#
# - First algorithm to use spectral decomposition as a weight compressor
#
# ---
#
# [*] D. E. Worrall, S. J. Garbin, D. Turmukhambetov, and G. J. Brostow, “Harmonic Networks: Deep Translation and Rotation Equivariance,” CVPR 2017, vol. 2017–Jan, pp. 7168–7177.


# %% [markdown]
# ## Going Spectral
#
# - function is like infinite-dimension vector
#
# - Spectral decomposition on functions is like eigen decomposition on vectors
#
# ---
#
# This gives 2 things (sorry more equations):
#
# - **orthonormal basis**: given a type of signals ${f}: X \longrightarrow Y$, sometimes it is possible to find a (probably infinite) series of eigen-functions ${\beta_1, \beta_2, ...}$ that satisfy:
#
#     - *orthonormality*: $<\beta_i(x), \beta_j(x)>_x = \delta_{ij}$ (Kronecker delta, or infinitely-big $I$)
#
#     - *completeness*: their linear combination can approximate arbitrary signal of such type
#
# $$
# f(x) = \sum_{i=1}^{\infty} \phi_i \beta_i(x)
# $$

# %% [markdown]
# ## Going Spectral - Orthonormal basis
#
# - **orthonormal basis**: given a type of signals ${f}: X \longrightarrow Y$, sometimes it is possible to find a (probably infinite) series of eigen-functions ${\beta_1, \beta_2, ...}$ that satisfies:
#
#     - *orthonormality*: $<\beta_i(x), \beta_j(x)>_x = \delta_{ij}$ (Kronecker delta, or infinitely-big $I$)
#
#     - *completeness*: their linear combination can approximate arbitrary signal of such type

# %% [markdown]
# ## Spherical CNNs (ICLR 2018* best paper award)
#
# | - | Input $f(x)$ | High-level $f_+(y)$, $f_{++}(z)$, ... | Augmentation $A_{ug}$, $U_{ga}$, ...
# | --- |---|---|---
# | domain | $R^2$ | $O(2)$ | $O(2) \cong R^2 \times SO(2)$ (translation, arbitrary rotation)
#
# ---
#
# ---
