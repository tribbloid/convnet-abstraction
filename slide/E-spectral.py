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
# [*] D. E. Worrall, S. J. Garbin, D. Turmukhambetov, and G. J. Brostow, “Harmonic Networks: Deep Translation and Rotation Equivariance” CVPR 2017, vol. 2017–Jan, pp. 7168–7177.


# %% [markdown]
# ## Going Spectral
#
# - function is like infinite-dimension vector
#
# - Spectral decomposition for functions is like eigen-decomposition for vectors
#
# ---
#
# *orthonormal basis*: given a function domain $X \longrightarrow C$, there may exist a (likely infinite) series of bases ${u_1, u_2, ...}$ that are both:
#
# - **complete**: linear combination can approximate arbitrary function on the domain
#     
# $$
# f(.) = \sum_{\forall i} \phi_i u_i(.) = \Big[ \phi_1, \phi_2, ... \Big]
# \begin{bmatrix}
#     u_1(.) \\
#     u_2(.) \\
#     ...
# \end{bmatrix}
# $$
#
# - ... and **orthonormal**: have unit norms and orthogonal to each other
#
# $$
# <u_i(.), u_j(.)> = I_{ij} \text{  (Kronecker delta)}
# $$
#
# - ... which implies:
#
# $$
# \phi_i = \widehat{f}(i) = <f(.), u_i(.)> \text{  (GFT)}
# $$

# %% [markdown]
# ## Going Spectral - What's the point?
#
# Remember G-conv?
#
# \begin{align}
# & & f_+(y) &= <A_{ug} \circ f(x), w_0(x)>_x \\
# & & &= \sum_i \sum_j \widehat{A_{ug} \circ f}(i) \widehat{w_0}(j) <u_i(.), u_j(.)> \\
# &\text{(orthonormal components ignored)} & &= \sum_i \widehat{A_{ug} \circ f}(i) \widehat{w_0}(i) \\
# &\text{(GFT)} & &= \sum_i < A_{ug} \circ f(x), u_i(x) >_x \cdot < w_0(x), u_i(x) >_x
# \end{align}
#

# %% [markdown]
# ## Going Spectral - Weight Compressor
#
# If luckily $A_{ug}$ is linear (sorry more equations):
#
# \begin{align}
# &\text{(bijectory)} & \widehat{f_+}(j) &= \sum_i \Big< < \bbox[yellow]{A_{ug}(y)} \circ f(x), u_i(x) >_x \Big| u_j(y) \Big>_y \cdot < w_0(x), u_i(x) >_x \\
# &\text{(linear)} & &= \sum_i < u_j(y), A_{ug}(y)>_y \circ < f(x), u_i(x) >_x \cdot < w_0(x), u_i(x) >_x \\
# & & &= < \widehat{A_{ug}}(j) \circ \widehat{f}(i), \widehat{w_0}(i) >_i \text{  (looks like convolution theorem still works here)} \\
# & \text{(ONLY IF $A_{ug}$ is also distance-preserving)} & &= <\widehat{f}(i), \widehat{A_{ug}^{-1}}(j) \circ \widehat{w_0}(i) >_i 
# \end{align}
#
# Convnet layers are small and use low-frequency features, so:
#
# - size of $i, j$ can be absurdly small
#
# - 

# %% [markdown]
# ## Harmonic Net (CVPR 2017*)
#
# | - | Input $f(x)$ | High-level $f_+(y)$, $f_{++}(z)$, ... | Augmentation $A_{ug}$, $U_{ga}$, ...
# | --- |---|---|---
# | domain | $R^2$ | $O(2)$ | $O(2) \cong R^2 \times SO(2)$ (translation, arbitrary rotation)
#
# ---
#
# - $u_i(x) \longleftarrow e^{i(x[0] + i x[1])}$ (2D Fourier series)
# - GFT $\longleftarrow$ FT (with Gaussian resampling)
# - $size(i) \longleftarrow 2$
#
#
#
# ---
#
# [*] D. E. Worrall, S. J. Garbin, D. Turmukhambetov, and G. J. Brostow, “Harmonic Networks: Deep Translation and Rotation Equivariance” CVPR 2017, vol. 2017–Jan, pp. 7168–7177.

# %% [markdown]
# ## Harmonic Net (CVPR 2017*)
#
# | - | Input $f(x)$ | High-level $f_+(y)$, $f_{++}(z)$, ... | Augmentation $A_{ug}$, $U_{ga}$, ...
# | --- |---|---|---
# | domain | $R^2$ | $O(2)$ | $O(2) \cong R^2 \times SO(2)$ (translation, arbitrary rotation)
#
# ---
#
#
#
# ---
#
# [*] D. E. Worrall, S. J. Garbin, D. Turmukhambetov, and G. J. Brostow, “Harmonic Networks: Deep Translation and Rotation Equivariance” CVPR 2017, vol. 2017–Jan, pp. 7168–7177.

# %% [markdown]
# ## Spherical CNNs (ICLR 2018* best paper)
#
# | - | Input $f(x)$ | High-level $f_+(y)$, $f_{++}(z)$, ... | Augmentation $A_{ug}$, $U_{ga}$, ...
# | --- |---|---|---
# | domain | $R^2$ | $O(2)$ | $O(2) \cong R^2 \times SO(2)$ (translation, arbitrary rotation)
#
# ---
#
#
#
# ---
