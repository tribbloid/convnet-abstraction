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
# - Size of the filter bank become intolerably big, can't take it any more
#
# - First algorithm to use **spectral decomposition as a weight compressor**
#
# <img src="assets/hnet.gif">
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
# f(.) = \sum_{\forall m} \phi_m u_m(.) = \Big[ \phi_1, \phi_2, ... \Big]
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
# <u_m(.), u_n(.)> = I_{mn} \text{  (Kronecker delta)}
# $$
#
# - ... which implies:
#
# $$
# \phi_m = \widehat{f}(m) = <f(.), u_m(.)> \text{  (GFT)}
# $$

# %% [markdown]
# ## Going Spectral - What's the point?
#
# It makes a few things easier:
#
# - Convolution theorem still works in most cases! a.k.a. G-conv theorem
#
# $$
# \widehat{f_+}(m) = < \widehat{A_{ug}}(m) \circ \widehat{f}(m), \widehat{w_0}(m) >_m
# $$
#
# (This makes dot product and G-conv much faster)
#
# - Most G-ConvNet features are smooth on all dimensions
#
# (This means low-frequency coefficients can compress high-dimension filter banks)

# %% [markdown]
# ## *G-conv Theorem - Proof*
#
#
# \begin{align}
# & & f_+(y) &= <A_{ug} \circ f(x), w_0(x)>_x \\
# & & &= \sum_m \sum_n \widehat{A_{ug} \circ f}(m) \widehat{w_0}(j) <u_m(.), u_n(.)> \\
# &\text{(orthonormal components ignored)} & &= \sum_m \widehat{A_{ug} \circ f}(m) \widehat{w_0}(m) \\
# &\text{(GFT)} & &= \sum_m < A_{ug} \circ f(x), u_m(x) >_x \cdot < w_0(x), u_m(x) >_x
# \end{align}
#
# If luckily $A_{ug}$ is linear:
#
# \begin{align}
# &\text{(bijectory)} & \widehat{f_+}(n) &= \sum_m \Big< < \bbox[yellow]{A_{ug}(y)} \circ f(x), u_m(x) >_x \Big| u_n(y) \Big>_y \cdot < w_0(x), u_m(x) >_x \\
# &\text{(linear)} & &= \sum_m < u_n(y), A_{ug}(y)>_y \circ < f(x), u_m(x) >_x \cdot < w_0(x), u_m(x) >_x \\
# & & &= < \widehat{A_{ug}}(n) \circ \widehat{f}(m), \widehat{w_0}(m) >_m \text{  (looks like convolution theorem still works here)} \\
# & \text{(ONLY IF $A_{ug}$ is also distance-preserving)} & &= <\widehat{f}(m), \widehat{A_{ug}^{-1}}(n) \circ \widehat{w_0}(i) >_m 
# \end{align}
#

# %% [markdown]
# ## Harmonic Net (CVPR 2017*)
#
# | - | Input $f(x)$ | High-level $f_+(y)$, $f_{++}(z)$, ... | Augmentation $A_{ug}$, $U_{ga}$, ...
# | --- |---|---|---
# | domain | $R^2$ | $O(2)$ | $O(2) \cong R^2 \times SO(2)$ (translation, arbitrary rotation)
#
# ---
#
# - $u_m(x) \longleftarrow e^{m x}$ (2D Fourier series, x is a complex number)
# - GFT $\longleftarrow$ FFT (with Gaussian resampling)
# - number of coefficients $\longleftarrow 2$: $m \in {0, 1}$
#
# Orthonormal bases looks like this:
#
# <img src="assets/hnet-bases.png">
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

# %%
g = nx.DiGraph(directed=True)

tail = "$f: R^2$"

angles = [0, 1]

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
# ## Spherical CNNs (ICLR 2018* best paper)
#
# | - | Input $f(x)$ | High-level $f_+(y)$, $f_{++}(z)$, ... | Augmentation $A_{ug}$, $U_{ga}$, ...
# | --- |---|---|---
# | domain | $S^2$ | $SO(3)$ | $SO(3) \cong SU(2)$ (3d rotation)
#
# ---
#
# You can use many cameras for situation awareness
#
# <img src="assets/tesla-cameras.png">
#
# ---
#
# [*] Image Courtesy: https://www.tesla.com/en_CA/autopilot

# %% [markdown]
# ## Spherical CNNs (ICLR 2018* best paper)
#
# | - | Input $f(x)$ | High-level $f_+(y)$, $f_{++}(z)$, ... | Augmentation $A_{ug}$, $U_{ga}$, ...
# | --- |---|---|---
# | domain | $S^2$ | $SO(3)$ | $SO(3) \cong SU(2)$ (3d rotation)
#
# ---
#
# ... Or you can use few fisheye camera(s)
#
# <img src="assets/drone-fisheye.png">
#
# ---
#
# [*] Image courtesy: https://www.halfchrome.com/dji-360-drone/

# %% [markdown]
# ## Spherical CNNs (ICLR 2018* best paper)
#
# | - | Input $f(x)$ | High-level $f_+(y)$, $f_{++}(z)$, ... | Augmentation $A_{ug}$, $U_{ga}$, ...
# | --- |---|---|---
# | domain | $S^2$ | $SO(3)$ | $SO(3) \cong SU(2)$ (3d rotation)
#
# ---
#
# Instead ...
#
# <img src="assets/beholder.jpg" width="300px">
#
# ---
#
# [*] Image Courtesy: https://www.skydio.com/technology/
#

# %% [markdown]
# ## Spherical CNNs (ICLR 2018* best paper)
#
# | - | Input $f(x)$ | High-level $f_+(y)$, $f_{++}(z)$, ... | Augmentation $A_{ug}$, $U_{ga}$, ...
# | --- |---|---|---
# | domain | $S^2$ | $SO(3)$ | $SO(3) \cong SU(2)$ (3d rotation)
#
# ---
#
# Instead ...
#
# <img src="assets/skydioR1.jpg">
#
# ---
#
# [*] Image Courtesy: https://www.skydio.com/technology/
#
#

# %% [markdown]
# ## Spherical CNNs (ICLR 2018* best paper)
#
# | - | Input $f(x)$ | High-level $f_+(y)$, $f_{++}(z)$, ... | Augmentation $A_{ug}$, $U_{ga}$, ...
# | --- |---|---|---
# | domain | $S^2$ | $SO(3)$ | $SO(3) \cong SU(2)$ (3d rotation)
#
# ---
#
# - $u_m(x) \longleftarrow D_m(x)$ (Wigner-D function, $x$ is an Euler-angle tuple or quaternion)
#     - collapses to spherical harmonics on the first layer
# - GFT $\longleftarrow$ SO(3) FFT (with Gaussian resampling)
# - number of coefficients $\longleftarrow 2$: $m \in [0, 4]$ (increasing beyond that contribute little to accuracy)
#
# <img src="assets/sh_basis.png"> 
#
# ---
#
# [*] T. S. Cohen, M. Geiger, J. Koehler, and M. Welling, “Spherical CNNs,” no. 3, pp. 1–15, 2018.

# %%

