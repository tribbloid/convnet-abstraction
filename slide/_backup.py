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

# %% [markdown]
# ## Equivariance
#
# input f(x): (F = R^n => S) (can be a graph, use R^n for generalisability)
# fully connected layer w(x, x+): R^n x R^n => S
# morphing operator: A {f(x)} = AL(f(AR(x))): F => F
#
# output g(x+): R^n => S = <f(x), w(x, x+)> dx
#
# equivariant condition: <A {f(x)}, w(x, x+)> dx = Q {<f(x), w(x, x+)> dx}    \forall f(x)
#
# assuming only AR is in consideration (e.g. a translation):
#
# <f(AR x), w(x, x+)> dx = <f(x), w(x, QR x+))> dx
#
# <f(x), w(AR^-1 x, x+)> dx = <f(x), w(x, QR x+))> dx     \forall f(x)
#
# <f(x), w(AR^-1 x, QR^-1 y+)> dx = <f(x), w(x, y+))> dx     \forall f(x)
#
# w(AR^-1 x, QR^-1 y+) = w(x, y+)
#
# w(AR x, QR y+) = w(x, y+)
#
# as a result:
#
# assuming that x+ = H x, v(x) = w(H x, QH H x) = w(x, H x), then:
#
# g(x+) = g(H x) = <f(x), w(x, H x)> dH =
#
# assuming that x+ = Q x0, then
#
# \begin{align}
# g(x+) &= g(Q x0) = <f(y), w(y, Q x0)> dy \\
#     &= <f(y), w(P^{-1} y, x0)> dy \\
#     &= conv(f(y), w(y, x0))(P^{-1})
# \end{align}
#
# QED
#
#
#

# %% [markdown]
#
# ## Data Augmentation - Similarity
#
# Hypothesis 2:
#
# $$
# <f_1(x), f_2(x)> dx = k <A_{ug} \circ f_1(x), A_{ug} \circ f_2(x)> dx
# $$
#
# OR: inner-product preserving (isometry) + scaling
#     (I know it sounds even worse, please don't judge me)


# %% [markdown]
# ## Data Augmentation - Put It All Together
#
# - linear fully-connected dense
#
# \begin{align}
# & & U_{ga} \circ f_+(y) = U_{ga}|_y \circ <f(x), w(x, y)> dx &= <A_{ug} \circ f(x), w(x, y)> dx \\
# \text{(similarity)} \\
# & \Longrightarrow & <f(x), U_{ga}|_y \circ w(x, y)> dx &= \frac{1}{k} <A_{ug}^{-1} \circ A_{ug} \circ f(x), A_{ug}^{-1}|_x \circ w(x, y)> dx \\
# & & &= \frac{1}{k} <f(x), A_{ug}^{-1}|_x \circ w(x, y)> dx \\
# \forall f(x) \\
# & \Longrightarrow & U_{ga}|_y \circ w(x, y) &= \frac{1}{k} A_{ug}^{-1}|_x \circ w(x, y) \\
# & & & \text{(lemma of weight sharing!)}\\
# \end{align}

# %% [markdown]
# - now assuming that there is a reference point $y_0$
#     - ... such that for any $y$ there is a data augmentation $\bar{U}_{ga}$ that satisfy **transitivity**:
#     
#     $f_+(y) = \bar{U}_{ga} \circ f_+(y_0)$
#     
#     $w(x, y) = \bar{U}_{ga}|_y \circ w(x, y_0)$
#
# \begin{align}
# & & f_+(y) &= <f(x), w(x, y)> dx = <f(x), \bar{U}_{ga}|_y \circ w(x, y_0)> dx \\
# & \text{(lemma of weight sharing)} & &= <f(x), \bar{A}_{ug}^{-1}|_x \circ w(x, y_0)> dx \\
# & \text{(similarity)} & &= <\bar{A}_{ug} \circ f(x), w(x, z_0)> dx = <\bar{A}_{ug} \circ f(x), w_0(x)> dx
# \end{align}
#
# - looks familiar yet?
#
# $$
# conv(f(- \Delta), w_0(\Delta)) = corr(f(\Delta), w_0(\Delta)) = \int_{\text{Manifold}} f(\Delta + x) w_0(x) d x = <f(\Delta + x), w_0(x)> d x
# $$

# %%
import numpy as np

left = np.zeros((3, 2))
right = np.zeros((2, 4))

np.matmul(left, right)

# %% [markdown]
#
## Glossary

# - A monoid is anything compatible with MapReduce
#
# ```python
# apple.merge(pen).merge(pineapple).merge(pen).merge(face).merge(palm) =
# apple.merge(pen)                    # core 1
#   .merge(pineapple.merge(pen))      # core 2
#   .merge(face.merge(palm))          # core 3
# ```

# %% [markdown]
#
## Glossary

# - A group member is an invertible monoid
#
# ```python
# pineapplepen.merge(-pen) = pineapple
# ```
#
# A commutative (Abelian) monoid/group member is an monoid i which order of reduce doesn't matter
#
# ```python
# pineapple.merge(pen) = pen.merge(pineapple)
# ```


# %% [markdown] {"slideshow": {"slide_type": "slide"}}
#

# %% [markdown]
# ## Glossary
#
# - Higher-order function (a.k.a. operator) is
# - Group action is an operator that is also a group member
#
#

# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# ## What is conv? (actually cross-correlation)
#
# assuming f_1, f_2: (F = R^n => S)
#
# defined for an operator A: F => F
#
# conv(f_1, f_2)(A) = <f_1(x), A {f_2(x)}> dx: (F => F) => S
#
# The common assumption is that operator A can be broken into left AL and right AR:
#
# A {f(x)} = AL(f(AR(x)))

# %% [markdown]
# ## How does it help? - Equivariance
#
# ... define equivariance

# %% [markdown]
#
