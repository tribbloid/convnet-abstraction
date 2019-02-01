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
import networkx as nx

from graphPlot import drawGraph
from const import *

if '' in sys.path:
    sys.path.remove('')

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

print(sys.path)

plt.rcParams['figure.figsize'] = [10, 10]

# %% [markdown]
# ## Data Augmentation
#
# - is a higher-order function$*$, specifically, a unary (arity=1) operator: $A_{ug}$ that change a signal to anther signal
#     - (as in "Laplace operator" & "operator overloading")

# %%

g = nx.DiGraph(directed=True)

n = 'numbers'
n2 = ' numbers'
f = 'function'
f2 = ' function'
o = 'operator*'
c = 'currying*'
fl = 'functional*'

g.add_nodes_from([n])
g.add_nodes_from([f, fl, o, c])
g.add_nodes_from([n2, f2])

g.add_edge(n, f, wedge=True)
g.add_edge(f, n2)
g.add_edge(f, o, wedge=True)
g.add_edge(o, f2)
g.add_edge(f, fl, wedge=True)
g.add_edge(fl, n2)
g.add_edge(n, c, wedge=True)
g.add_edge(c, f2)

drawGraph(g, font='humor sans', arrow='-|>')

plt.show()

# %%
list(map(lambda x: x * x, range(1, 5)))  # Map: (R -> R) -> (R^n -> R^n)

# %% [markdown]
# - can be applied to an ordindary function: $A_{ug} \circ f(x)$
#
# - can be applied with operand delimiter to a multivariable function: $A_{ug}|_{x} \circ U_{ag}|_{y} \circ w(x, y)$

# %% [markdown]
# ## Data Augmentation
#
# Is very situational:
#     
# - With gravity vs without gravity
#
# - [insert pictures]

# %% [markdown]
# ## Data Augmentation
#
# Is very situational:
#     
# - Fixed air pressure vs dependent air pressure
#
# - [Insert pictures]

# %% [markdown]
# ## Data Augmentation
#
# Hypothesis 1 (**equivariance**): Should be applicable to any layer

# %%


g = nx.DiGraph(directed=True)

fs = [
    "$f(.)$",
    "$f_+(.)$",
    "$f_{++}(.)$",
    "$f_{+++}(.)$",
    "..."
]

afs = [
    "$A_{ug} \circ f(.)$",
    "$U_{ga} \circ f_+(.)$",
    "$G_{au} \circ f_{++}(.)$",
    "$A_{gu} \circ f_{++}(.)$",
    " ... "
]

for i in range(0, 4):
    g.add_edge(fs[i], fs[i + 1], text='||')
    g.add_edge(afs[i], afs[i + 1], text='||')
#     g.add_edge(fs[i], afs[i])

g2 = g.copy()

for i in range(0, 4):
    g2.add_edge(fs[i], afs[i], text='Augment')

g2.add_edge(afs[0], fs[1], text='Invariant || (DON\'T DO THIS)')

drawGraph(g2, font='humor sans', layoutG=g)

plt.show()

# %% [markdown]
# ## Data Augmentation
#
# Hypothesis 1 (equivariance): Should be applicable to any layer
#
# $$
# U_{ga} \circ f_+(y) = U_{ga}|_y \circ <f(x), w(x, y)> dx = <A_{ug} \circ f(x), w(x, y)> dx
# $$
#
# Hypothesis 2 (**transitivity**): in any signal we can find a reference point $x_0$, such that values of **any other points** can be found on $x_0$ of an augmented signal:
#
# $$
# \forall x : f(x) = \Big( \bar{A}_{ug} \circ f(x) \Big) (x_0)
# $$
#
# What does that even mean?
#
# -- [insert pictures that demonstrates transitive augmentation]
#

# %% [markdown]
# ## Data Augmentation
#
# Hypothesis 1 (equivariance): Should be applicable to any layer
#
# $$
# U_{ga} \circ f_+(y) = U_{ga}|_y \circ <f(x), w(x, y)> dx = <A_{ug} \circ f(x), w(x, y)> dx
# $$
#
# Hypothesis 2 (transitivity): in any signal we can find a reference point $x_0$, such that values of **any other points** can be found on $x_0$ of an augmented signal:
#
# $$
# \forall x : f(x) = \Big( \bar{A}_{ug} \circ f(x) \Big) (x_0)
# $$
#
# **Combining all together**:
#
# $$
# f_+(y) = \Big( \bar{U}_{ga} \circ f_+(y) \Big)(y_0) = <\bar{A}_{ug} \circ f(x), w(x, y_0)> dx = <\bar{A}_{ug} \circ f(x), w_0(x)> dx
# $$
#
# Looks familiar yet?
#
# $$
# conv(f(- \Delta), w_0(\Delta)) = corr(f(\Delta), w_0(\Delta)) = \int\limits_{x \in \text{Manifold}} f(\Delta + x) w_0(x) d x = <f(\Delta + x), w_0(x)> d x
# $$
#
#


# %% [markdown]
# ## Data Augmentation
#
# - Looks like ConvNet is a FC Net with translation as family of augmentations!
#
# - Still feel strange? Let's do a thought experiment:
#
#     - Assuming you start with a deep FC-ResNet, with all weights initialized to 0
#     - Then you run back-propagation once, but this time instead of using 1 picture at time, you use a minibatch consisting of **all augmented pictures** (including the original one)
#     - What will be the weight?
#

# %% [markdown]
#

# %% [markdown]
#

# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Glossary
#
# - A monoid is anything compatible with MapReduce
#
# ```python
# apple.merge(pen).merge(pineapple).merge(pen).merge(face).merge(palm) =
# apple.merge(pen)                    # core 1
#   .merge(pineapple.merge(pen))      # core 2
#   .merge(face.merge(palm))          # core 3
# ```

# %% [markdown]
# ## Glossary
#
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
