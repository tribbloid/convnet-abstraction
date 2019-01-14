# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import pygraphviz as pgv

G=pgv.AGraph()
G.add_node('a')
G.add_edge('b','c')
G
# -


