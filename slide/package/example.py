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
from __future__ import absolute_import
import os
import sys
    

print(sys.path)
for p in sys.path:
    print(p)

print("=================")

import networkx
print(os.path.abspath(networkx.__file__))

from networkx import drawing
# -


