"""
cgenn's n-body experiment with the layer of `Flash Clifford <https://github.com/maxxxzdn/flash-clifford>`_ (Zhdanov, 2025), which offers it as a faster replacement for
the layers of cgenn's n-body MLPs. flash-clifford ships the layer only, so the graph network around it is cgenn's.
"""

from .nbody import CEMLP, NBodyCGGNN
