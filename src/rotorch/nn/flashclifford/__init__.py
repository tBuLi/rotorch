"""
The layer of `Flash Clifford <https://github.com/maxxxzdn/flash-clifford>`_ (Zhdanov, 2025), written against kingdon multivectors: where flash-clifford hand-writes a
triton kernel for each of Cl(2) and Cl(3), one kingdon operator holds the same fused computation for any algebra.
"""

from .layer import Layer, gelu, gelu_wgp, rms_norm
