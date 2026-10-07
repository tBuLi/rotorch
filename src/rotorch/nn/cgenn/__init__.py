"""
The layers of `Clifford Group Equivariant Neural Networks
<https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks>`_ (cgenn),
written against kingdon multivectors rather than a dense array of all :math:`2^n` blades.
"""

from .gp import GeometricProduct, geometric_product, number_of_weights_wgp, paths, wgp
from .fcgp import FullyConnectedGeometricProduct, fc_geometric_product, fc_geometric_product_unnormalized, fc_wgp
from .linear import MVLinear, gradewise_linear, linear
from .mvlayernorm import MVLayerNorm, layernorm
from .mvsilu import MVSiLU, mvsilu
from .normalization import NormalizationLayer, normalize
