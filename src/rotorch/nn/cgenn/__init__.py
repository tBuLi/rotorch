"""
The layers of `Clifford Group Equivariant Neural Networks
<https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks>`_ (cgenn),
written against kingdon multivectors rather than a dense array of all :math:`2^n` blades.
"""

from .gp import GeometricProduct
from .fcgp import FullyConnectedGeometricProduct
from .linear import MVLinear
from .mvlayernorm import MVLayerNorm
from .mvsilu import MVSiLU
from .normalization import NormalizationLayer
from .utils import no_weight_decay, parameter_groups
