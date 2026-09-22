"""
The layers of the `Geometric Algebra Transformer
<https://github.com/Qualcomm-AI-research/geometric-algebra-transformer>`_ (GATr), written against
kingdon multivectors rather than a dense array of all :math:`2^n` blades. Where the reference
tabulates its equivariant maps, its products and its duals as constant arrays, here each is a
kingdon expression compiled over the blades the data turns out to have.

Every layer carries two multivectors, the geometric one and the scalars that ride along with it,
and hands both back. The two meet only on the scalar blade, the one place the group cannot tell
them apart.
"""

from .attention import GeometricAttention, SelfAttention
from .bilinear import GeometricBilinear
from .block import GATrBlock
from .layernorm import EquiLayerNorm
from .linear import EquiLinear
from .mlp import GeoMLP
from .nonlinearity import ScalarGatedNonlinearity
