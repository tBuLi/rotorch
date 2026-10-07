"""
The five experiments of `Clifford Group Equivariant Neural Networks
<https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks>`_ (cgenn): the
O(3) and O(5) invariant regressions, the volume of a convex hull, charged n-body, and
tagging the jets of top quarks.
"""

from .hulls import ConvexHullCGMLP
from .lorentz import CGLayer, FCLayer, GradeGate, LorentzCGGNN, fc_layer, fc_layer_unnormalized, grade_gate
from .nbody import CEMLP, CEMLPLayer, EGCL, NBodyCGGNN, cemlp_layer
from .o3 import O3CGMLP
from .o5 import O5CGMLP
