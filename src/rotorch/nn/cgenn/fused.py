"""
The cgenn layers as functions of multivectors, their parameters included. Registered as a single operator,
a stack of them is traced into one generated function, einops calls and all.
"""
import math

import sympy
from einops import einsum, reduce
from kingdon.multivector import Scalar, MultiVector
from torch import nn
from torch.nn.parameter import UninitializedParameter

from .gp import wgp
from ..utils import EPS, _root, mag2, materialize_constants, norm, register, scalar_normsq


class sigmoid(sympy.Function):
    """The logistic function, printed as one torch.sigmoid rather than a negation, an exp, a sum and a reciprocal, each a node for autograd."""

    def fdiff(self, argindex=1):
        return self * (1 - self)

    def _torchcode(self, printer):
        return f"torch.sigmoid({printer._print(self.args[0])})"


def linear(X: MultiVector, W: Scalar[None], b: Scalar = 0) -> MultiVector:
    """:class:`~rotorch.nn.cgenn.MVLinear`, with W[k] the matrix of the k-th grade of X."""
    return einsum(X, W[X.gradeidx_of_blades], "... i, o i -> ... o") + b


def mvsilu(X, a, b):
    """:class:`~rotorch.nn.cgenn.MVSiLU`."""
    return sum(X.grade(g) * sigmoid(a[k].e * (X.e if g == 0 else mag2(X.grade(g))) + b[k].e) for k, g in enumerate(X.grades))

def normalize(X, a):
    """:class:`~rotorch.nn.cgenn.NormalizationLayer`."""
    return sum(X.grade(g) / (sigmoid(a[k].e) * (norm(X.grade(g)) - 1) + 1 + EPS) for k, g in enumerate(X.grades))


def cemlp_layer(X, W: Scalar[None], b, a: Scalar[None], c: Scalar[None], Wr: Scalar[None], n: Scalar[None], Wl: Scalar[None], bl, w: Scalar[None], s):
    """A layer of :class:`~rotorch.models.cgenn.nbody.CEMLP`: MVLinear, MVSiLU, GeometricProduct and MVLayerNorm."""
    H = mvsilu(linear(X, W, b), a, c)
    Y = (linear(H, Wl, bl) + wgp(H, normalize(linear(H, Wr), n), w)) / math.sqrt(2)
    return s * Y / (reduce(scalar_normsq(Y).map(_root), "... o -> ... 1", "mean") + EPS)


class CEMLPLayer(nn.Sequential):
    """
    A layer of :class:`~rotorch.models.cgenn.nbody.CEMLP` -- its MVLinear, MVSiLU, GeometricProduct and MVLayerNorm, parameters and all -- computed by :func:`cemlp_layer`,
    one operator: under the triton backend one kernel forward and one backward.
    """

    def forward(self, X: MultiVector) -> MultiVector:
        linear, silu, gp, layernorm = self
        # The lazy modules size their parameters on the first input, and the operator is generated then too, not inside a torch.compile trace.
        # An isinstance, unlike has_uninitialized_params, that trace goes through.
        if isinstance(linear.weight, UninitializedParameter):
            super().forward(X)
        params = linear.weight, linear.bias, silu.a, silu.b, gp.linear_right.weight, gp.normalization.a, gp.linear_left.weight, gp.linear_left.bias, gp.weight, layernorm.a
        return register(X.algebra, cemlp_layer, codegen_symbolcls=sympy.Symbol)(materialize_constants(X), *(X.algebra.scalar(e=p) for p in params))
