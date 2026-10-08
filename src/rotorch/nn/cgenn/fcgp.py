import math

import sympy
import torch
from einops import einsum
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter
from torch import nn
from kingdon import MultiVector, add_operator
from kingdon.multivector import Scalar

from .gp import cayley, number_of_weights_wgp, signed
from .linear import MVLinear, gradewise_linear
from .normalization import NormalizationLayer, normalize
from ..utils import materialize_constants


def fc_wgp(X: MultiVector, Y: MultiVector, w: Scalar) -> MultiVector:
    """:func:`~rotorch.nn.cgenn.gp.wgp` with w[k] the matrix that mixes the features of the k-th path."""
    J, P = cayley(X, Y)
    return einsum(X.blades, Y.blades[J], signed(w)[P], "a ... i, a ... i, a o i -> ... o")


@add_operator(symbolic=True, codegen_symbolcls=sympy.Symbol)
def fc_geometric_product(X: MultiVector, Wr: Scalar[None], n: Scalar, Wl: Scalar[None], bl, w: Scalar) -> MultiVector:
    """:class:`FullyConnectedGeometricProduct`."""
    return (gradewise_linear(X, Wl, bl) + fc_wgp(X, normalize(gradewise_linear(X, Wr), n), w)) / math.sqrt(2)


@add_operator(symbolic=True, codegen_symbolcls=sympy.Symbol)
def fc_geometric_product_unnormalized(X: MultiVector, Wr: Scalar[None], Wl: Scalar[None], bl, w: Scalar) -> MultiVector:
    """:class:`FullyConnectedGeometricProduct` without its normalization."""
    return (gradewise_linear(X, Wl, bl) + fc_wgp(X, gradewise_linear(X, Wr), w)) / math.sqrt(2)


class FullyConnectedGeometricProduct(LazyModuleMixin, nn.Module):
    """
    Known as the FullyConnectedSteerableGeometricProductLayer in cgenn: a
    GeometricProduct whose weights mix the input features as well.
    """

    weight: UninitializedParameter

    def __init__(self, in_features, out_features, normalization_init=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = UninitializedParameter()
        self.normalization = NormalizationLayer(normalization_init) if normalization_init is not None else None
        self.linear_right = MVLinear(in_features, in_features, bias=False)
        self.linear_left = MVLinear(in_features, out_features)

    def initialize_parameters(self, input: MultiVector):
        if not self.has_uninitialized_params():
            return

        self.algebra = input.algebra
        # The layer is one operator, so its modules are never run: each is sized on the input, whose keys and features are those of what it takes.
        for module in (self.linear_right, self.normalization, self.linear_left):
            if module is not None:
                module.initialize_parameters(input)

        with torch.no_grad():
            self.weight.materialize((number_of_weights_wgp(input, input), self.out_features, self.in_features))
            self.reset_parameters()

    def reset_parameters(self):
        std = 1 / math.sqrt(self.in_features * (self.algebra.d + 1))
        torch.nn.init.normal_(self.weight, std=std)

    def forward(self, input: MultiVector) -> MultiVector:
        input = materialize_constants(input)
        normalization = () if self.normalization is None else (self.normalization.a,)
        product = fc_geometric_product if normalization else fc_geometric_product_unnormalized
        params = self.linear_right.weight, *normalization, self.linear_left.weight, self.linear_left.bias, self.weight
        return product(input, *(input.algebra.scalar(e=p) for p in params))
