import itertools
import math

import sympy
import torch
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter
from torch import nn
from kingdon import MultiVector, add_operator
from kingdon.multivector import Scalar

from .linear import MVLinear, gradewise_linear
from .normalization import NormalizationLayer, normalize
from ..utils import materialize_constants


def paths(X: MultiVector, Y: MultiVector):
    """Every grade of X times every grade of Y, split into the grades of the product: what a weighted geometric product gives a weight each."""
    for gx, gy in itertools.product(X.grades, Y.grades):
        Z = X.grade(gx) * Y.grade(gy)
        yield from (Z.grade(gz) for gz in Z.grades)


def number_of_weights_wgp(X: MultiVector, Y: MultiVector) -> int:
    return sum(1 for _ in paths(X, Y))


def wgp(X: MultiVector, Y: MultiVector, weights: MultiVector[None]) -> MultiVector:
    """The geometric product of X and Y, with a weight for each of its :func:`paths`."""
    return sum(weights[k] * Z for k, Z in enumerate(paths(X, Y)))


@add_operator(symbolic=True, codegen_symbolcls=sympy.Symbol)
def geometric_product(X: MultiVector, Wr: Scalar[None], n: Scalar[None], Wl: Scalar[None], bl, w: Scalar[None]) -> MultiVector:
    """:class:`GeometricProduct`."""
    return (gradewise_linear(X, Wl, bl) + wgp(X, normalize(gradewise_linear(X, Wr), n), w)) / math.sqrt(2)


class GeometricProduct(LazyModuleMixin, nn.Module):
    """
    Known as the SteerableGeometricProductLayer in cgenn. The "steerable" is dropped
    here because a geometric product of multivectors is always steerable so long as one
    resists the temptation to touch coefficients: the weights
    are scalars applied per grade, so they commute with the action of the group.
    """

    weight: UninitializedParameter

    def __init__(self, features, normalization_init=0):
        super().__init__()
        self.features = features
        self.weight = UninitializedParameter()
        self.normalization = NormalizationLayer(normalization_init)
        self.linear_right = MVLinear(features, features, bias=False)
        self.linear_left = MVLinear(features, features)

    def initialize_parameters(self, input: MultiVector):
        if not self.has_uninitialized_params():
            return

        self.algebra = input.algebra
        # The layer is one operator, so its modules are never run: each is sized on the input, whose keys and features are those of what it takes.
        for module in (self.linear_right, self.normalization, self.linear_left):
            module.initialize_parameters(input)

        with torch.no_grad():
            self.weight.materialize((number_of_weights_wgp(input, input), self.features))
            self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, std=1 / math.sqrt(self.algebra.d + 1))

    def forward(self, input: MultiVector) -> MultiVector:
        input = materialize_constants(input)
        params = self.linear_right.weight, self.normalization.a, self.linear_left.weight, self.linear_left.bias, self.weight
        return geometric_product(input, *(input.algebra.scalar(e=p) for p in params))
