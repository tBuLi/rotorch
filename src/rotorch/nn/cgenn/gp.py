import itertools
import math

from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter
from torch import nn
import torch
from kingdon import MultiVector

from .linear import MVLinear
from .normalization import NormalizationLayer
from ..utils import register


def number_of_weights_wgp(X: MultiVector, Y: MultiVector) -> int:
    i = 0
    for gx, gy in itertools.product(X.grades, Y.grades):
        Z = X.grade(gx) * Y.grade(gy)
        i += len(Z.grades)
    return i

def wgp(X: MultiVector, Y: MultiVector, weights: MultiVector[None]) -> MultiVector:
    """
    Compute the weighted geometric product between X and Y.
    The multivectors are mutiplied grade-wise, and a unique weight
    is applied to each grade in the output.
    """
    tot = 0
    i = 0
    for gx, gy in itertools.product(X.grades, Y.grades):
        Z = X.grade(gx) * Y.grade(gy)
        for gz in Z.grades:
            tot += weights[i] * Z.grade(gz)
            i += 1
    return tot

class GeometricProduct(LazyModuleMixin, nn.Module):
    """
    Known as the SteerableGeometricProductLayer in cgenn. The "steerable" is dropped
    here because a geometric product of multivectors is always steerable so long as one
    resists the temptation to touch coefficients: the weights
    are scalars applied per grade, so they commute with the action of the group.
    """

    weight: UninitializedParameter

    def __init__(self, features, include_first_order=True, normalization_init=0):
        super().__init__()
        self.wgp = None
        self.features = features
        self.include_first_order = include_first_order
        self.weight = UninitializedParameter()
        if normalization_init is not None:
            self.normalization = NormalizationLayer(normalization_init)
        else:
            self.normalization = nn.Identity()
        self.linear_right = MVLinear(features, features, bias=False)
        if include_first_order:
            self.linear_left = MVLinear(features, features, bias=True)

    def initialize_parameters(self, input: MultiVector):
        if not self.has_uninitialized_params():
            return

        self.algebra = input.algebra
        self.wgp = register(self.algebra, wgp)

        with torch.no_grad():
            self.weight.materialize((number_of_weights_wgp(input, input), self.features))
            self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, std=1 / math.sqrt(self.algebra.d + 1))

    def forward(self, input: MultiVector) -> MultiVector:
        input_right = self.linear_right(input)
        input_right = self.normalization(input_right)
        weights = self.algebra.scalar(e=self.weight)

        if self.include_first_order:
            return (self.linear_left(input) + self.wgp(input, input_right, weights)) / math.sqrt(2)
        else:
            return self.wgp(input, input_right, weights)
