import math

import einops
import kingdon.einops_backend  # noqa: F401  Registers MultiVector with einops.
import torch
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter
from torch import nn
from kingdon import MultiVector

from .gp import number_of_weights_wgp, wgp
from .linear import MVLinear
from .normalization import NormalizationLayer
from .utils import full_precision, materialize_constants, register

def insert_out_features(X: MultiVector) -> MultiVector:
    """Make room for the output features, so that the weights broadcast over them."""
    return einops.rearrange(materialize_constants(X), "... f -> ... 1 f")


class FullyConnectedGeometricProduct(LazyModuleMixin, nn.Module):
    """
    Known as the FullyConnectedSteerableGeometricProductLayer in cgenn: a
    GeometricProduct whose weights mix the input features as well.
    """

    weight: UninitializedParameter

    def __init__(self, in_features, out_features, include_first_order=True, normalization_init=0):
        super().__init__()
        self.wgp = None
        self.in_features = in_features
        self.out_features = out_features
        self.include_first_order = include_first_order
        self.weight = UninitializedParameter()
        if normalization_init is not None:
            self.normalization = NormalizationLayer(normalization_init)
        else:
            self.normalization = nn.Identity()
        self.linear_right = MVLinear(in_features, in_features, bias=False)
        if include_first_order:
            self.linear_left = MVLinear(in_features, out_features, bias=True)

    def initialize_parameters(self, input: MultiVector):
        if not self.has_uninitialized_params():
            return

        self.algebra = input.algebra
        self.wgp = register(self.algebra, wgp)

        with torch.no_grad():
            n_weights = number_of_weights_wgp(input, input)
            self.weight.materialize((n_weights, self.out_features, self.in_features))
            self.reset_parameters()

    def reset_parameters(self):
        std = 1 / math.sqrt(self.in_features * (self.algebra.d + 1))
        torch.nn.init.normal_(self.weight, std=std)
        
    @full_precision
    def forward(self, input: MultiVector) -> MultiVector:
        input_right = self.linear_right(input)
        input_right = self.normalization(input_right)
        weights = self.algebra.scalar(e=self.weight)

        product = self.wgp(insert_out_features(input), insert_out_features(input_right), weights)
        product = einops.reduce(product, "... o f -> ... o", "sum")  # Contract the input features.

        if self.include_first_order:
            return (self.linear_left(input) + product) / math.sqrt(2)
        else:
            return product
