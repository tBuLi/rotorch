import math

import sympy
import torch
from einops import einsum
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter
from torch import nn
from kingdon import MultiVector, add_operator
from kingdon.multivector import Scalar

from ..utils import materialize_constants


@add_operator(symbolic=True, codegen_symbolcls=sympy.Symbol)
def linear(X: MultiVector, W: Scalar, b: Scalar = 0) -> MultiVector:
    """:class:`MVLinear` with gradewise False, W the one matrix of every grade of X."""
    return einsum(X, W, "... i, o i -> ... o") + b


@add_operator(symbolic=True, codegen_symbolcls=sympy.Symbol)
def gradewise_linear(X: MultiVector, W: Scalar[None], b: Scalar = 0) -> MultiVector:
    """:class:`MVLinear`, with W[k] the matrix of the k-th grade of X."""
    return einsum(X, W[X.gradeidx_of_blades], "... i, o i -> ... o") + b


class MVLinear(LazyModuleMixin, nn.Module):
    """Linear map that gives every grade its own mixing matrix, unless gradewise is False."""

    weight: UninitializedParameter
    bias: UninitializedParameter

    def __init__(self, in_features, out_features, gradewise=True, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gradewise = gradewise
        self.weight = UninitializedParameter()
        if bias:
            self.bias = UninitializedParameter()
        else:
            self.register_parameter("bias", None)

    def initialize_parameters(self, input: MultiVector):
        if not self.has_uninitialized_params():
            return

        with torch.no_grad():
            grades = (len(materialize_constants(input).grades),) if self.gradewise else ()
            self.weight.materialize((*grades, self.out_features, self.in_features))
            if self.bias is not None:
                self.bias.materialize((self.out_features,))
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=1 / math.sqrt(self.in_features))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input: MultiVector) -> MultiVector:
        input = materialize_constants(input)
        return (gradewise_linear if self.gradewise else linear)(input, *(input.algebra.scalar(e=p) for p in (self.weight, self.bias) if p is not None))
