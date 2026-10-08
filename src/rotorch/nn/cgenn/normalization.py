import sympy
import torch
from einops import einsum
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter
from torch import nn
from kingdon import MultiVector, add_operator
from kingdon.multivector import Scalar

from ..utils import EPS, magnitudes, materialize_constants, sigmoid


@add_operator(symbolic=True, codegen_symbolcls=sympy.Symbol)
def normalize(X: MultiVector, a: Scalar) -> MultiVector:
    """:class:`NormalizationLayer`, with a[k] how far the k-th grade of X goes towards its normalized self."""
    shrink = einsum(a.map(sigmoid), magnitudes(X) - 1, "k f, k ... f -> k ... f") + 1 + EPS
    return einsum(X, (1 / shrink)[X.gradeidx_of_blades], "... f, ... f -> ... f")


class NormalizationLayer(LazyModuleMixin, nn.Module):
    """Interpolate grade-wise between the input and its normalized version."""

    a: UninitializedParameter

    def __init__(self, init: float = 0):
        super().__init__()

        self.init = init
        self.a = UninitializedParameter()

    def initialize_parameters(self, input: MultiVector):
        if not self.has_uninitialized_params():
            return

        with torch.no_grad():
            input = materialize_constants(input)
            self.a.materialize((len(input.grades), input.shape[-1]))
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.a, self.init)

    def forward(self, input: MultiVector) -> MultiVector:
        input = materialize_constants(input)
        return normalize(input, input.algebra.scalar(e=self.a))
