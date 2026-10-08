import sympy
import torch
from einops import einsum
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter
from torch import nn
from kingdon import MultiVector, add_operator
from kingdon.multivector import Scalar

from ..utils import invariants, materialize_constants, sigmoid


@add_operator(symbolic=True, codegen_symbolcls=sympy.Symbol)
def mvsilu(X: MultiVector, a: Scalar[None], b: Scalar[None]) -> MultiVector:
    """:class:`MVSiLU`, with a[k] and b[k] the map from the invariant of the k-th grade of X to its gate."""
    k = X.gradeidx_of_blades
    return einsum(X, (einsum(invariants(X)[k], a[k], "... f, f -> ... f") + b[k]).map(sigmoid), "... f, ... f -> ... f")


class MVSiLU(LazyModuleMixin, nn.Module):
    """Gate every grade by a sigmoid of an invariant of it, which the group cannot see."""

    a: UninitializedParameter
    b: UninitializedParameter

    def __init__(self):
        super().__init__()

        self.a = UninitializedParameter()
        self.b = UninitializedParameter()

    def initialize_parameters(self, input: MultiVector):
        if not self.has_uninitialized_params():
            return

        with torch.no_grad():
            input = materialize_constants(input)
            self.a.materialize((len(input.grades), input.shape[-1]))
            self.b.materialize((len(input.grades), input.shape[-1]))
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.a)
        nn.init.zeros_(self.b)

    def forward(self, input: MultiVector) -> MultiVector:
        input = materialize_constants(input)
        return mvsilu(input, *(input.algebra.scalar(e=p) for p in (self.a, self.b)))
