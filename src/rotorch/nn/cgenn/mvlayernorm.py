import sympy
import torch
from einops import reduce
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter
from torch import nn
from kingdon import MultiVector, add_operator

from ..utils import EPS, _root, materialize_constants, scalar_normsq


@add_operator(symbolic=True, codegen_symbolcls=sympy.Symbol)
def layernorm(X: MultiVector, a) -> MultiVector:
    """:class:`MVLayerNorm`."""
    return a * X / (reduce(scalar_normsq(X).map(_root), "... o -> ... 1", "mean") + EPS)


class MVLayerNorm(LazyModuleMixin, nn.Module):
    """Divide by the norm of the input averaged over the channels, times a learned scale."""

    a: UninitializedParameter

    def __init__(self):
        super().__init__()

        self.a = UninitializedParameter()

    def initialize_parameters(self, input: MultiVector):
        if not self.has_uninitialized_params():
            return

        with torch.no_grad():
            self.a.materialize((input.shape[-1],))
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.a)

    def forward(self, input: MultiVector) -> MultiVector:
        input = materialize_constants(input)
        return layernorm(input, input.algebra.scalar(e=self.a))
