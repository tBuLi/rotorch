import einops
import torch
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter
from torch import nn
from kingdon import MultiVector

from ..utils import grade_of_blades, materialize_constants, mag2, norm


class MVSiLU(LazyModuleMixin, nn.Module):
    """Gate every grade by a sigmoid of an invariant of it, which the group cannot see."""

    a: UninitializedParameter
    b: UninitializedParameter

    def __init__(self, invariant="mag2"):
        super().__init__()

        self.invariant = {"mag2": mag2, "norm": norm}[invariant]
        self.a = UninitializedParameter()
        self.b = UninitializedParameter()

    def initialize_parameters(self, input: MultiVector):
        if not self.has_uninitialized_params():
            return

        with torch.no_grad():
            input = materialize_constants(input)
            self.grades = input.grades
            self.register_buffer("blade_grades", grade_of_blades(input))
            self.a.materialize((len(self.grades), input.shape[-1]))
            self.b.materialize((len(self.grades), input.shape[-1]))
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.a)
        nn.init.zeros_(self.b)

    def forward(self, input: MultiVector) -> MultiVector:
        input = materialize_constants(input)
        gates = [torch.sigmoid(self.a[i] * (input.e if g == 0 else self.invariant(input.grade(g))) + self.b[i])
                 for i, g in enumerate(self.grades)]
        gates = torch.stack(torch.broadcast_tensors(*gates))
        gates = input.algebra.multivector(gates.index_select(0, self.blade_grades), keys=input.keys())
        return einops.einsum(input, gates, "..., ... -> ...")
