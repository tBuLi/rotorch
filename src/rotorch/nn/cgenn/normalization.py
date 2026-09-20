import einops
import torch
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter
from torch import nn
from kingdon import MultiVector

from .utils import EPS, full_precision, grade_of_blades, materialize_constants, norm

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
            self.grades = input.grades
            self.register_buffer("blade_grades", grade_of_blades(input))
            self.a.materialize((len(self.grades), input.shape[-1]))
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.a, self.init)

    def no_weight_decay(self):
        return {"a"}

    @full_precision
    def forward(self, input: MultiVector) -> MultiVector:
        input = materialize_constants(input)
        s_a = torch.sigmoid(self.a)
        # Interpolate between 1 and the norm of each grade. A null grade has no norm to speak of,
        # so the entries need broadcasting against each other before they can be stacked.
        norms = [s_a[i] * (norm(input.grade(g)) - 1) + 1 for i, g in enumerate(self.grades)]
        norms = torch.stack(torch.broadcast_tensors(*norms))
        scale = input.algebra.multivector(1 / (norms[self.blade_grades] + EPS), keys=input.keys())
        return einops.einsum(input, scale, "..., ... -> ...")
