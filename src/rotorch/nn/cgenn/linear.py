import math

import einops
import torch
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter
from torch import nn
from kingdon import MultiVector

from ..utils import grade_of_blades, materialize_constants

def gradewise_linear(X: MultiVector, weights: MultiVector[None]) -> MultiVector:
    """
    Apply a weight to every grade of X seperatelly.
    """
    tot = 0
    for g, w in zip(X.grades, weights):
        tot += w * X.grade(g)
    return tot

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
            blade_grades = grade_of_blades(materialize_constants(input))
            if not self.gradewise:  # Without gradewise every grade shares one matrix.
                blade_grades = torch.zeros_like(blade_grades)
            self.register_buffer("blade_grades", blade_grades)
            self.weight.materialize((1 + int(blade_grades.max()), self.out_features, self.in_features))
            if self.bias is not None:
                self.bias.materialize((self.out_features,))
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=1 / math.sqrt(self.in_features))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input: MultiVector) -> MultiVector:
        input = materialize_constants(input)
        # A multivector holding the matrix of each blade, rather than the matrix of each grade.
        weight = input.algebra.multivector(self.weight[self.blade_grades], keys=input.keys())
        result = einops.einsum(input, weight, "... i, o i -> ... o")
        if self.bias is not None:
            result = result + input.algebra.scalar(e=self.bias)
        return result
