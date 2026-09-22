import math

import einops
import torch
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter
from torch import nn
from kingdon import MultiVector

from ..utils import degenerate, insert_out_features, materialize_constants, register

# How much of the variance a layer is meant to pass on, over the multivectors and over the
# scalars, and where to sit its bias. `small` quietens the branch of a residual, and
# `almost_unit_scalar` starts a factor of the geometric product off near one.
INITIALIZATIONS = dict(default=(1.0, 1.0, 0.0), small=(0.1, 0.1, 0.0), almost_unit_scalar=(0.5, 1.0, 1.0))

# The equivariance constraint leaves the grades with unequal variance, which GATr measured and
# corrects for by widening the bounds of some maps and narrowing others. The numbers are for a
# three dimensional projective algebra; another algebra has none and makes do with one.
VARIANCE = {(0, False): 1.0, (1, False): 4.0, (2, False): 6.0, (3, False): 2.0, (4, False): 0.5,
            (0, True): 0.5, (1, True): 1.5, (2, True): 1.5, (3, True): 0.5}


def equivariant_maps(X: MultiVector):
    """
    Every Pin equivariant linear map of X there is, each labelled by the grade it comes from and
    whether it is the shifted one. They are the projection onto each grade, and that projection
    shifted along the degenerate vector wherever the shift does not vanish. An algebra without
    such a vector, where every basis vector squares to something, has only the projections.
    """
    e0 = degenerate(X.algebra)
    for g in X.grades:
        yield (g, False), X.grade(g)
        if e0 is not None and (shifted := e0 * X.grade(g)).keys():
            yield (g, True), shifted


def equi_linear(X: MultiVector, weights: MultiVector[None]) -> MultiVector:
    """Weigh every equivariant map of X and add them up, which is the general equivariant map."""
    tot = 0
    for w, (_, Y) in zip(weights, equivariant_maps(X)):
        tot += w * Y
    return tot


class EquiLinear(LazyModuleMixin, nn.Module):
    """
    The most general equivariant linear map, over the multivectors and the scalars that ride along
    with them at once. The two meet only on the scalar blade, the one place a group cannot tell
    them apart, so the scalars feed that blade and read it back and leave the rest alone.
    """

    weight: UninitializedParameter

    def __init__(self, in_features, out_features, in_s_features=None, out_s_features=None,
                 bias=True, initialization="default"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.in_s_features = in_s_features
        self.initialization = initialization
        self.weight = UninitializedParameter()
        # A bias of its own is only needed where no scalars come in, since the map from those
        # carries one already and two would say the same thing twice.
        if bias and in_s_features is None:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        self.s2mvs = nn.Linear(in_s_features, out_features, bias=bias) if in_s_features else None
        self.mvs2s = nn.Linear(in_features, out_s_features, bias=bias) if out_s_features else None
        self.s2s = nn.Linear(in_s_features, out_s_features, bias=False) if in_s_features and out_s_features else None

    def initialize_parameters(self, input: MultiVector, scalars: MultiVector = None):
        if not self.has_uninitialized_params():
            return

        self.equi_linear = register(input.algebra, equi_linear)

        with torch.no_grad():
            # Which maps there are depends on the grades the input turns out to carry, so a sparse
            # input, a point say, is a layer with fewer weights rather than one full of zeros.
            self.maps = [label for label, _ in equivariant_maps(materialize_constants(input))]
            self.weight.materialize((len(self.maps), self.out_features, self.in_features))
            self.reset_parameters()

    def reset_parameters(self):
        mv_scale, s_scale, shift = INITIALIZATIONS[self.initialization]
        # Following He et al, the variance of a weight should be one over its fan in. Summing the
        # maps is near enough an identity that only the input features count towards that. Where
        # the scalars write to a blade as well, the two paths are each given half of the variance.
        halved = (0, False) if self.s2mvs is not None else None
        for i, label in enumerate(self.maps):
            variance = 0.5 if label == halved else VARIANCE.get(label, 1.0)
            bound = mv_scale * math.sqrt(variance / self.in_features)
            nn.init.uniform_(self.weight[i], -bound, bound)

        if self.s2mvs is not None:
            bound = mv_scale * math.sqrt(0.5 / self.in_s_features)
            nn.init.uniform_(self.s2mvs.weight, -bound, bound)
            if self.s2mvs.bias is not None:  # The bias answers to both paths, so to both fan ins.
                bound = 1 / math.sqrt(self.in_s_features + self.in_features)
                nn.init.uniform_(self.s2mvs.bias, shift - bound, shift + bound)

        into_s = [module for module in (self.s2s, self.mvs2s) if module is not None]
        for module in into_s:
            bound = s_scale / math.sqrt(module.in_features * len(into_s))
            nn.init.uniform_(module.weight, -bound, bound)
        if self.mvs2s is not None and self.mvs2s.bias is not None:
            bound = s_scale / math.sqrt(sum(module.in_features for module in into_s))
            nn.init.uniform_(self.mvs2s.bias, -bound, bound)

    def forward(self, input: MultiVector, scalars: MultiVector = None) -> tuple:
        input = materialize_constants(input)
        weights = input.algebra.scalar(e=self.weight)
        result = self.equi_linear(insert_out_features(input), weights)
        result = einops.reduce(result, "... o f -> ... o", "sum")  # Contract the input features.
        if self.bias is not None:
            result = result + input.algebra.scalar(e=self.bias)
        if self.s2mvs is not None:
            result = result + self.s2mvs(scalars)

        if self.mvs2s is None:
            return result, None
        # A multivector with no scalar blade holds no invariant to hand on, and so hands on none.
        outputs_s = self.mvs2s(input.grade(0)) if 0 in input.grades else 0
        if self.s2s is not None:
            outputs_s = outputs_s + self.s2s(scalars)
        return result, outputs_s
