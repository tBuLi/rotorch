import math

import einops
import kingdon.einops_backend  # noqa: F401  Registers MultiVector with einops.
import torch
from torch import nn
from kingdon import MultiVector

from .linear import EquiLinear
from .utils import inner, join_normsq
from ..utils import degenerate, mag2, register


def packed_features(X: MultiVector, points: int) -> int:
    """
    How many numbers GATr packs a query into, which is what its scores are scaled by: one for
    every blade the inner product keeps, save those holding the points, which are counted instead
    by the handful of numbers it takes to write down the distance between two of them.
    """
    null = degenerate(X.algebra).keys()[0]
    kept = [k for k in X.keys() if not k & null and k.bit_count() != points]
    return len(kept) + X.algebra.d + 1


class GeometricAttention(nn.Module):
    """
    Attention by how alike two multivectors are and by how close together they lie. The first is
    the inner product; the second is the magnitude of the join of the points they carry, which is
    the distance between those points, and the only part of the score that is learned is how much
    to weigh one against the other.
    """

    def __init__(self, heads, features, eps=1e-3):
        super().__init__()

        # A point divided by its own weight sits where it says it sits. Dividing by w / (w^2 + eps)
        # rather than by w keeps a point of no weight at all from running off to infinity.
        self.eps = eps
        self.log_weights = nn.Parameter(torch.zeros(heads, 1, 1, features))

    def forward(self, q: MultiVector, k: MultiVector, v: MultiVector,
                q_s: MultiVector, k_s: MultiVector, v_s: MultiVector) -> tuple:
        points = q.algebra.d - 1  # The grade a point lives on, a trivector in three dimensions.
        others = tuple(g for g in q.grades if g != points)
        query = einops.rearrange(q, "... head item feature -> ... head item 1 feature")
        key = einops.rearrange(k, "... item feature -> ... 1 1 item feature")

        alike = inner(query.grade(*others), key.grade(*others))
        close = register(q.algebra, join_normsq)(query.grade(points), key.grade(points)).e
        close = close * self._dehomogenize(query.grade(points)) * self._dehomogenize(key.grade(points))
        scores = einops.reduce(alike - self.log_weights.exp() * close, "... q k f -> ... q k", "sum")
        scores = scores + einops.einsum(q_s, k_s, "... head q f, ... k f -> ... head q k").e

        scale = q.shape[-1] * packed_features(q, points) + q_s.shape[-1]
        weights = q.algebra.scalar(e=torch.softmax(scores / math.sqrt(scale), dim=-1))
        return (einops.einsum(weights, v, "... head q k, ... k f -> ... head q f"),
                einops.einsum(weights, v_s, "... head q k, ... k f -> ... head q f"))

    def _dehomogenize(self, points: MultiVector):
        """The weight a point is divided by to put it where it says it is, squared and smoothed."""
        return mag2(points) / (mag2(points) + self.eps) ** 2


class SelfAttention(nn.Module):
    """
    Queries, keys and values off the same input, attention over the items, and a linear map back.
    Every head asks its own question, but they all share the keys and the values, as GATr's
    multi-query attention does.
    """

    def __init__(self, in_features, in_s_features, out_features=None, out_s_features=None,
                 heads=8, widen=2, output_init="default"):
        super().__init__()

        self.heads = heads
        hidden = max(widen * in_features // heads, 1)
        hidden_s = max(widen * in_s_features // heads, 4)
        self.q_linear = EquiLinear(in_features, hidden * heads, in_s_features, hidden_s * heads)
        self.k_linear = EquiLinear(in_features, hidden, in_s_features, hidden_s)
        self.v_linear = EquiLinear(in_features, hidden, in_s_features, hidden_s)
        self.attention = GeometricAttention(heads, hidden)
        self.out_linear = EquiLinear(hidden * heads, out_features or in_features,
                                     hidden_s * heads, out_s_features or in_s_features,
                                     initialization=output_init)

    def forward(self, input: MultiVector, scalars: MultiVector) -> tuple:
        q, q_s = self.q_linear(input, scalars)
        k, k_s = self.k_linear(input, scalars)
        v, v_s = self.v_linear(input, scalars)
        split = lambda x: einops.rearrange(x, "... item (head f) -> ... head item f", head=self.heads)
        merge = lambda x: einops.rearrange(x, "... head item f -> ... item (head f)")

        h, h_s = self.attention(split(q), k, v, split(q_s), k_s, v_s)
        return self.out_linear(merge(h), merge(h_s))
