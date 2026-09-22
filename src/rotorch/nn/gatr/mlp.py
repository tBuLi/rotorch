from torch import nn
from kingdon import MultiVector

from .bilinear import GeometricBilinear
from .linear import EquiLinear
from .nonlinearity import ScalarGatedNonlinearity


class GeoMLP(nn.Module):
    """An ordinary two layer MLP, except that a geometric bilinear stands in for the first linear."""

    def __init__(self, features, hidden_features, s_features, hidden_s_features):
        super().__init__()

        self.bilinear = GeometricBilinear(features, hidden_features, in_s_features=s_features,
                                          out_s_features=hidden_s_features)
        self.nonlinearity = ScalarGatedNonlinearity()
        self.linear = EquiLinear(hidden_features, features, hidden_s_features, s_features)

    def forward(self, input: MultiVector, scalars: MultiVector, reference: MultiVector) -> tuple:
        return self.linear(*self.nonlinearity(*self.bilinear(input, scalars, reference)))
