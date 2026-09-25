from torch import nn
from kingdon import MultiVector

from ...nn.cgenn import MVLinear
from ...nn.flashclifford import Layer
from .. import cgenn


class CEMLP(nn.Module):
    """
    cgenn's Clifford equivariant MLP as flash-clifford builds it: in each layer, the nonlinearity, product and normalization that follow the linear map to the layer's width
    are one flash-clifford :class:`~rotorch.nn.flashclifford.Layer`.
    """

    def __init__(self, in_features, hidden_features, out_features, n_layers=2):
        super().__init__()

        features = [in_features] + [hidden_features] * (n_layers - 1) + [out_features]
        self.layers = nn.Sequential(*(nn.Sequential(MVLinear(i, o), Layer(o)) for i, o in zip(features, features[1:])))

    def forward(self, input: MultiVector) -> MultiVector:
        return self.layers(input)


class NBodyCGGNN(cgenn.NBodyCGGNN):
    """cgenn's n-body graph network, with flash-clifford's layers in its MLPs."""

    def __init__(self, **kwargs):
        super().__init__(mlp=CEMLP, **kwargs)
