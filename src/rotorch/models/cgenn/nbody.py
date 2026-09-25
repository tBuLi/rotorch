from functools import partial

from torch import nn
from kingdon import MultiVector

from ...nn.cgenn import GeometricProduct, MVLayerNorm, MVLinear, MVSiLU
from ...nn.utils import cat, segment_mean


class CEMLP(nn.Module):
    """Clifford equivariant MLP: a stack of linear, nonlinear and product layers."""

    def __init__(self, in_features, hidden_features, out_features, n_layers=2,
                 normalization_init=0):
        super().__init__()

        features = [in_features] + [hidden_features] * (n_layers - 1) + [out_features]
        self.layers = nn.Sequential(*(
            nn.Sequential(
                MVLinear(i, o),
                MVSiLU(),
                GeometricProduct(o, normalization_init=normalization_init),
                MVLayerNorm(),
            )
            for i, o in zip(features, features[1:])
        ))

    def forward(self, input: MultiVector) -> MultiVector:
        return self.layers(input)


class EGCL(nn.Module):
    """
    Equivariant graph convolution: message, mean aggregation and update.

    :param mlp: (in, hidden, out features) -> the edge or the node model; cgenn's :class:`CEMLP` by default.
    """

    def __init__(self, in_features, hidden_features, out_features, edge_attr_features=0,
                 node_attr_features=0, residual=True, normalization_init=0, mlp=None):
        super().__init__()

        mlp = mlp or partial(CEMLP, normalization_init=normalization_init)
        self.residual = residual
        self.edge_model = mlp(in_features + edge_attr_features, hidden_features, out_features)
        self.node_model = mlp(in_features + out_features + node_attr_features, hidden_features, out_features)

    def message(self, h_i, h_j, edge_attr=None):
        input = h_i - h_j if edge_attr is None else cat([h_i - h_j, edge_attr])
        return self.edge_model(input)

    def aggregate(self, h_msg, segment_ids, num_segments):
        return segment_mean(h_msg, segment_ids, num_segments)

    def update(self, h_agg, h, node_attr=None):
        input = [h, h_agg] if node_attr is None else [h, h_agg, node_attr]
        out_h = self.node_model(cat(input))
        return h + out_h if self.residual else out_h

    def forward(self, h, edge_index, edge_attr=None, node_attr=None):
        rows, cols = edge_index
        h_msg = self.message(h[rows], h[cols], edge_attr)
        h_agg = self.aggregate(h_msg, rows, num_segments=h.shape[0])
        return self.update(h_agg, h, node_attr)


class NBodyCGGNN(nn.Module):
    """Predict the displacement of charged particles from their positions and velocities."""

    def __init__(self, in_features=3, hidden_features=28, out_features=1, edge_features_in=1,
                 n_layers=3, normalization_init=0, residual=True, mlp=None):
        super().__init__()

        self.embedding = MVLinear(in_features, hidden_features, gradewise=False)
        self.layers = nn.ModuleList(
            EGCL(hidden_features, hidden_features, hidden_features, edge_features_in,
                 residual=residual, normalization_init=normalization_init, mlp=mlp)
            for _ in range(n_layers)
        )
        self.projection = MVLinear(hidden_features, out_features)

    def forward(self, h: MultiVector, edges, edge_attr=None) -> MultiVector:
        h = self.embedding(h)
        for layer in self.layers:
            h = layer(h, edges, edge_attr=edge_attr)
        return self.projection(h).grade(1)
