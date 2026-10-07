import sympy
from torch import nn
from torch.nn.parameter import UninitializedParameter
from kingdon import MultiVector, add_operator
from kingdon.multivector import Scalar

from ...nn.cgenn import GeometricProduct, MVLayerNorm, MVLinear, MVSiLU, geometric_product, gradewise_linear, layernorm, mvsilu
from ...nn.utils import cat, materialize_constants, segment_mean


@add_operator(symbolic=True, codegen_symbolcls=sympy.Symbol)
def cemlp_layer(X, W: Scalar[None], b, a: Scalar[None], c: Scalar[None], Wr: Scalar[None], n: Scalar[None], Wl: Scalar[None], bl, w: Scalar, s):
    """A layer of :class:`CEMLP`: MVLinear, MVSiLU, GeometricProduct and MVLayerNorm."""
    return layernorm(geometric_product(mvsilu(gradewise_linear(X, W, b), a, c), Wr, n, Wl, bl, w), s)


class CEMLPLayer(nn.Sequential):
    """A layer of :class:`CEMLP`, its MVLinear, MVSiLU, GeometricProduct and MVLayerNorm computed as one operator, :func:`cemlp_layer`."""

    def forward(self, X: MultiVector) -> MultiVector:
        lin, silu, gp, norm = self
        # Each module takes what the one before it gives, so they size themselves by running once. An isinstance, unlike has_uninitialized_params, torch.compile traces through.
        if isinstance(lin.weight, UninitializedParameter):
            super().forward(X)
        params = lin.weight, lin.bias, silu.a, silu.b, gp.linear_right.weight, gp.normalization.a, gp.linear_left.weight, gp.linear_left.bias, gp.weight, norm.a
        return cemlp_layer(materialize_constants(X), *(X.algebra.scalar(e=p) for p in params))


class CEMLP(nn.Module):
    """Clifford equivariant MLP: a stack of layers, each a linear map, a nonlinearity, a product and a normalization."""

    def __init__(self, in_features, hidden_features, out_features, n_layers=2,
                 normalization_init=0):
        super().__init__()

        features = [in_features] + [hidden_features] * (n_layers - 1) + [out_features]
        self.layers = nn.Sequential(*(
            CEMLPLayer(
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
    """Equivariant graph convolution: message, mean aggregation and update."""

    def __init__(self, in_features, hidden_features, out_features, edge_attr_features=0,
                 node_attr_features=0, residual=True, normalization_init=0):
        super().__init__()

        self.residual = residual
        self.edge_model = CEMLP(in_features + edge_attr_features, hidden_features, out_features, normalization_init=normalization_init)
        self.node_model = CEMLP(in_features + out_features + node_attr_features, hidden_features, out_features, normalization_init=normalization_init)

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
                 n_layers=3, normalization_init=0, residual=True):
        super().__init__()

        self.embedding = MVLinear(in_features, hidden_features, gradewise=False)
        self.layers = nn.ModuleList(
            EGCL(hidden_features, hidden_features, hidden_features, edge_features_in,
                 residual=residual, normalization_init=normalization_init)
            for _ in range(n_layers)
        )
        self.projection = MVLinear(hidden_features, out_features)

    def forward(self, h: MultiVector, edges, edge_attr=None) -> MultiVector:
        h = self.embedding(h)
        for layer in self.layers:
            h = layer(h, edges, edge_attr=edge_attr)
        return self.projection(h).grade(1)
