import math

import einops
import sympy
import torch
from torch import nn
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter
from kingdon import MultiVector, add_operator
from kingdon.multivector import Scalar

from ...nn.cgenn import FullyConnectedGeometricProduct, MVLayerNorm, MVLinear, fc_geometric_product, fc_geometric_product_unnormalized, layernorm
from ...nn.utils import cat, invariants, materialize_constants, segment_mean, sigmoid


def invariant_features(X: MultiVector) -> MultiVector:
    """The :func:`~rotorch.nn.utils.invariants` of X as scalar features, those of each of its features one after another."""
    return einops.rearrange(invariants(materialize_constants(X)), "k ... f -> ... (f k)")


class Bladewise(nn.Module):
    """
    Apply `module` to the coefficients of every blade in turn, for the modules that count their
    axes from the front and would take the blades for a batch, such as a batch norm.
    """

    def __init__(self, module: nn.Module):
        super().__init__()

        self.module = module

    def forward(self, input: MultiVector) -> MultiVector:
        return input.map(self.module)


def scalar_mlp(features, bias=True, activate=False):
    """cgenn's phi_h and theta_h: an MLP over scalars, normalized over the batch of nodes."""
    return nn.Sequential(
        nn.LazyLinear(features, bias=bias),
        Bladewise(nn.BatchNorm1d(features)),
        nn.ReLU(),
        nn.Linear(features, features),
        *([nn.ReLU()] if activate else []),
    )


@add_operator(symbolic=True, codegen_symbolcls=sympy.Symbol)
def grade_gate(X: MultiVector, H, W: Scalar[None], b: Scalar[None]) -> MultiVector:
    """:class:`GradeGate`, with W[k] and b[k] the map from the hidden features H to the gates of the k-th grade of X."""
    k = X.gradeidx_of_blades
    return einops.einsum(X, (einops.einsum(H, W[k], "... h, f h -> ... f") + b[k]).map(sigmoid), "... f, ... f -> ... f")


class GradeGate(LazyModuleMixin, nn.Module):
    """
    Known as psi and chi in cgenn: gate every grade of a multivector by a sigmoid of the scalar
    features that come with it. The gates are invariant, so what comes out transforms like what
    went in.
    """

    weight: UninitializedParameter
    bias: UninitializedParameter

    def __init__(self, hidden_features, features):
        super().__init__()
        self.features = features
        self.hidden_features = hidden_features
        self.hidden = nn.Sequential(nn.LazyLinear(hidden_features), nn.ReLU())
        self.weight = UninitializedParameter()
        self.bias = UninitializedParameter()

    def initialize_parameters(self, input: MultiVector, h: MultiVector):
        if not self.has_uninitialized_params():
            return

        with torch.no_grad():
            grades = len(materialize_constants(input).grades)  # A gate per grade of every feature.
            self.weight.materialize((grades, self.features, self.hidden_features))
            self.bias.materialize((grades, self.features))
            self.reset_parameters()

    def reset_parameters(self):
        # As nn.Linear's, from the hidden features to all the gates at once.
        nn.init.kaiming_uniform_(self.weight.view(-1, self.hidden_features), a=math.sqrt(5))
        bound = 1 / math.sqrt(self.hidden_features)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: MultiVector, h: MultiVector) -> MultiVector:
        input = materialize_constants(input)
        return grade_gate(input, self.hidden(h), *(input.algebra.scalar(e=p) for p in (self.weight, self.bias)))


@add_operator(symbolic=True, codegen_symbolcls=sympy.Symbol)
def fc_layer(X: MultiVector, Wr: Scalar[None], n: Scalar, Wl: Scalar[None], bl, w: Scalar, s) -> MultiVector:
    """A layer of :class:`CGLayer`, cgenn's "fc" one: FullyConnectedGeometricProduct and MVLayerNorm."""
    return layernorm(fc_geometric_product(X, Wr, n, Wl, bl, w), s)


@add_operator(symbolic=True, codegen_symbolcls=sympy.Symbol)
def fc_layer_unnormalized(X: MultiVector, Wr: Scalar[None], Wl: Scalar[None], bl, w: Scalar, s) -> MultiVector:
    """:func:`fc_layer` without the normalization of its product."""
    return layernorm(fc_geometric_product_unnormalized(X, Wr, Wl, bl, w), s)


class FCLayer(nn.Sequential):
    """A layer of :class:`CGLayer`, its FullyConnectedGeometricProduct and MVLayerNorm computed as one operator, :func:`fc_layer`."""

    def forward(self, X: MultiVector) -> MultiVector:
        fc, norm = self
        # The norm takes what the product gives, so they size themselves by running once. An isinstance, unlike has_uninitialized_params, torch.compile traces through.
        if isinstance(fc.weight, UninitializedParameter):
            super().forward(X)
        normalization = () if fc.normalization is None else (fc.normalization.a,)
        params = fc.linear_right.weight, *normalization, fc.linear_left.weight, fc.linear_left.bias, fc.weight, norm.a
        X = materialize_constants(X)
        return (fc_layer if normalization else fc_layer_unnormalized)(X, *(X.algebra.scalar(e=p) for p in params))


class CGLayer(nn.Module):
    """
    One round of message passing over the multivectors x and the scalars h at once, each feeding
    the other: the messages of h are built from invariants of x, and gate x in return.
    """

    def __init__(self, features_x, features_h, edge_attr_x=3, node_attr_x=1, residual=False, normalization_init=None):
        super().__init__()

        self.residual = residual
        product = lambda i, o: FCLayer(FullyConnectedGeometricProduct(i, o, normalization_init=normalization_init), MVLayerNorm())
        self.phi_x = product(3 * features_x + edge_attr_x, features_x)
        self.theta_x = product(2 * features_x + node_attr_x, features_x)
        self.phi_h = scalar_mlp(features_h, bias=False, activate=True)
        self.theta_h = scalar_mlp(features_h)
        self.psi_x = GradeGate(features_h, features_x)
        self.chi_x = GradeGate(features_h, features_x)

    def message(self, h_i, h_j, x_i, x_j, edge_attr_x):
        x_msg = self.phi_x(cat([x_i, x_j, x_i - x_j, edge_attr_x]))
        h_msg = self.phi_h(cat([invariant_features(x_msg), h_i, h_j, h_i - h_j]))
        return h_msg, self.psi_x(x_msg, h_msg)

    def update(self, h, x, h_agg, x_agg, node_attr_h, node_attr_x):
        x_out = self.theta_x(cat([x, x_agg, node_attr_x]))
        h_out = self.theta_h(cat([h, h_agg, invariant_features(x), node_attr_h]))
        return h_out, self.chi_x(x_out, h_out)

    def forward(self, h, x, edges, node_attr_h, node_attr_x, edge_attr_x):
        rows, cols = edges
        h_msg, x_msg = self.message(h[rows], h[cols], x[rows], x[cols], edge_attr_x)
        h_agg, x_agg = (segment_mean(msg, rows, len(h)) for msg in (h_msg, x_msg))
        h_out, x_out = self.update(h, x, h_agg, x_agg, node_attr_h, node_attr_x)
        return (h + h_out, x + x_out) if self.residual else (h_out, x_out)


class LorentzCGGNN(nn.Module):
    """Tag the jets of top quarks, reading the classes off invariants of the constituents."""

    def __init__(self, in_features_x=1, features_x=8, in_features_h=2, features_h=72,
                 edge_attr_x=3, node_attr_x=1, decoder_features=64, n_class=2, n_layers=4,
                 dropout=0.2, residual=False, normalization_init=None):
        super().__init__()

        self.embedding_x = MVLinear(in_features_x, features_x, gradewise=False)
        self.embedding_h = nn.Linear(in_features_h, features_h)
        layer = lambda: CGLayer(features_x, features_h, edge_attr_x, node_attr_x, residual, normalization_init)
        self.layers = nn.ModuleList(layer() for _ in range(n_layers))
        self.decoder = nn.Sequential(
            nn.LazyLinear(decoder_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_features, n_class),
        )

    def forward(self, h: MultiVector, x: MultiVector, edges, node_attr_h, node_attr_x, edge_attr_x, n_nodes) -> MultiVector:
        h, x = self.embedding_h(h), self.embedding_x(x)
        for layer in self.layers:
            h, x = layer(h, x, edges, node_attr_h, node_attr_x, edge_attr_x)

        jets = einops.reduce(cat([h, invariant_features(x)]), "(jet node) feature -> jet feature", "mean", node=n_nodes)
        return self.decoder(jets)
