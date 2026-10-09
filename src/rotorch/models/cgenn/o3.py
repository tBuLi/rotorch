import sympy
import torch
from torch import nn
from torch.nn.parameter import UninitializedParameter
from kingdon import MultiVector, add_operator
from kingdon.multivector import Scalar

from ...nn.cgenn import FullyConnectedGeometricProduct, MVSiLU, fc_geometric_product, mvsilu
from ...nn.utils import materialize_constants


@add_operator(symbolic=True, codegen_symbolcls=sympy.Symbol)
def mvsilus(X: MultiVector, a: Scalar[None], b: Scalar[None]) -> MultiVector:
    """:func:`mvsilu` again for every further len(X.grades) rows of a and b: nonlinearities stacked without products in between."""
    G = len(X.grades)
    for l in range(0, a.shape[0], G):
        X = mvsilu(X, a[l:l + G], b[l:l + G])
    return X


@add_operator(symbolic=True, codegen_symbolcls=sympy.Symbol)
def pseudoscalar_product(X: MultiVector, Wr: Scalar[None], n: Scalar, Wl: Scalar[None], bl, w: Scalar) -> MultiVector:
    """:func:`fc_geometric_product` of which only the pseudoscalar is read, and so computed."""
    return fc_geometric_product(X, Wr, n, Wl, bl, w).grade(X.algebra.d)


class O3CGMLP(nn.Module):
    """
    Regress an O(3) invariant from three vectors, read off the pseudoscalar.
    The nonlinearities are one operator, :func:`mvsilus`, and the last product another, :func:`pseudoscalar_product`: apart, since a layer that contracts nothing recomputes its
    forward in its backward, rather than store it.
    """

    def __init__(self, in_features=3, hidden_features=32, out_features=1, num_layers=6,
                 normalization_init=0):
        super().__init__()

        product = lambda i, o: FullyConnectedGeometricProduct(i, o, normalization_init=normalization_init)
        self.net = nn.Sequential(
            product(in_features, hidden_features),
            # As in cgenn, the nonlinearities are stacked without products in between.
            *(MVSiLU() for _ in range(num_layers - 1)),
            product(hidden_features, out_features),
        )

    def forward(self, input: MultiVector) -> MultiVector:
        first, *silus, last = self.net
        # Each module takes what the one before it gives, so they size themselves by running once. An isinstance, unlike has_uninitialized_params, torch.compile traces through.
        if isinstance(last.weight, UninitializedParameter):
            self.net(input)
        scalar = input.algebra.scalar
        X = mvsilus(materialize_constants(first(input)), scalar(e=torch.cat([m.a for m in silus])), scalar(e=torch.cat([m.b for m in silus])))
        params = last.linear_right.weight, last.normalization.a, last.linear_left.weight, last.linear_left.bias, last.weight
        return pseudoscalar_product(X, *(scalar(e=p) for p in params))
