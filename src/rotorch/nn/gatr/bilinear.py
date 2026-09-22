from torch import nn
from kingdon import MultiVector

from .linear import EquiLinear
from .utils import join
from ..utils import cat, register


class GeometricBilinear(nn.Module):
    """
    The geometric product of two projections of the input, alongside their join, which is the one
    equivariant product the geometric product cannot reach. The join needs a reference to fix the
    sign a reflection would otherwise turn over; the scalars skip both products and are picked up
    again by the linear map at the end.
    """

    def __init__(self, in_features, out_features, hidden_features=None, in_s_features=None,
                 out_s_features=None):
        super().__init__()

        hidden_features = hidden_features or out_features
        each = hidden_features // 2
        if 2 * each != hidden_features:
            raise ValueError("A GeometricBilinear splits its hidden features in two, so it needs "
                             f"an even number of them, not {hidden_features}.")
        product = lambda initialization="default": EquiLinear(
            in_features, each, in_s_features, initialization=initialization)
        self.linear_left = product()
        self.linear_right = product("almost_unit_scalar")  # Start the product off near an identity.
        self.linear_join_left = product()
        self.linear_join_right = product()
        self.linear_out = EquiLinear(hidden_features, out_features, in_s_features, out_s_features)

    def forward(self, input: MultiVector, scalars: MultiVector, reference: MultiVector) -> tuple:
        left, _ = self.linear_left(input, scalars)
        right, _ = self.linear_right(input, scalars)
        join_left, _ = self.linear_join_left(input, scalars)
        join_right, _ = self.linear_join_right(input, scalars)

        joined = register(input.algebra, join)(join_left, join_right, reference)
        return self.linear_out(cat([left * right, joined]), scalars)
