import einops
import kingdon.einops_backend  # noqa: F401  Registers MultiVector with einops.
from torch import nn
from kingdon import MultiVector

from ..utils import materialize_constants


class ScalarGatedNonlinearity(nn.Module):
    """
    Gate every blade by a gelu of the scalar one, which the group cannot see and so cannot turn,
    leaving what comes out transforming like what went in. The scalars are gated by nothing and
    simply pass through the gelu themselves.
    """

    def forward(self, input: MultiVector, scalars: MultiVector) -> tuple:
        input = materialize_constants(input)
        gates = input.algebra.scalar(e=nn.functional.gelu(input.e, approximate="tanh"))
        return einops.einsum(input, gates, "..., ... -> ..."), nn.functional.gelu(scalars)
