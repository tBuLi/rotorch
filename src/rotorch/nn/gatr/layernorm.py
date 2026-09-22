import einops
import kingdon.einops_backend  # noqa: F401  Registers MultiVector with einops.
from torch import nn
from kingdon import MultiVector

from ..utils import mag2, materialize_constants


class EquiLayerNorm(nn.Module):
    """
    Divide by the magnitude of the input averaged over the channels, and normalize the scalars
    riding along with it the ordinary way. Nothing is learned here, unlike cgenn's MVLayerNorm,
    so one instance of it will do for a whole block.
    """

    def __init__(self, eps=0.01):
        super().__init__()

        # Half the blades of a projective algebra are null and contribute nothing to the
        # magnitude, so the floor under it sits far higher than that of an ordinary layer norm.
        self.eps = eps

    def forward(self, input: MultiVector, scalars: MultiVector) -> tuple:
        input = materialize_constants(input)
        norms = einops.reduce(mag2(input), "... f -> ... 1", "mean").clamp(min=self.eps)
        scale = input.algebra.scalar(e=norms ** -0.5)  # A scalar multiplies every blade.
        return (einops.einsum(input, scale, "..., ... -> ..."),
                nn.functional.layer_norm(scalars, scalars.shape[-1:]))
