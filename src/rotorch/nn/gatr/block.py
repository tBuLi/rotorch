from torch import nn
from kingdon import MultiVector

from .attention import SelfAttention
from .layernorm import EquiLayerNorm
from .mlp import GeoMLP


class GATrBlock(nn.Module):
    """
    A transformer block that normalizes before it acts rather than after: attention over the
    items and then an MLP over each of them, each with what went in added back to what came out.
    """

    def __init__(self, features, s_features, heads=8):
        super().__init__()

        self.norm = EquiLayerNorm()  # Learns nothing, so both halves of the block can share it.
        self.attention = SelfAttention(features, s_features, heads=heads, output_init="small")
        self.mlp = GeoMLP(features, 2 * features, s_features, 2 * s_features)

    def forward(self, input: MultiVector, scalars: MultiVector, reference: MultiVector) -> tuple:
        attended, attended_s = self.attention(*self.norm(input, scalars))
        input, scalars = input + attended, scalars + attended_s
        mixed, mixed_s = self.mlp(*self.norm(input, scalars), reference)
        return input + mixed, scalars + mixed_s
