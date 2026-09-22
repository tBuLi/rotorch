import einops
import kingdon.einops_backend  # noqa: F401  Registers MultiVector with einops.
from torch import nn
from kingdon import MultiVector

from ...nn.gatr import EquiLinear, GATrBlock


class GATr(nn.Module):
    """
    A transformer over multivectors: a linear map in, a stack of blocks, a linear map out. The
    only thing it asks of its input beyond the items themselves is somewhere to stand while it
    takes a join, and it finds that in the average of the data.
    """

    def __init__(self, in_features=1, out_features=1, hidden_features=16, in_s_features=1,
                 out_s_features=1, hidden_s_features=128, num_blocks=10, heads=8):
        super().__init__()

        self.linear_in = EquiLinear(in_features, hidden_features, in_s_features, hidden_s_features)
        self.blocks = nn.ModuleList(GATrBlock(hidden_features, hidden_s_features, heads)
                                    for _ in range(num_blocks))
        self.linear_out = EquiLinear(hidden_features, out_features, hidden_s_features, out_s_features)

    def forward(self, input: MultiVector, scalars: MultiVector) -> tuple:
        reference = einops.reduce(input, "... item feature -> ... 1 1", "mean")
        h, h_s = self.linear_in(input, scalars)
        for block in self.blocks:
            h, h_s = block(h, h_s, reference)
        return self.linear_out(h, h_s)
