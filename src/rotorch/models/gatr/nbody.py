from torch import nn
from kingdon import MultiVector

from .gatr import GATr
from ...nn.gatr.utils import weight


class NBodyGATr(nn.Module):
    """
    Predict where planets orbiting a star end up, reading the positions off points.

    Nothing holds the multivector that comes out to being a point, so the prediction is the grade
    the points live on and the regularization is how far the rest of it has wandered: how far the
    weight sits from one, and how much the scalars riding along have grown.
    """

    def __init__(self, hidden_features=16, hidden_s_features=128, num_blocks=10, heads=8):
        super().__init__()

        self.net = GATr(hidden_features=hidden_features, hidden_s_features=hidden_s_features,
                        num_blocks=num_blocks, heads=heads)

    def forward(self, input: MultiVector, scalars: MultiVector) -> tuple:
        prediction, scalars = self.net(input, scalars)
        points = prediction.grade(input.algebra.d - 1)
        regularization = (weight(points).e.abs() - 1) ** 2 + scalars.e ** 2
        return points, input.algebra.scalar(e=regularization)
