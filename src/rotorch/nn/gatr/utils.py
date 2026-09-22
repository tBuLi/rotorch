from kingdon import MultiVector

from ..utils import degenerate, register


def weight(X: MultiVector) -> MultiVector:
    """
    How heavily a point counts, the coefficient it puts on the Euclidean pseudoscalar. Of the
    scalars one can read off a multivector this is the one a reflection flips the sign of.
    """
    return (degenerate(X.algebra) & X).grade(0)


def join(X: MultiVector, Y: MultiVector, Z: MultiVector) -> MultiVector:
    """
    The join of X and Y, weighed by the reference Z. The join alone turns with the rotors but
    flips under a reflection, and the reference, flipping with it, takes that back out.

    A dual is only fixed up to a convention, and kingdon's differs from GATr's by the grade
    involution. The linear map this feeds into weighs every grade on its own, so it absorbs that
    sign and the two spell out the same map by different signs on the same weights.
    """
    return (X & Y) * weight(Z)


def scalar_product(X: MultiVector, Y: MultiVector) -> MultiVector:
    """
    The inner product GATr attends by. Blades along the degenerate vector square to zero, so they
    drop out of their own accord rather than being masked out by hand.
    """
    return (~X * Y).grade(0)


def join_normsq(X: MultiVector, Y: MultiVector) -> MultiVector:
    """
    Squared magnitude of the join, which for two unit points is the square of the distance between
    them. GATr reaches the same number through a basis built to turn it into a dot product.
    """
    return scalar_product(X & Y, X & Y)


def inner(X: MultiVector, Y: MultiVector):
    """The inner product of X and Y as a number, the way :func:`~rotorch.nn.utils.mag2` is one."""
    return register(X.algebra, scalar_product)(X, Y).e
