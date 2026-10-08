import einops
import kingdon.einops_backend  # noqa: F401  Registers MultiVector with einops.
import sympy
import torch
from kingdon import MultiVector, add_operator
from kingdon.codegen import Stack
from kingdon.multivector import Scalar

EPS = 1e-6


def cat(mvs: list[MultiVector]) -> MultiVector:
    """Concatenate multivectors along their feature axis, however many axes come before it."""
    batch = " ".join(f"d{i}" for i in range(mvs[0].ndim - 1))
    packed, _ = einops.pack([mv.asmvtype() for mv in mvs], f"{batch} *")
    return packed


def asscalar(X: MultiVector) -> Scalar:
    """The coefficients of X as a scalar whose first axis runs over its blades: no longer a multivector, it leaves its geometry to the tables an einsum takes it with, as those of :func:`~rotorch.nn.cgenn.gp.cayley`."""
    if not X.issymbolic:
        return X.algebra.scalar(e=X.values())
    res = X.algebra.scalar(e=Stack(*X.values()))
    res.shape = (len(X.keys()), *X.shape)
    return res


def segment_mean(X: MultiVector, segment_ids: torch.Tensor, num_segments: int) -> MultiVector:
    """Average the multivectors that share a segment id, over the axis the ids index."""
    counts = segment_ids.new_zeros(num_segments).index_add_(0, segment_ids, torch.ones_like(segment_ids))
    counts = einops.rearrange(counts.clamp(min=1), "segment -> segment 1")
    values = X.values()
    sums = values.new_zeros(len(values), num_segments, *values.shape[2:]).index_add_(1, segment_ids, values)
    return X.fromkeysvalues(X.algebra, X.keys(), sums / counts)


def materialize_constants(mv: MultiVector) -> MultiVector:
    """
    Turn the structural constants of a fixed layout, e.g. the scalar 1.0 of a
    Translation, into values, since only those take part in the arithmetic.
    """
    if any(v is not ... for v in mv.type_layout.values()):
        return mv.asmvtype()
    return mv


def insert_out_features(X: MultiVector) -> MultiVector:
    """Make room for the output features, so that the weights broadcast over them."""
    return einops.rearrange(materialize_constants(X), "... f -> ... 1 f")


def degenerate(algebra) -> MultiVector | None:
    """
    The basis vector that squares to zero, or nothing at all in an algebra where every basis
    vector squares to something. A projective algebra owes its extra equivariant maps to it.

    Built by hand rather than taken from :code:`algebra.blades`, which under a projective basis
    hands back a point carrying this vector as a constant of its layout rather than as a
    coefficient, and so with no key to its name.
    """
    if 0 not in algebra.signature:
        return None
    key = algebra.canon2bin[f"e{algebra.signature.index(0) + algebra.start_index}"]
    return MultiVector.fromkeysvalues(algebra, (key,), [1])


@add_operator(symbolic=True)
def scalar_normsq(X: MultiVector) -> MultiVector:
    """Scalar part of X times its reverse, unlike kingdon's normsq which keeps all grades."""
    return (~X * X).grade(0)


def mag2(X: MultiVector):
    """Squared magnitude of the single grade multivector X. Zero for null blades."""
    return sum(scalar_normsq(X).values())


def _root(squared):
    return (squared ** 2 + 1e-16) ** 0.25


class sigmoid(sympy.Function):
    """The logistic function, printed as one torch.sigmoid rather than a negation, an exp, a sum and a reciprocal, each a node for autograd."""

    def fdiff(self, argindex=1):
        return self * (1 - self)

    def _torchcode(self, printer):
        return f"torch.sigmoid({printer._print(self.args[0])})"


@add_operator(symbolic=True, codegen_symbolcls=sympy.Symbol)
def invariants(X: MultiVector) -> MultiVector:
    """
    One invariant per grade of X, stacked along a first axis over its grades: the scalar part as it is, and the
    squared magnitude of every other grade, neither of which the group can see.
    """
    return einops.rearrange([X.grade(0) if g == 0 else scalar_normsq(X.grade(g)) for g in X.grades], "k ... -> k ...")


def magnitudes(X: MultiVector) -> MultiVector:
    """The magnitude of every grade of X, smoothed to stay differentiable at zero, stacked along a first axis over its grades."""
    return einops.rearrange([scalar_normsq(X.grade(g)) for g in X.grades], "k ... -> k ...").map(_root)
