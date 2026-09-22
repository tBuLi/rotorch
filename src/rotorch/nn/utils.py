import einops
import kingdon.einops_backend  # noqa: F401  Registers MultiVector with einops.
import torch
from kingdon import MultiVector

EPS = 1e-6


def cat(mvs: list[MultiVector]) -> MultiVector:
    """Concatenate multivectors along their feature axis, however many axes come before it."""
    batch = " ".join(f"d{i}" for i in range(mvs[0].ndim - 1))
    packed, _ = einops.pack([mv.asmvtype() for mv in mvs], f"{batch} *")
    return packed


def segment_mean(X: MultiVector, segment_ids: torch.Tensor, num_segments: int) -> MultiVector:
    """Average the multivectors that share a segment id, over the axis the ids index."""
    counts = segment_ids.new_zeros(num_segments).index_add_(0, segment_ids, torch.ones_like(segment_ids))
    counts = einops.rearrange(counts.clamp(min=1), "segment -> segment 1")
    return X.map(lambda v: v.new_zeros(num_segments, *v.shape[1:]).index_add_(0, segment_ids, v) / counts)


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


def grade_of_blades(mv: MultiVector) -> torch.Tensor:
    """
    For every blade of `mv`, the index of its grade among the grades present, so that a layer
    can hold one parameter per grade and still apply them all in one go.
    """
    index = {g: i for i, g in enumerate(mv.grades)}
    return torch.tensor([index[k.bit_count()] for k in mv.keys()])


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


def register(algebra, expr, **kwargs):
    """
    Compile `expr` for `algebra`, or hand back the operator registered under its name
    before, since registering anew would drop the codegen cached on it. The name of
    `expr` therefore has to be unique within `algebra.registry`. Any keyword arguments
    are passed on to :meth:`~kingdon.algebra.Algebra.add_operator`.
    """
    if expr.__name__ not in algebra.registry:
        algebra.add_operator(expr, symbolic=True, **kwargs)
    return algebra.registry[expr.__name__]


def scalar_normsq(X: MultiVector) -> MultiVector:
    """Scalar part of X times its reverse, unlike kingdon's normsq which keeps all grades."""
    return (~X * X).grade(0)


def mag2(X: MultiVector):
    """Squared magnitude of the single grade multivector X. Zero for null blades."""
    return sum(register(X.algebra, scalar_normsq)(X).values())


def norm(X: MultiVector):
    """Magnitude of the single grade multivector X, smoothed to stay differentiable at zero."""
    return (mag2(X) ** 2 + 1e-16) ** 0.25


def invariants(X: MultiVector) -> MultiVector:
    """
    One invariant per grade of X, laid out feature by feature: the scalar part as it is, and the
    squared magnitude of every other grade, neither of which the group can see.
    """
    X = materialize_constants(X)
    per_grade = torch.broadcast_tensors(*(X.e if g == 0 else mag2(X.grade(g)) for g in X.grades))
    return X.algebra.scalar(e=einops.rearrange(torch.stack(per_grade), "grade ... feature -> ... (feature grade)"))
