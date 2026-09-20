import functools
from typing import NamedTuple
import einops
import kingdon.einops_backend  # noqa: F401  Registers MultiVector with einops.
import torch
from torch import nn
from kingdon import MultiVector

EPS = 1e-6


def cat(mvs: list[MultiVector]) -> MultiVector:
    """Concatenate multivectors along their feature axis."""
    packed, _ = einops.pack([mv.asmvtype() for mv in mvs], "n *")
    return packed


class SegmentPlan(NamedTuple):
    """The sort order of a set of segment ids and the length of each segment; see :func:`segment_plan`."""
    perm: torch.Tensor
    lengths: torch.Tensor


def segment_plan(segment_ids: torch.Tensor, num_segments: int) -> SegmentPlan:
    """Compute once per forward when several layers aggregate over the same edges."""
    perm = torch.argsort(segment_ids, stable=True)  # Stable, so ties keep their order.
    bounds = torch.arange(num_segments + 1, device=segment_ids.device)
    lengths = torch.diff(torch.searchsorted(segment_ids[perm], bounds))
    return SegmentPlan(perm, lengths)


def segment_mean(X: MultiVector, segment_ids: torch.Tensor, num_segments: int,
                 plan: SegmentPlan | None = None) -> MultiVector:
    """
    Average the multivectors that share a segment id, over the axis the ids index. Empty segments
    give zero.

    Sorted and reduced per segment rather than scattered with atomics, so the result is the same
    on every run. On the CPU it matches the old index_add_ bit for bit, gradients included.
    """
    perm, lengths = segment_plan(segment_ids, num_segments) if plan is None else plan
    # unsafe skips the check that the lengths add up, which on CUDA is a host sync per call.
    return X.map(lambda v: torch.segment_reduce(v[perm], "mean", lengths=lengths, axis=0,
                                                initial=0.0, unsafe=True))

def materialize_constants(mv: MultiVector) -> MultiVector:
    """
    Turn the structural constants of a fixed layout, e.g. the scalar 1.0 of a
    Translation, into values, since only those take part in the arithmetic.
    """
    if any(v is not ... for v in mv.type_layout.values()):
        return mv.asmvtype()
    return mv


def grade_of_blades(mv: MultiVector) -> torch.Tensor:
    """
    For every blade of `mv`, the index of its grade among the grades present, so that a layer
    can hold one parameter per grade and still apply them all in one go.
    """
    index = {g: i for i, g in enumerate(mv.grades)}
    return torch.tensor([index[k.bit_count()] for k in mv.keys()], device=device_of(mv))

def device_of(mv: MultiVector) -> torch.device:
    """Where the coefficients of `mv` live; plain numbers count as CPU."""
    values = mv.values()
    if isinstance(values, torch.Tensor):
        return values.device
    return next((v.device for v in values if isinstance(v, torch.Tensor)), torch.device("cpu"))


def full_precision(forward):
    """
    Run this forward outside autocast, inputs cast to the dtype of the weights. Products and norms
    square their inputs, which bf16 cannot afford; linear layers can stay in half precision.
    Does nothing when autocast is off.
    """
    @functools.wraps(forward)
    def wrapper(self, *args, **kwargs):
        mvs = [a for a in args if isinstance(a, MultiVector)]
        device_type = device_of(mvs[0]).type if mvs else "cpu"
        if not torch.is_autocast_enabled(device_type):
            return forward(self, *args, **kwargs)
        dtype = next(self.parameters()).dtype
        args = tuple(a.map(lambda v: v.to(dtype)) if isinstance(a, MultiVector) else a for a in args)
        with torch.autocast(device_type=device_type, enabled=False):
            return forward(self, *args, **kwargs)
    return wrapper


def no_weight_decay(model: nn.Module) -> set[str]:
    """
    Names of the parameters to spare from weight decay: everything one dimensional, plus whatever
    a module lists in a ``no_weight_decay`` method. The cgenn gains are (grades, features), so
    the ndim rule alone would decay them.
    """
    names = {name for name, p in model.named_parameters() if p.ndim <= 1}
    for prefix, module in model.named_modules():
        listed = module.no_weight_decay() if hasattr(module, "no_weight_decay") else ()
        names |= {f"{prefix}.{name}" if prefix else name for name in listed}
    return names


def parameter_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """
    Decayed and spared parameters as two optimizer groups. Run a batch through the model first,
    so the lazy layers have their parameters::

        optimizer = torch.optim.AdamW(parameter_groups(model, weight_decay=0.01), lr=1e-3)
    """
    exempt = no_weight_decay(model)
    params = list(model.named_parameters())
    return [
        {"params": [p for name, p in params if name not in exempt], "weight_decay": weight_decay},
        {"params": [p for name, p in params if name in exempt], "weight_decay": 0.0},
    ]

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
