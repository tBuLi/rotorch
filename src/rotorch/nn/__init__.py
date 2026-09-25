"""
Equivariant layers, grouped by the architecture they were introduced in.

Each architecture gets its own subpackage, so that layers sharing a name between papers stay
distinct and each stays recognisable to readers of the paper it comes from. So far those are
:mod:`rotorch.nn.cgenn`, :mod:`rotorch.nn.gatr` and :mod:`rotorch.nn.flashclifford`; others are
to follow. What they have in common is about multivectors rather than about any one paper, and
lives in :mod:`rotorch.nn.utils`.
"""
