"""
A random group element and a check that a layer commutes with it. rotorch's tests use these; so
can tests of layers built on rotorch::

    from rotorch.testing import assert_equivariant, unit_rotor

    alg = Algebra(3, 0, 1, backend="torch")
    a = alg.bivector(torch.randn(6, 5, 4))
    assert_equivariant(MyLayer(4, 8), unit_rotor(alg), a)
"""
import torch
from kingdon import MultiVector


def unit_rotor(alg) -> MultiVector:
    """A random unit rotor of `alg`, as four random reflections in a row."""
    def reflection():
        # Only a vector that squares to a positive number has a norm to divide by, which in a
        # mixed signature leaves the timelike ones. The bar is one rather than zero because a
        # vector that barely clears the light cone normalizes to a boost of absurd rapidity.
        # Turning away a length and not a direction, so the rotor is uniform either way.
        while ((v := alg.vector(torch.randn(alg.d))) ** 2).e <= 1:
            pass
        return v.normalized()
    v1, v2, v3, v4 = (reflection() for _ in range(4))
    return v1 * v2 * v3 * v4


def assert_equivariant(layer, rotor: MultiVector, a: MultiVector, ulps: int = 256):
    """
    Assert ``layer(rotor >> a) == rotor >> layer(a)`` to `ulps` units in the last place of the
    largest output coefficient. A boost through a deep model needs far more than the default.
    """
    out = layer(a)
    eps = torch.finfo(out.values()[0].dtype).eps
    tol = ulps * eps * max(v.abs().max() for v in out.values())
    diff = layer(rotor >> a) - (rotor >> out)
    assert all(v.abs().max() < tol for v in diff.values())
