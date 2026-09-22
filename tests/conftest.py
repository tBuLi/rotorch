from kingdon import Algebra, EvenMV, MultiVector
import torch
import pytest


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)

@pytest.fixture
def alg():
    return Algebra(3, 0, 1, backend='torch')

@pytest.fixture
def pga():
    """The same algebra under the basis that gives its points and its motors their usual signs."""
    return Algebra.fromname('3DPGA', backend='torch')

@pytest.fixture
def alg3():
    return Algebra(3, backend='torch')

@pytest.fixture
def alg5():
    return Algebra(5, backend='torch')

@pytest.fixture
def sta():
    return Algebra(1, 3, backend='torch')

@pytest.fixture
def double():
    """Run a test in double precision, parameters and all, since the lazy layers follow this."""
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(torch.float32)

@pytest.fixture
def versor():
    def versor(alg, reflections=4):
        """
        Composition of reflections. An even number of them is a rotor, which is all a layer built
        of products has to answer to; an odd one turns orientation over as well, which is what
        tells a layer that is only Spin equivariant apart from one that is Pin equivariant.
        """
        def reflection():
            # Only a vector that squares to a positive number has a norm to divide by, which in a
            # mixed signature leaves the timelike ones. The bar is one rather than zero because a
            # vector that barely clears the light cone normalizes to a boost of absurd rapidity.
            # Turning away a length and not a direction, so the rotor is uniform either way.
            while ((v := alg.vector(torch.randn(alg.d))) ** 2).e <= 1:
                pass
            return v.normalized()
        product = reflection()
        for _ in range(reflections - 1):
            product = product * reflection()
        return product
    return versor

@pytest.fixture
def assert_equivariant():
    def assert_equivariant(layer, versor, a, ulps=256):
        """
        Assert that f(w >> x) == w >> f(x), to within a few hundred ulps. A layer handing back
        several multivectors, the geometric one and the scalars riding along with it, is held to
        this for each of them; a scalar the group cannot turn has to come back unturned.
        """
        outs, turned = layer(a), layer(versor >> a)
        listed = lambda out: [out] if isinstance(out, MultiVector) else out
        for out, turn in zip(listed(outs), listed(turned)):
            eps = torch.finfo(out.values()[0].dtype).eps
            tol = ulps * eps * max(v.abs().max() for v in out.values())
            diff = turn - (versor >> out)
            assert all(v.abs().max() < tol for v in diff.values())
    return assert_equivariant