from kingdon import Algebra, EvenMV
import torch
import pytest
import rotorch.testing

@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)

@pytest.fixture
def alg():
    return Algebra(3, 0, 1, backend='torch')

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
def rotor():
    def rotor(alg):
    """The public harness of :mod:`rotorch.testing`, so that the tests exercise what users get."""
    return rotorch.testing.unit_rotor


@pytest.fixture
def assert_equivariant():
    return rotorch.testing.assert_equivariant
