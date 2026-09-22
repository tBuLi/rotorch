import einops
import pytest
import torch
from kingdon import Scalar
from rotorch.nn.gatr import (EquiLayerNorm, EquiLinear, GATrBlock, GeoMLP, GeometricBilinear,
                             ScalarGatedNonlinearity, SelfAttention)
from rotorch.nn.gatr.linear import equivariant_maps
from rotorch.nn.gatr.utils import join_normsq, weight
from rotorch.nn.utils import materialize_constants, register


@pytest.fixture
def data(alg):
    """Five items of four multivector channels each, with six scalars riding along with them."""
    return alg.multivector(torch.randn(16, 5, 4)), alg.scalar(e=torch.randn(5, 6))

@pytest.fixture(params=[4, 3], ids=["spin", "pin"])
def w(request, alg, versor):
    """A rotor, and then a versor that turns orientation over as well."""
    return versor(alg, reflections=request.param)

def referenced(layer, scalars):
    """Feed a layer that takes a join reference the average of the data, as GATr does."""
    return lambda x: layer(x, scalars, einops.reduce(x, "item feature -> 1 1", "mean"))


def test_equivariant_maps(alg, alg3):
    # The nine maps GATr tabulates: a projection onto each of the five grades, and four of those
    # shifted along the degenerate vector, the fifth shift running off the top of the algebra.
    full = alg.multivector(torch.randn(16, 5, 4))
    assert [label for label, _ in equivariant_maps(full)] == [
        (0, False), (0, True), (1, False), (1, True), (2, False), (2, True), (3, False), (3, True), (4, False)]
    # Sparse data is a layer with fewer weights rather than one full of zeros.
    assert [label for label, _ in equivariant_maps(alg.trivector(torch.randn(4, 5, 4)))] == [(3, False), (3, True)]
    # Without a degenerate vector to shift along only the grade projections are left.
    assert [label for label, _ in equivariant_maps(alg3.multivector(torch.randn(8, 5, 4)))] == [
        (0, False), (1, False), (2, False), (3, False)]

def test_linear(w, data, assert_equivariant):
    x, scalars = data
    linear = EquiLinear(4, 8, in_s_features=6, out_s_features=3)
    y, y_s = linear(x, scalars)
    assert y.shape == (5, 8) and y.keys() == x.keys()
    assert y_s.shape == (5, 3) and type(y_s) == Scalar
    assert linear.weight.shape == (9, 8, 4)  # One matrix per equivariant map there is.
    assert_equivariant(lambda a: linear(a, scalars), w, x)

def test_linear_without_scalars(alg, w, assert_equivariant):
    x = alg.bivector(torch.randn(6, 5, 4))
    linear = EquiLinear(4, 8)
    y, y_s = linear(x)
    assert y.shape == (5, 8) and y_s is None
    assert linear.weight.shape == (2, 8, 4)  # The bivectors, and the bivectors shifted.
    assert_equivariant(lambda a: linear(a)[0], w, x)

def test_layernorm(w, data, assert_equivariant):
    x, scalars = data
    layernorm = EquiLayerNorm()
    y, y_s = layernorm(x, scalars)
    assert y.shape == (5, 4) and type(y) == type(x)
    assert y_s.shape == (5, 6)
    assert_equivariant(lambda a: layernorm(a, scalars), w, x)

def test_nonlinearity(w, data, assert_equivariant):
    x, scalars = data
    nonlinearity = ScalarGatedNonlinearity()
    y, y_s = nonlinearity(x, scalars)
    assert y.shape == (5, 4) and type(y) == type(x)
    assert y_s.shape == (5, 6)
    assert_equivariant(lambda a: nonlinearity(a, scalars), w, x)

def test_bilinear(w, data, assert_equivariant):
    x, scalars = data
    bilinear = GeometricBilinear(4, 8, in_s_features=6, out_s_features=3)
    y, y_s = referenced(bilinear, scalars)(x)
    assert y.shape == (5, 8) and y_s.shape == (5, 3)
    assert_equivariant(referenced(bilinear, scalars), w, x)

def test_attention(w, data, assert_equivariant):
    x, scalars = data
    attention = SelfAttention(4, 6, heads=2)
    y, y_s = attention(x, scalars)
    assert y.shape == (5, 4) and y_s.shape == (5, 6)
    assert_equivariant(lambda a: attention(a, scalars), w, x)

def test_mlp(w, data, assert_equivariant):
    x, scalars = data
    mlp = GeoMLP(4, 8, 6, 12)
    y, y_s = referenced(mlp, scalars)(x)
    assert y.shape == (5, 4) and y_s.shape == (5, 6)
    assert_equivariant(referenced(mlp, scalars), w, x)

def test_block(w, data, assert_equivariant):
    x, scalars = data
    block = GATrBlock(4, 6, heads=2)
    y, y_s = referenced(block, scalars)(x)
    assert y.shape == (5, 4) and y_s.shape == (5, 6)
    assert_equivariant(referenced(block, scalars), w, x)

def test_join_is_the_distance(pga):
    """
    What the attention leans on: the join of two unit points is the line through them, and its
    magnitude is how far apart they are. GATr reaches the same number through a basis built by
    hand to turn it into a dot product.
    """
    p, q = (pga.point(torch.randn(3, 7, 1)) for _ in range(2))
    distance = register(pga, join_normsq)(p, q).e
    coordinates = lambda point: torch.stack([-v for v in point.grade(3).values()[:3]])
    assert torch.allclose(distance, ((coordinates(p) - coordinates(q)) ** 2).sum(dim=0))
    assert torch.allclose(weight(materialize_constants(p)).e, torch.ones(()))

def test_weight_flips_under_a_reflection(pga, versor):
    """The one scalar of a multivector that a reflection turns over, which is what fixes the join."""
    t = pga.trivector(torch.randn(4, 7, 1))
    assert torch.allclose(weight(versor(pga, 4) >> t).e, weight(t).e)
    assert torch.allclose(weight(versor(pga, 3) >> t).e, -weight(t).e)
