import math

import einops
import pytest
from kingdon import Algebra, Bireflection, EvenMV
import torch
import torch.nn as nn
from rotorch.models.cgenn import CEMLPLayer, FCLayer
from rotorch.nn.cgenn import (GeometricProduct, FullyConnectedGeometricProduct, MVLinear,
                           MVLayerNorm, MVSiLU, NormalizationLayer)
from rotorch.nn.cgenn.gp import paths
from rotorch.nn.utils import insert_out_features


def test_linear(alg, versor, assert_equivariant):
    a = alg.bivector(torch.randn(6, 5, 4))
    l_nobias = MVLinear(4, 8, bias=False)
    b = l_nobias(a)
    assert b.shape == (5, 8)
    assert type(b) == type(a)
    assert_equivariant(l_nobias, versor(alg), a)

    # With bias the type will change because that adds a scalar.
    l_bias = MVLinear(4, 8, bias=True)
    b = l_bias(a)
    assert b.shape == (5, 8)
    assert type(b) == Bireflection
    assert_equivariant(l_bias, versor(alg), a)

def test_linear_gradewise(alg, versor, assert_equivariant):
    """One matrix for every grade is the gradewise map with that matrix in each."""
    a = alg.multivector(torch.randn(len(alg), 5, 4))
    single, gradewise = MVLinear(4, 8, gradewise=False), MVLinear(4, 8)
    single(a), gradewise(a)
    with torch.no_grad():
        gradewise.weight.copy_(single.weight.expand_as(gradewise.weight))
        single.bias.normal_()
        gradewise.bias.copy_(single.bias)
    torch.testing.assert_close(single(a).values(), gradewise(a).values())
    assert_equivariant(single, versor(alg), a)

def test_gp(alg, versor, assert_equivariant):
    a = alg.bivector(torch.randn(6, 5, 4))
    gp = GeometricProduct(4,)
    b = gp(a)
    assert b.shape == (5, 4)
    assert type(b) == EvenMV
    assert_equivariant(gp, versor(alg), a)

def test_fcgp(alg, versor, assert_equivariant):
    a = alg.bivector(torch.randn(6, 5, 4))
    fcgp = FullyConnectedGeometricProduct(4, 8)
    b = fcgp(a)
    assert b.shape == (5, 8)
    assert type(b) == EvenMV
    assert_equivariant(fcgp, versor(alg), a)

def test_mvsilu(alg, versor, assert_equivariant):
    a = alg.bivector(torch.randn(6, 5, 4))
    silu = MVSiLU()
    b = silu(a)
    assert b.shape == (5, 4)
    assert type(b) == type(a)
    assert_equivariant(silu, versor(alg), a)

def test_mvlayernorm(alg, versor, assert_equivariant):
    a = alg.bivector(torch.randn(6, 5, 4))
    layernorm = MVLayerNorm()
    b = layernorm(a)
    assert b.shape == (5, 4)
    assert type(b) == type(a)
    assert_equivariant(layernorm, versor(alg), a)

def test_normalization(alg, versor, assert_equivariant):
    a = alg.bivector(torch.randn(6, 5, 4))
    normalization = NormalizationLayer()
    b = normalization(a)
    assert b.shape == (5, 4)
    assert type(b) == type(a)
    assert_equivariant(normalization, versor(alg), a)


def weighted_paths(X, Y, w):
    """The weighted geometric product by its definition, a weight times every path, which the layers gather into one einsum."""
    return sum(w[k] * Z for k, Z in enumerate(paths(X, Y)))

def unfused_gp(gp, X):
    """:class:`GeometricProduct` as its modules and an eager weighted product."""
    return (gp.linear_left(X) + weighted_paths(X, gp.normalization(gp.linear_right(X)), X.algebra.scalar(e=gp.weight))) / math.sqrt(2)

def unfused_fcgp(fc, X):
    """:class:`FullyConnectedGeometricProduct` as its modules and an eager weighted product over the plane of in and out features."""
    Y = fc.linear_right(X) if fc.normalization is None else fc.normalization(fc.linear_right(X))
    product = weighted_paths(insert_out_features(X), insert_out_features(Y), X.algebra.scalar(e=fc.weight))
    return (fc.linear_left(X) + einops.reduce(product, "... o f -> ... o", "sum")) / math.sqrt(2)

@pytest.mark.parametrize('backend', ['torch', pytest.param('triton', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='triton kernels need a gpu'))])
@pytest.mark.parametrize('make, unfused, name', [
    (lambda: CEMLPLayer(MVLinear(4, 6), MVSiLU(), GeometricProduct(6), MVLayerNorm()), nn.Sequential.forward, 'cemlp_layer'),
    (lambda: FCLayer(FullyConnectedGeometricProduct(4, 6), MVLayerNorm()), nn.Sequential.forward, 'fc_layer'),
    (lambda: FCLayer(FullyConnectedGeometricProduct(4, 6, normalization_init=None), MVLayerNorm()), nn.Sequential.forward, 'fc_layer_unnormalized'),
    (lambda: GeometricProduct(4), unfused_gp, 'geometric_product'),
    (lambda: FullyConnectedGeometricProduct(4, 6), unfused_fcgp, 'fc_geometric_product'),
    (lambda: FullyConnectedGeometricProduct(4, 6, normalization_init=None), unfused_fcgp, 'fc_geometric_product_unnormalized'),
])
def test_layer_is_one_operator(alg3, double, backend, make, unfused, name):
    """A layer, its modules and all, is one operator, which computes what its modules do one by one."""
    device = 'cuda' if backend == 'triton' else 'cpu'
    layer = make()
    layer(alg3.multivector(torch.randn(8, 1, 4)))
    layer.to(device)
    with torch.no_grad():
        for p in layer.parameters():
            p.normal_()  # Away from their initial values, so that every parameter shows.
    alg = alg3 if backend == 'torch' else Algebra(3, backend=backend)

    for batch in [(5,), (2, 3), (37,)]:  # Generated for the first, the same code has to hold for the others, the last over several blocks of rows.
        x = torch.randn(8, *batch, 4, device=device, requires_grad=True)
        expected = unfused(layer, alg3.multivector(x))
        Y = layer(alg.multivector(x))
        assert Y.keys() == expected.keys()
        torch.testing.assert_close(Y.values(), expected.values())

        cotangent = torch.randn_like(Y.values())
        grads = torch.autograd.grad((Y.values() * cotangent).sum(), [x, *layer.parameters()])
        expected_grads = torch.autograd.grad((expected.values() * cotangent).sum(), [x, *layer.parameters()])
        for grad, expected_grad in zip(grads, expected_grads):
            torch.testing.assert_close(grad, expected_grad)
    operator = alg.registry[name]
    assert len(operator.operator_dict) == 1
    if backend == 'triton':
        dispatch = next(iter(operator.operator_dict.values())).func
        built = dict(zip(dispatch.__code__.co_freevars, (c.cell_contents for c in dispatch.__closure__)))['built']
        assert built and all(built.values()), 'fell back to torch'

@pytest.mark.skipif(not torch.cuda.is_available(), reason='triton kernels need a gpu')
def test_gp_loops(alg3, double, monkeypatch):
    """Past the gathered products a kernel spells out, it loops over them by tables, and computes the same, gradients and all, over several blocks of rows."""
    import kingdon.triton_codegen
    monkeypatch.setattr(kingdon.triton_codegen, '_UNROLLED', 0)
    gp = GeometricProduct(4)
    gp(alg3.multivector(torch.randn(8, 1, 4)))
    gp.to('cuda')
    alg = Algebra(3, backend='triton')
    x = torch.randn(8, 37, 4, device='cuda', requires_grad=True)
    Y, expected = gp(alg.multivector(x)), unfused_gp(gp, alg3.multivector(x))
    torch.testing.assert_close(Y.values(), expected.values())
    cotangent = torch.randn_like(Y.values())
    for grad, expected_grad in zip(*(torch.autograd.grad((v * cotangent).sum(), [x, *gp.parameters()]) for v in (Y.values(), expected.values()))):
        torch.testing.assert_close(grad, expected_grad)
    dispatch = next(iter(alg.registry['geometric_product'].operator_dict.values())).func
    assert all(dict(zip(dispatch.__code__.co_freevars, (c.cell_contents for c in dispatch.__closure__)))['built'].values()), 'fell back to torch'

@pytest.mark.parametrize('algebra, grades', [('alg', (2,)), ('alg', (0, 1)), ('sta', (0, 2)), ('alg5', (0, 1, 2, 3, 4, 5))])
def test_gp_weighs_every_path(request, double, algebra, grades):
    """The layer gathers its product into one einsum, which weighs every path as their sum does, also where the metric is degenerate and the keys sparse."""
    alg = request.getfixturevalue(algebra)
    keys = tuple(alg.indices_for_grades(grades))
    x = torch.randn(len(keys), 5, 4, requires_grad=True)
    gp = GeometricProduct(4)
    Y, expected = gp(alg.multivector(keys=keys, values=x)), unfused_gp(gp, alg.multivector(keys=keys, values=x))
    assert set(Y.keys()) == set(expected.keys())
    expected = torch.stack([dict(expected.items())[k] for k in Y.keys()])
    torch.testing.assert_close(Y.values(), expected)
    cotangent = torch.randn_like(expected)
    for grad, expected_grad in zip(*(torch.autograd.grad((v * cotangent).sum(), [x, *gp.parameters()]) for v in (Y.values(), expected))):
        torch.testing.assert_close(grad, expected_grad)
