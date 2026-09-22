from kingdon import Bireflection, EvenMV
import torch
import torch.nn as nn
from rotorch.nn.cgenn import (GeometricProduct, FullyConnectedGeometricProduct, MVLinear,
                           MVLayerNorm, MVSiLU, NormalizationLayer)


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
