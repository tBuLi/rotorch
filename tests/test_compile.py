"""
Whole-model compilation, behind ROTORCH_COMPILE_TESTS=1 since each compile takes a while on a
CPU. Always fullgraph=True, through a backend that runs the graph as is and counts compiles.
"""
import os
import sys

import pytest
import torch

from rotorch.models.cgenn import NBodyCGGNN, O5CGMLP

pytestmark = [
    pytest.mark.skipif(os.environ.get("ROTORCH_COMPILE_TESTS") != "1",
                       reason="compile gates run with ROTORCH_COMPILE_TESTS=1"),
    pytest.mark.skipif(sys.version_info < (3, 12),
                       reason="before 3.12 cached_property holds a lock dynamo cannot enter"),
]


def counting_backend():
    """Eager backend that counts how often dynamo compiles."""
    def backend(gm, example_inputs):
        backend.calls += 1
        return gm
    backend.calls = 0
    return backend


def assert_same(a, b):
    assert a.keys() == b.keys()
    for u, v in zip(a.values(), b.values()):
        assert torch.equal(u, v)


@pytest.fixture
def o5(alg5):
    torch._dynamo.reset()
    model = O5CGMLP(mlp_features=8)
    model(alg5.vector(torch.randn(5, 3, 2)))  # Sizes the lazy layers and generates the operators.
    return model


def test_o5_fullgraph_matches_eager_and_recompiles_once(alg5, o5):
    """First size static, second dynamic, then no more."""
    backend = counting_backend()
    compiled = torch.compile(o5, backend=backend, fullgraph=True)
    for n in (5, 7, 11, 13):
        a = alg5.vector(torch.randn(5, n, 2))
        assert_same(compiled(a), o5(a))
    assert backend.calls == 2


def test_o5_marked_dynamic_compiles_once(alg5, o5):
    backend = counting_backend()
    compiled = torch.compile(o5, backend=backend, fullgraph=True)
    for n in (5, 7, 11):
        a = alg5.vector(torch.randn(5, n, 2))
        torch._dynamo.mark_dynamic(a.values(), 1)
        assert_same(compiled(a), o5(a))
    assert backend.calls == 1


@pytest.mark.xfail(strict=True, raises=Exception,
                   reason="dynamic=True makes the blade keys symbolic and their bit_count "
                          "untraceable; the day this passes, drop the caveat in docs/benchmark.rst")
def test_o5_dynamic_true_is_not_an_option(alg5, o5):
    compiled = torch.compile(o5, dynamic=True, fullgraph=True)
    compiled(alg5.vector(torch.randn(5, 5, 2)))


def test_nbody_segment_mean_compiles(alg3):
    """segment_mean traces, with a different number of edges each call."""
    torch._dynamo.reset()
    model = NBodyCGGNN(hidden_features=6, n_layers=2, edge_features_in=0)
    h = alg3.multivector(torch.randn(8, 5, 3))
    model(h, (torch.tensor([0, 1, 2, 3, 4]), torch.tensor([1, 2, 3, 4, 0])))
    backend = counting_backend()
    compiled = torch.compile(model, backend=backend, fullgraph=True)
    for n_edges in (5, 8, 11):
        rows, cols = torch.randint(5, (2, n_edges))
        assert_same(compiled(h, (rows, cols)), model(h, (rows, cols)))
    assert backend.calls == 2
