import pytest
import torch
from torch import nn

from rotorch.nn.cgenn import FullyConnectedGeometricProduct, GeometricProduct, MVLayerNorm, MVSiLU, NormalizationLayer
from rotorch.nn.cgenn import no_weight_decay, parameter_groups
from rotorch.nn.cgenn.utils import grade_of_blades, segment_mean, segment_plan
from rotorch.models.cgenn import NBodyCGGNN


def scatter_mean(X, segment_ids, num_segments):
    """The index_add_ segment_mean used to be; the new one must match it."""
    counts = segment_ids.new_zeros(num_segments).index_add_(0, segment_ids, torch.ones_like(segment_ids))
    counts = counts.clamp(min=1)[:, None]
    return X.map(lambda v: v.new_zeros(num_segments, *v.shape[1:]).index_add_(0, segment_ids, v) / counts)


@pytest.mark.parametrize("with_plan", [False, True])
def test_segment_mean_matches_scatter_bitwise(alg, with_plan):
    """Same bits, forward and backward, empty segments included."""
    X = alg.multivector(torch.randn(16, 9, 4, dtype=torch.float64, requires_grad=True))
    ids = torch.tensor([3, 0, 3, 1, 0, 3, 6, 6, 1])  # Segments 2, 4 and 5 stay empty.
    plan = segment_plan(ids, 7) if with_plan else None

    out = segment_mean(X, ids, 7, plan)
    ref = scatter_mean(X, ids, 7)
    assert out.keys() == ref.keys()
    assert all(torch.equal(a, b) for a, b in zip(out.values(), ref.values()))
    assert all((v[[2, 4, 5]] == 0).all() for v in out.values())

    grad_out = torch.randn(7, 4, dtype=torch.float64)
    g_out, = torch.autograd.grad(sum((v * grad_out).sum() for v in out.values()), X.values())
    g_ref, = torch.autograd.grad(sum((v * grad_out).sum() for v in ref.values()), X.values())
    assert torch.equal(g_out, g_ref)


def test_segment_plan_is_stable():
    ids = torch.tensor([2, 0, 2, 1, 0])
    perm, lengths = segment_plan(ids, 4)
    assert perm.tolist() == [1, 4, 3, 0, 2]  # Ties keep the order they came in.
    assert lengths.tolist() == [2, 1, 2, 0]


def test_grade_of_blades_follows_the_input_device(alg):
    a = alg.bivector(torch.randn(6, 5, 4, device="meta"))
    assert grade_of_blades(a).device.type == "meta"


@pytest.mark.parametrize("layer", [
    lambda: GeometricProduct(4),
    lambda: FullyConnectedGeometricProduct(4, 8),
    lambda: NormalizationLayer(),
    lambda: MVLayerNorm(),
    lambda: MVSiLU(),
])
def test_full_precision_under_autocast(alg, layer):
    """Under autocast these layers still give the float32 answer."""
    layer = layer()
    a = alg.bivector(torch.randn(6, 5, 4))
    expected = layer(a)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = layer(a)
        # A linear layer under the same autocast does go to bfloat16, which is the point of it.
        assert nn.functional.linear(torch.randn(5, 4), torch.randn(3, 4)).dtype == torch.bfloat16
    assert all(v.dtype == torch.float32 for v in out.values())
    assert all(torch.equal(u, v) for u, v in zip(out.values(), expected.values()))


def test_no_weight_decay_lists_gains_and_biases(alg3):
    model = NBodyCGGNN(hidden_features=6, n_layers=1)
    h = alg3.multivector(torch.randn(8, 5, 3))
    edges = (torch.tensor([0, 1, 2, 3, 4]), torch.tensor([1, 2, 3, 4, 0]))
    model(h, edges, alg3.scalar(e=torch.randn(5, 1)))  # Materialize the lazy layers.

    exempt = no_weight_decay(model)
    decayed = {name for name, _ in model.named_parameters()} - exempt
    # The 2-D gains, which the ndim rule alone would decay.
    assert "layers.0.edge_model.layers.0.1.a" in exempt and "layers.0.edge_model.layers.0.1.b" in exempt
    assert "layers.0.edge_model.layers.0.2.normalization.a" in exempt
    assert "layers.0.edge_model.layers.0.3.a" in exempt
    assert "layers.0.edge_model.layers.0.0.bias" in exempt
    assert all(model.get_parameter(name).ndim >= 2 for name in decayed)
    assert "embedding.weight" in decayed and "layers.0.edge_model.layers.0.2.weight" in decayed

    groups = parameter_groups(model, weight_decay=0.01)
    assert groups[0]["weight_decay"] == 0.01 and groups[1]["weight_decay"] == 0.0
    assert len(groups[0]["params"]) + len(groups[1]["params"]) == len(list(model.parameters()))
    torch.optim.AdamW(groups, lr=1e-3)  # Accepted as is.
