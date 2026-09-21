"""
Regress the determinant of three vectors in 3D, the o3 example of cgenn.

    python examples/o3.py
    python examples/o3.py --impl cgenn
    python examples/o3.py --impl e3nn

The determinant is a pseudoscalar, so every model here has to be equivariant under rotations
and anti-equivariant under reflections.
"""
import einops
import numpy as np
import torch

import benchmark

DIM = 3


def generate(n_samples, seed):
    """Three random vectors in 3D, labelled with a tenth of the volume they span."""
    points = np.random.default_rng(seed).standard_normal((n_samples, DIM, DIM))
    return points.astype(np.float32), (np.linalg.det(points) / 10).astype(np.float32)


def embed(algebra, points, volumes):
    """The three vectors, and the determinant they span, which is a pseudoscalar."""
    return (algebra.vector(einops.rearrange(points, "batch vector coord -> coord batch vector")),
            algebra.pseudoscalar(e123=einops.rearrange(volumes, "batch -> batch 1")))


def rotorch(args):
    from kingdon import Algebra
    from rotorch.models.cgenn import O3CGMLP

    algebra = Algebra(DIM, **benchmark.codegen(args))
    model = O3CGMLP(DIM, args.hidden_features, num_layers=args.num_layers).to(args.device)

    def loss_fn(points, volumes):
        input, target = embed(algebra, points, volumes)
        return benchmark.mse_loss(model(input), target)

    return model, loss_fn


def e3nn(args):
    """
    The same task in irreps rather than multivectors, laid out like the model above: two
    bilinear layers with a norm gated nonlinearity between them, ending on the odd scalar a
    determinant is. The second product is with the input again, since a product of two even
    parity irreps could never be odd.
    """
    import e3nn.nn
    from e3nn import o3
    from torch import nn

    vectors = o3.Irreps(f"{DIM}x1o")
    hidden = o3.Irreps(f"{args.hidden_features}x0e + {args.hidden_features}x1e")
    first = o3.FullyConnectedTensorProduct(vectors, vectors, hidden)
    gates = nn.ModuleList(e3nn.nn.NormActivation(hidden, torch.sigmoid)
                          for _ in range(args.num_layers - 1))
    last = o3.FullyConnectedTensorProduct(hidden, vectors, o3.Irreps("1x0o"))
    model = nn.ModuleList([first, gates, last]).to(args.device)

    def loss_fn(points, volumes):
        input = einops.rearrange(points, "batch vector coord -> batch (vector coord)")
        h = first(input, input)
        for gate in gates:
            h = gate(h)
        return torch.nn.functional.mse_loss(last(h, input).squeeze(-1), volumes)

    return model, loss_fn


def cgenn(args):
    from models.o3_cgmlp import O3CGMLP

    model = O3CGMLP(DIM, args.hidden_features, num_layers=args.num_layers).to(args.device)
    return model, lambda points, volumes: model((points, volumes), 0)[0]


task = benchmark.Task(name="o3", generate=generate,
                      models=dict(rotorch=rotorch, cgenn=cgenn, e3nn=e3nn),
                      defaults=dict(hidden_features=32, num_layers=6))

if __name__ == "__main__":
    benchmark.run(task)
