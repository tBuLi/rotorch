"""
Regress an O(5) invariant of two vectors in 5D, the o5 regression example of cgenn.

    python examples/o5.py
    python examples/o5.py --impl cgenn
"""
import einops
import numpy as np
import torch

import benchmark

DIM, N_VECTORS = 5, 2


def generate(n_samples, seed):
    """
    Two random vectors in 5D, labelled with the invariant cgenn regresses. Both are standardized,
    the vectors by one scale each so that the standardizing stays equivariant.
    """
    x = np.random.default_rng(seed).standard_normal((n_samples, N_VECTORS, DIM))
    r1, r2 = x[:, 0], x[:, 1]
    norm1, norm2 = np.linalg.norm(r1, axis=-1), np.linalg.norm(r2, axis=-1)
    y = np.sin(norm1) - 0.5 * norm2 ** 3 + (r1 * r2).sum(-1) / (norm1 * norm2)
    x = x / np.sqrt((x ** 2).mean((0, 2)))[None, :, None]
    y = (y - y.mean()) / y.std()
    return x.astype(np.float32), y[:, None].astype(np.float32)


def embed(algebra, points, values):
    """The two vectors, and the invariant they are labelled with, so a scalar."""
    return (algebra.vector(einops.rearrange(points, "batch vector coord -> coord batch vector")),
            algebra.scalar(e=values))


def rotorch(args):
    from kingdon import Algebra
    from rotorch.models.cgenn import O5CGMLP

    algebra = Algebra(DIM, **benchmark.codegen(args))
    model = O5CGMLP(N_VECTORS, args.hidden_features).to(args.device)

    def loss_fn(points, values):
        input, target = embed(algebra, points, values)
        return benchmark.mse_loss(model(input), target)

    return model, loss_fn


def cgenn(args):
    from models.o5_cgmlp import O5CGMLP

    # The targets are standardized already, so this model has nothing left to undo.
    model = O5CGMLP(ymean=0.0, ystd=1.0).to(args.device)
    return model, lambda points, values: model((points, values), 0)[0]


task = benchmark.Task(name="o5", generate=generate,
                      models=dict(rotorch=rotorch, cgenn=cgenn),
                      defaults=dict(hidden_features=8, train_samples=1024))

if __name__ == "__main__":
    benchmark.run(task)
