"""
Predict where charged particles end up, the nbody example of cgenn.

    python examples/nbody.py
    python examples/nbody.py --impl fk
    python examples/nbody.py --impl cgenn

cgenn reads this dataset from the files the EGNN repository ships. This one is simulated here
instead, so the example runs on its own; the physics is the same, the trajectories are not.
"""
import einops
import numpy as np
import torch

import benchmark
from rotorch.nn.utils import cat

N_BODIES, DIM = 5, 3
DT, SETTLE, SPAN = 0.001, 1000, 1000


def _accelerations(loc, charges):
    """Coulomb, with the force between two particles that meet capped rather than infinite."""
    delta = loc[:, :, None, :] - loc[:, None, :, :]
    distance = np.linalg.norm(delta, axis=-1, keepdims=True)
    pull = np.divide(1.0, distance ** 3, out=np.zeros_like(distance), where=distance > 1e-2)
    return ((charges[:, :, None] * charges[:, None, :]) * pull * delta).sum(axis=2).clip(-100, 100)


def generate(n_samples, seed):
    """Five charged particles, sampled once they have settled and again a thousand steps later."""
    rng = np.random.default_rng(seed)
    charges = rng.choice([-1.0, 1.0], (n_samples, N_BODIES, 1))
    loc = rng.standard_normal((n_samples, N_BODIES, DIM))
    vel = rng.standard_normal((n_samples, N_BODIES, DIM))
    vel /= np.linalg.norm(vel, axis=-1, keepdims=True)

    for step in range(SETTLE + SPAN):
        if step == SETTLE:
            start_loc, start_vel = loc.copy(), vel.copy()
        vel = vel + DT * _accelerations(loc, charges)
        loc = loc + DT * vel

    rows, cols = np.array([(i, j) for i in range(N_BODIES) for j in range(N_BODIES) if i != j]).T
    edges = np.broadcast_to(np.stack([rows, cols]), (n_samples, 2, len(rows)))
    edge_attr = charges[:, rows] * charges[:, cols]
    return (*(a.astype(np.float32) for a in (start_loc, start_vel, edge_attr, charges, loc)),
            edges.astype(np.int64))


def embed(algebra, loc, vel, edge_attr, charges, loc_end, edges):
    """
    Pack the batch into one graph of nodes, and give the model its arguments, the positions it
    predicts a step away from, and the positions it is aiming for.
    """
    samples, nodes, _ = loc.shape
    centred = loc - loc.mean(dim=1, keepdim=True)
    flatten = lambda t: t.reshape(-1, *t.shape[2:])
    centred, loc, vel, charges, loc_end = map(flatten, (centred, loc, vel, charges, loc_end))
    offset = nodes * torch.arange(samples, device=loc.device)[:, None, None]
    rows, cols = (edges + offset).transpose(0, 1).flatten(1)

    # One feature per quantity: the charge is a scalar, the position and velocity are vectors.
    as_vector = lambda t: algebra.vector(einops.rearrange(t, "node coord -> coord node 1"))
    input = cat([algebra.scalar(e=charges), as_vector(centred), as_vector(vel)])
    return ((input, (rows, cols), algebra.scalar(e=flatten(edge_attr))),
            as_vector(loc), as_vector(loc_end))


def rotorch(args, network=None):
    from kingdon import Algebra
    from rotorch.models.cgenn import NBodyCGGNN

    algebra = Algebra(DIM, **benchmark.codegen(args))
    model = (network or NBodyCGGNN)(hidden_features=args.hidden_features, n_layers=args.num_layers).to(args.device)

    def loss_fn(*batch):
        arguments, position, target = embed(algebra, *batch)
        return benchmark.mse_loss(position + model(*arguments), target)

    return model, loss_fn


def fk(args):
    """The same network with the layers of flash-clifford (Zhdanov, 2025) in its MLPs, which it offers as their faster replacement, built by rotorch out of kingdon operators."""
    from rotorch.models.flashclifford import NBodyCGGNN

    return rotorch(args, NBodyCGGNN)


def cgenn(args):
    from models.nbody_cggnn import NBodyCGGNN

    model = NBodyCGGNN(hidden_features=args.hidden_features, n_layers=args.num_layers).to(args.device)

    def loss_fn(loc, vel, edge_attr, charges, loc_end, edges):
        return model((loc, vel, edge_attr, charges, loc_end, edges), 0)[0]

    return model, loss_fn


task = benchmark.Task(name="nbody", generate=generate,
                      models=dict(rotorch=rotorch, fk=fk, cgenn=cgenn),
                      defaults=dict(hidden_features=28, num_layers=3, batch_size=100,
                                    train_samples=3000, val_samples=512))

if __name__ == "__main__":
    benchmark.run(task)
