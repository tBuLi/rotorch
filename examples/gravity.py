"""
Predict where planets end up, the n-body example of GATr.

    python examples/gravity.py
    python examples/gravity.py --impl gatr

GATr reads this dataset from files a script of its own writes out. This one is simulated here
instead, so the example runs on its own; the physics is the same, the systems are not.
"""
import einops
import numpy as np
import torch

import benchmark
from rotorch.nn.gatr.utils import join_normsq
from rotorch.nn.utils import mag2, register

N_PLANETS, DIM = 5, 3
DT, STEPS = 0.001, 100
SHIFT = 20.0  # How far from the origin a system is put, which only a translation equivariant model shrugs off.


def _accelerations(loc, masses):
    """Newton, in units where G is one and two bodies that meet pull on each other finitely."""
    delta = loc[:, None, :, :] - loc[:, :, None, :]
    distance = np.linalg.norm(delta, axis=-1, keepdims=True)
    pull = np.divide(1.0, distance ** 3, out=np.zeros_like(distance), where=distance > 1e-2)
    return (masses[:, None, :, :] * pull * delta).sum(axis=2)


def generate(n_samples, seed):
    """A star and five planets on noisy circular orbits, sampled before and after a hundred steps."""
    rng = np.random.default_rng(seed)
    masses = np.concatenate([rng.uniform(1.0, 10.0, (n_samples, 1, 1)),
                             rng.uniform(0.01, 0.1, (n_samples, N_PLANETS, 1))], axis=1)

    # The planets start in the xy plane, each at the speed that would hold it in a circle.
    angle = rng.uniform(0, 2 * np.pi, (n_samples, N_PLANETS, 1))
    radius = rng.uniform(0.1, 1.0, (n_samples, N_PLANETS, 1))
    around = np.concatenate([np.cos(angle), np.sin(angle), np.zeros_like(angle)], axis=-1)
    along = np.concatenate([-np.sin(angle), np.cos(angle), np.zeros_like(angle)], axis=-1)
    speed = np.sqrt(masses[:, :1] / radius)
    loc = np.concatenate([np.zeros((n_samples, 1, DIM)), radius * around], axis=1)
    vel = np.concatenate([np.zeros((n_samples, 1, DIM)), speed * along], axis=1)
    vel = vel + rng.normal(0, 0.01, vel.shape)

    # Turn the whole system somewhere else, so that only a model that does not care where it is
    # pointed or where it sits has an easy time of it.
    turn = np.linalg.qr(rng.standard_normal((n_samples, DIM, DIM))).Q
    turn = turn * np.sign(np.linalg.det(turn))[:, None, None]  # An odd number of axes, so this is a rotation.
    loc = loc @ turn + rng.normal(0, SHIFT, (n_samples, 1, DIM))
    vel = vel @ turn

    start_loc = loc.copy()
    for _ in range(STEPS):
        vel = vel + DT * _accelerations(loc, masses)
        loc = loc + DT * vel
    return tuple(a.astype(np.float32) for a in (masses, start_loc, vel, loc))


def embed(algebra, masses, loc, vel, target):
    """One channel per body, holding its mass, where it is and how fast it is going at once."""
    coordinates = lambda t: einops.rearrange(t, "batch body coord -> coord batch body 1")
    input = (algebra.scalar(e=masses) + algebra.point(coordinates(loc))
             + algebra.translation(coordinates(vel)))
    return input, algebra.scalar(e=torch.zeros_like(masses)), algebra.point(coordinates(target))


def rotorch(args):
    from kingdon import Algebra
    from rotorch.models.gatr import NBodyGATr

    algebra = Algebra.fromname("3DPGA", **benchmark.codegen(args))
    model = NBodyGATr(args.hidden_features, args.hidden_s_features, args.num_layers,
                      args.heads).to(args.device)

    def loss_fn(*batch):
        input, scalars, target = embed(algebra, *batch)
        prediction, regularization = model(input, scalars)
        # The join of two points is the line through them and its magnitude is how far apart they
        # are, once the prediction is divided by the weight it puts on itself. Nothing holds that
        # weight to one, which the regularization is there to do.
        distance = register(algebra, join_normsq)(prediction, target).e / mag2(prediction).clamp(min=1e-6)
        return distance.mean() + 0.01 * regularization.e.mean()

    return model, loss_fn


def gatr(args):
    """
    The reference implementation, which needs xformers: it is imported unconditionally by
    ``gatr.primitives.attention`` whether or not the attention ever dispatches to it.
    """
    from gatr import GATr, MLPConfig, SelfAttentionConfig
    from gatr.interface import embed_point, embed_scalar, embed_translation, extract_point

    model = GATr(in_mv_channels=1, out_mv_channels=1, hidden_mv_channels=args.hidden_features,
                 in_s_channels=1, out_s_channels=1, hidden_s_channels=args.hidden_s_features,
                 attention=SelfAttentionConfig(num_heads=args.heads), mlp=MLPConfig(),
                 num_blocks=args.num_layers).to(args.device)

    def loss_fn(masses, loc, vel, target):
        input = embed_scalar(masses) + embed_point(loc) + embed_translation(vel)
        output, scalars = model(input.unsqueeze(-2), scalars=torch.zeros_like(masses))
        prediction, weight = extract_point(output[..., 0, :]), output[..., 0, 14]
        regularization = (weight.abs() - 1) ** 2 + scalars[..., 0] ** 2
        return ((prediction - target) ** 2).sum(-1).mean() + 0.01 * regularization.mean()

    return model, loss_fn


task = benchmark.Task(name="gravity", generate=generate,
                      models=dict(rotorch=rotorch, gatr=gatr),
                      defaults=dict(hidden_features=16, hidden_s_features=128, num_layers=10,
                                    heads=8, batch_size=64, lr=3e-4,
                                    train_samples=3000, val_samples=512))

if __name__ == "__main__":
    benchmark.run(task)
