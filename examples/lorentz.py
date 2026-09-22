"""
Tell jets from decaying top quarks apart from ordinary ones, the top tagging example of cgenn.

    python examples/lorentz.py
    python examples/lorentz.py --impl cgenn

cgenn reads this dataset from the h5 files the top tagging reference set ships. This one is
simulated here instead, so the example runs on its own. Both classes are sprays of massless
particles of the same invariant mass, so only the substructure tells them apart: a top decays
into three prongs where the background is one.
"""
import einops
import numpy as np
import torch

import benchmark
from rotorch.nn.utils import cat, mag2

N_PRONGS, N_CONSTITUENTS = 3, 12
BEAM_MASS, JET_MASS, JET_MOMENTUM, PRONG_SPREAD = 1.0, (150.0, 200.0), (400.0, 600.0), 0.1


def _isotropic(rng, shape):
    """Unit vectors spread evenly over the sphere."""
    cos = 2 * rng.random(shape) - 1
    phi = 2 * np.pi * rng.random(shape)
    sin = np.sqrt(1 - cos ** 2)
    return np.stack([sin * np.cos(phi), sin * np.sin(phi), cos], axis=-1)


def _boost(p, beta):
    """The four-momenta `p` of a system seen from a frame that it moves at `beta` relative to."""
    beta2 = (beta ** 2).sum(-1)[..., None, None]
    gamma, beta = 1 / np.sqrt(1 - beta2), beta[..., None, :]
    bp = (p[..., 1:] * beta).sum(-1, keepdims=True)
    energy = gamma * (p[..., :1] + bp)
    space = p[..., 1:] + ((gamma - 1) * bp / beta2 + gamma * p[..., :1]) * beta
    return np.concatenate([energy, space], -1)


def _rambo(rng, directions, mass):
    """
    Massless momenta along `directions`, scaled and boosted until they add up to a particle of
    `mass` at rest. Isotropic directions leave them uniform in phase space, as in RAMBO.
    """
    energy = -np.log(rng.random(directions.shape[:-1]) * rng.random(directions.shape[:-1]))
    q = np.concatenate([energy[..., None], energy[..., None] * directions], axis=-1)
    total = q.sum(-2)
    rest = _boost(q, -total[..., 1:] / total[..., :1])
    scale = mass / np.sqrt(total[..., 0] ** 2 - (total[..., 1:] ** 2).sum(-1))
    return scale[..., None, None] * rest


def generate(n_samples, seed):
    """
    Jets of massless constituents, half of them sprayed from three prongs and half from one, of
    the same invariant mass either way and each boosted along a direction of its own. The two
    beams, which are the same in every jet, come first.
    """
    rng = np.random.default_rng(seed)
    label = rng.integers(2, size=n_samples)
    mass = rng.uniform(*JET_MASS, n_samples)

    prongs = _isotropic(rng, (n_samples, N_PRONGS, 1))
    spray = prongs + PRONG_SPREAD * rng.standard_normal((n_samples, N_PRONGS, N_CONSTITUENTS // N_PRONGS, 3))
    spray = einops.rearrange(spray, "jet prong n coord -> jet (prong n) coord")
    spray /= np.linalg.norm(spray, axis=-1, keepdims=True)
    directions = np.where(label[:, None, None] == 1, spray, _isotropic(rng, (n_samples, N_CONSTITUENTS)))

    momentum = rng.uniform(*JET_MOMENTUM, n_samples)
    beta = _isotropic(rng, n_samples) * (momentum / np.sqrt(mass ** 2 + momentum ** 2))[:, None]
    constituents = _boost(_rambo(rng, directions, mass), beta)

    energy = np.sqrt(1 + BEAM_MASS ** 2)
    beams = np.array([[energy, 0, 0, 1], [energy, 0, 0, -1]])
    beams = np.broadcast_to(beams, (n_samples, 2, 4))
    return np.concatenate([beams, constituents], axis=1).astype(np.float32), label.astype(np.int64)


def psi(p):
    """Squash the masses into a range a plain MLP is comfortable with, sign and all."""
    return torch.sign(p) * torch.log(torch.abs(p) + 1)


def scalars(algebra, momenta):
    """
    The mass of every particle, on the beam channel or the constituent channel depending on which
    of the two it is, which is cgenn's one-hot encoding of that distinction.
    """
    mass = mag2(algebra.vector(einops.rearrange(momenta, "jet node coord -> coord jet node")))
    mass = mass.abs().sqrt()
    beam = torch.zeros_like(mass).index_fill_(-1, torch.arange(2, device=mass.device), 1)
    return einops.rearrange([mass * (1 - beam), mass * beam], "channel jet node -> jet node channel")


def edges(jets, nodes, device):
    """
    Every ordered pair of distinct nodes of a jet, over one graph that holds the whole batch.
    Counted out rather than masked off a diagonal, so that the shapes stay static.
    """
    rows = einops.repeat(torch.arange(nodes, device=device), "node -> node other", other=nodes - 1)
    cols = (rows + 1 + torch.arange(nodes - 1, device=device)) % nodes
    offset = nodes * torch.arange(jets, device=device)[:, None, None]
    return (rows + offset).flatten(), (cols + offset).flatten()


def embed(algebra, momenta, label):
    """
    Pack the batch into one graph per jet and give the model its arguments: the constituents as
    vectors, their masses as the scalars the invariant half of the network runs on, and every
    edge labelled with the momenta of both of its endpoints.
    """
    jets, nodes, _ = momenta.shape
    x = algebra.vector(einops.rearrange(momenta, "jet node coord -> coord (jet node) 1"))
    mass = einops.rearrange(scalars(algebra, momenta), "jet node channel -> (jet node) channel")
    h = algebra.scalar(e=psi(mass))
    rows, cols = edges(jets, nodes, momenta.device)
    edge_attr_x = cat([x[rows] - x[cols], x[rows], x[cols]])
    return (h, x, (rows, cols), h, x, edge_attr_x, nodes), label


def rotorch(args):
    from kingdon import Algebra
    from rotorch.models.cgenn import LorentzCGGNN

    algebra = Algebra(1, 3, **benchmark.codegen(args))
    model = LorentzCGGNN(features_x=args.hidden_features, n_layers=args.num_layers).to(args.device)

    def loss_fn(momenta, label):
        arguments, target = embed(algebra, momenta, label)
        return torch.nn.functional.cross_entropy(model(*arguments).e, target)

    return model, loss_fn


def cgenn(args):
    from kingdon import Algebra
    from models.lorentz_cggnn import LorentzCGGNN

    algebra = Algebra(1, 3, backend="torch")
    model = LorentzCGGNN(hidden_features_x=args.hidden_features, n_layers=args.num_layers).to(args.device)

    def loss_fn(momenta, label):
        jets, nodes, _ = momenta.shape
        ones = lambda *shape: torch.ones(*shape, device=momenta.device)
        data = dict(Pmu=momenta, nodes=scalars(algebra, momenta), is_signal=label, edges=torch.stack(edges(jets, nodes, momenta.device)))
        masks = dict(atom_mask=ones(jets, nodes, 1), edge_mask=ones(jets, nodes, nodes))  # Every node is real.
        return model(data | masks, 0)[0]

    return model, loss_fn


task = benchmark.Task(name="lorentz", generate=generate,
                      models=dict(rotorch=rotorch, cgenn=cgenn),
                      defaults=dict(hidden_features=8, num_layers=4, batch_size=16,
                                    train_samples=1024, val_samples=256))

if __name__ == "__main__":
    benchmark.run(task)
