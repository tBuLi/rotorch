"""
The bits every example shares: generating and caching its data, and running the same
training loop over either implementation so that their timings can be compared.

An example describes itself with a :class:`Task` and calls :func:`run`.
"""
import argparse
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from rotorch.nn.utils import mag2
from kingdon import MultiVector
from torch.utils.data import DataLoader, TensorDataset

# The implementations rotorch is measured against, and the checkout each is imported from. Their
# names double as the names of the options that point at those checkouts.
REFERENCES = dict(cgenn="clifford-group-equivariant-neural-networks",
                  gatr="geometric-algebra-transformer")


@dataclass
class Task:
    """
    :param name: names the example, and the directory its data is cached in.
    :param generate: (n_samples, seed) -> the arrays of one split, cached as they are returned.
    :param models: implementation name -> (args) -> (model, loss_fn), where loss_fn takes the
        tensors of one batch. The first is the default of :code:`--impl`.
    :param defaults: command line defaults this example overrides.
    :param model_defaults: defaults that hold for one implementation only, since what counts as
        a layer differs between them.
    """
    name: str
    generate: Callable[[int, int], tuple[np.ndarray, ...]]
    models: dict[str, Callable]
    defaults: dict = field(default_factory=dict)
    model_defaults: dict = field(default_factory=dict)


def mse_loss(prediction: MultiVector, target: MultiVector) -> torch.Tensor:
    """
    The mean squared distance between two multivectors. Null blades contribute nothing,
    having no distance to speak of, so in a degenerate algebra they go unpenalized.
    """
    difference = prediction - target
    return mag2(difference).mean() / len(difference.keys())


def codegen(args):
    """Keyword arguments for :class:`Algebra`: which backend, and whether to compile."""
    kwargs = dict(backend=args.backend)
    if args.compile == "operators":
        kwargs["wrapper"] = torch.compile
    return kwargs


def synchronize(device):
    """Both mps and cuda queue work asynchronously, so time nothing until it has landed."""
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device.startswith("mps"):
        torch.mps.synchronize()


@contextmanager
def measure_memory(device):
    """
    The peak allocation over the block, in MiB, measured as flash-clifford measures it: the
    counter is reset first, so that what earlier steps left behind is not counted, and the work
    is waited for, so that a queue that has not run yet cannot hide it. Zero on cpu, where torch
    keeps no such counter.
    """
    peak = [0.0]
    if not device.startswith("cuda"):
        yield peak
        return
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    synchronize(device)
    yield peak
    synchronize(device)
    peak[0] = torch.cuda.max_memory_allocated() / 2 ** 20


def memory(loss_fn, batch, device):
    """
    What one batch costs the card: a forward on its own, holding the graph the backward would
    need, and then a forward and backward together. Both after training, so that a compiled
    model is measured compiled.
    """
    with measure_memory(device) as forward:
        loss = loss_fn(*batch)
    del loss  # Or its activations are still live when the next measurement starts.

    with measure_memory(device) as forward_backward:
        loss_fn(*batch).backward()
    return forward[0], forward_backward[0]


def dataset(task, data_dir, split, n_samples, seed):
    path = os.path.join(data_dir, f"{task.name}_{split}_{n_samples}.npz")
    if not os.path.exists(path):
        os.makedirs(data_dir, exist_ok=True)
        np.savez(path, *task.generate(n_samples, seed))
    with np.load(path) as data:
        return TensorDataset(*(torch.from_numpy(data[name]) for name in data.files))


def evaluate(loss_fn, loader, device):
    with torch.no_grad():
        losses = [loss_fn(*(t.to(device) for t in batch)).item() for batch in loader]
    return sum(losses) / len(losses)


def train(args, task):
    if args.train_samples < args.batch_size:
        raise SystemExit(f"--train-samples {args.train_samples} is below --batch-size "
                         f"{args.batch_size}, which leaves no batch to train on.")
    torch.manual_seed(args.seed)
    data_dir = args.data_dir or os.path.join(os.environ.get("DATAROOT", "data"), task.name)
    # The training set lives on the device and every batch is drawn there.
    train_set = [t.to(args.device) for t in dataset(task, data_dir, "train", args.train_samples, seed=0).tensors]
    per_epoch = len(train_set[0]) // args.batch_size
    val_loader = DataLoader(dataset(task, data_dir, "val", args.val_samples, seed=1),
                            batch_size=args.batch_size)

    if args.impl in REFERENCES:
        sys.path.insert(0, getattr(args, f"{args.impl}_path"))
    try:
        model, loss_fn = task.models[args.impl](args)
    except ModuleNotFoundError as error:
        if args.impl not in REFERENCES:
            raise
        raise SystemExit(f"Could not import the {args.impl} model ({error}). Point "
                         f"--{args.impl}-path at your {REFERENCES[args.impl]} checkout, and "
                         f"install what that needs beyond rotorch's own dependencies.") from error
    if args.compile == "model":
        # kingdon generates its operators on the first call, which dynamo cannot trace, and the
        # lazy layers size their parameters there too. So run once before compiling.
        loss_fn(*(t[:args.batch_size] for t in train_set))
        loss_fn = torch.compile(loss_fn)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    times = []
    for step in range(args.steps):
        if step % per_epoch == 0:
            order = torch.randperm(len(train_set[0]), device=args.device)
        index = order[step % per_epoch * args.batch_size:][:args.batch_size]
        batch = [t[index] for t in train_set]

        synchronize(args.device)
        start = time.perf_counter()
        loss = loss_fn(*batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        synchronize(args.device)
        times.append(time.perf_counter() - start)

        if step % args.print_interval == 0:
            print(f"step {step:5d}  train loss {loss.item():9.4f}")

    n_parameters = sum(p.numel() for p in model.parameters())  # Lazy layers exist by now.
    warm = torch.tensor(times[args.warmup:] or times)  # Keep the stats sane for short runs.
    print(f"\n{task.name}, {args.impl}: {n_parameters} parameters, "
          f"val loss {evaluate(loss_fn, val_loader, args.device):.4f}")
    print(f"  first step {times[0] * 1e3:.0f} ms, then {warm.median() * 1e3:.1f} ms/step "
          f"(mean {warm.mean() * 1e3:.1f}, total {sum(times):.1f} s)")
    if args.device.startswith("cuda"):  # How much of the card a batch this size needs.
        forward, forward_backward = memory(loss_fn, batch, args.device)
        print(f"  memory {forward:.1f} MiB forward, {forward_backward:.1f} MiB "
              f"forward and backward")


def run(task):
    parser = argparse.ArgumentParser(description=f"Benchmark the {task.name} example.")
    parser.add_argument("--impl", default=next(iter(task.models)), choices=list(task.models))
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--val-samples", type=int, default=1024)
    parser.add_argument("--hidden-features", type=int, default=32)
    parser.add_argument("--hidden-s-features", type=int, default=128,
                        help="scalar channels an example carries alongside its multivectors")
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--print-interval", type=int, default=32)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", choices=["torch", "triton"], default="torch",
                        help="how kingdon emits its operators: one torch call per symbolic "
                             "multiply, or one triton kernel per operator")
    parser.add_argument("--compile", choices=["none", "operators", "model"], default="none",
                        help="compile every operator kingdon generates, which hands torch.compile "
                             "to kingdon as its wrapper, or the model as a whole")
    for impl, checkout in REFERENCES.items():
        parser.add_argument(f"--{impl}-path",
                            default=os.path.join(os.path.dirname(__file__), "..", "..", checkout))
    parser.set_defaults(**task.defaults)
    parser.set_defaults(**task.model_defaults.get(parser.parse_known_args()[0].impl, {}))
    args = parser.parse_args()
    if args.impl != "rotorch" and (args.compile == "operators" or args.backend != "torch"):
        raise SystemExit("--backend and --compile operators are about the operators kingdon "
                         "generates, so rotorch only.")
    train(args, task)
