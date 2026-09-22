"""
Run one example over the whole benchmark matrix and write every result to one csv.

Each run is timed and, on cuda, weighed: the peak allocation of a forward on its own and of a
forward and backward together, so that the saving can be read as a ratio to cgenn the way
flash-clifford and flash-kingdon report theirs.

The matrix is every implementation x the batch sizes x cpu and cuda, each repeated so that the
faster run can be kept and so that a compiled run is seen once cold and once warm:

    python examples/sweep.py                        # everything this machine can run
    python examples/sweep.py --example examples/gravity.py   # GATr rather than cgenn
    python examples/sweep.py --devices cuda         # gpu only
    python examples/sweep.py --preset quick         # eager only, small batches, one rep
    python examples/sweep.py --dry-run              # print the plan and stop

Every example is measured against the implementation of its own paper, which for the cgenn
examples is cgenn and for gravity is GATr, so the reference columns follow the example.

Each run is a separate process, so one that dies takes its row down and nothing else. The csv
is appended to as results land and is re-read on startup, so the sweep can be interrupted and
restarted and will pick up where it left off. A configuration that fails at one batch size is
not tried at a larger one, since these failures are failures of size.

The whole thing is a few hours, most of it compiling: every batch size compiles anew.
:code:`--cuda-batches` and :code:`--cpu-batches` are the dials if that is too much, and
:code:`--configs` drops columns.

The cgenn columns need its checkout on :code:`--cgenn-path`, plus pyyaml, scipy and
scikit-learn; the GATr columns need its checkout on :code:`--gatr-path`, plus xformers, which
it imports whether or not the attention ever dispatches to it. None of these are rotorch's own
dependencies. The triton columns need a gpu; on the cpu the flag does nothing and the run would
only repeat the plain rotorch one, so they are left out of the plan.
"""
import argparse
import csv
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time

# Implementation -> the flags that select it. The order is the order of the table in the docs,
# and also the order to run them in: the cheap ones answer most of the question.
CONFIGS = {
    "cgenn": ["--impl", "cgenn"],
    "gatr": ["--impl", "gatr"],
    "rotorch": [],
    "rotorch-triton": ["--backend", "triton"],
    "cgenn-compiled": ["--impl", "cgenn", "--compile", "model"],
    "gatr-compiled": ["--impl", "gatr", "--compile", "model"],
    "rotorch-operators": ["--compile", "operators"],
    "rotorch-model": ["--compile", "model"],
    "rotorch-triton-model": ["--backend", "triton", "--compile", "model"],
}
# Run these first. The eager ones cost nothing to start; triton pays a compile per kernel, which
# is seconds against the minutes inductor wants for a whole model.
FIRST = ("cgenn", "gatr", "rotorch", "rotorch-triton")

# The reference every example is measured against, which is the one its own paper ships. An
# example knows nothing of the others, so asking it for one of theirs is an error, not a row.
REFERENCE = dict(hulls="cgenn", lorentz="cgenn", nbody="cgenn", o3="cgenn", o5="cgenn",
                 gravity="gatr")

FIELDS = ["timestamp", "host", "device", "config", "batch", "rep", "status", "median_ms",
          "mean_ms", "first_step_ms", "memory_forward_mib", "memory_step_mib", "total_s",
          "wall_s", "parameters", "val_loss", "returncode", "error", "steps", "warmup",
          "train_samples", "example", "python", "torch", "cuda", "gpu", "cpu", "platform"]

SUMMARY = re.compile(r"first step ([\d.]+) ms, then ([\d.]+) ms/step \(mean ([\d.]+), "
                     r"total ([\d.]+) s\)")
VAL_LOSS = re.compile(r"(\d+) parameters, val loss ([-\d.]+)")
MEMORY = re.compile(r"memory ([\d.]+) MiB forward, ([\d.]+) MiB forward and backward")
# Failures worth telling apart in the csv: the first two are limits of the card, the third of
# the toolchain, and an illegal access has been known to take the whole machine with it.
FAILURES = [("out of memory", "out-of-memory"),
            ("illegal memory access", "illegal-memory-access"),
            ("CUDA error", "cuda-error"),
            ("timed out after", "timeout"),
            # Inductor writes c++ for the cpu and needs a compiler for it. On windows that is
            # MSVC, which is only on the path inside a developer prompt. Triton carries the
            # cuda side, so this is a cpu-only failure.
            ("is not found", "no-cpp-compiler"),
            # Inductor builds its cpu wrapper with /openmp, so the MSVC include path has to
            # carry omp.h. A conda prompt has cl but not always the headers beside it.
            ("Cannot open include file: 'omp.h'", "no-openmp-headers"),
            ("BackendCompilerFailed", "compile-failed"),
            ("InductorError", "compile-failed"),
            ("InternalTorchDynamoError", "dynamo-failed"),
            ("Could not import the", "reference-path"),
            ("ModuleNotFoundError", "import-error"),
            # A triton kernel that asks for more registers or shared memory than the card has.
            ("OutOfResources", "triton-resources"),
            ("PTXASError", "ptxas-failed")]


def implementation(config):
    """Which implementation a configuration runs, rotorch's own columns being rotorch itself."""
    flags = CONFIGS[config]
    return flags[flags.index("--impl") + 1] if "--impl" in flags else "rotorch"


def reference(example):
    """The implementation `example` is measured against."""
    return REFERENCE[os.path.splitext(os.path.basename(example))[0]]


def environment(python):
    """Ask the interpreter that will run the benchmarks what it is about to run them on."""
    code = ("import json, platform, torch;"
            "print(json.dumps({'python': platform.python_version(),"
            "'torch': torch.__version__,"
            "'cuda': torch.version.cuda or '',"
            "'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else '',"
            "'cpu': platform.processor(), 'platform': platform.platform()}))")
    probe = subprocess.run([python, "-c", code], capture_output=True, text=True)
    if probe.returncode:
        raise SystemExit(f"Could not import torch with {python}:\n{probe.stderr}")
    return json.loads(probe.stdout)


def classify(output):
    """Name the failure, so that a row that has no timing still says something."""
    for signature, name in FAILURES:
        if signature in output:
            return name
    return "failed"


def run(example, config, device, batch, rep, args, environ):
    """One process, one row. Never raises: a failure is a result too."""
    command = [args.python, example, "--device", device, "--batch-size", str(batch),
               "--steps", str(args.steps), "--warmup", str(args.warmup),
               "--train-samples", str(max(args.train_samples, batch)),
               "--val-samples", str(args.val_samples), "--print-interval", str(args.steps),
               *CONFIGS[config]]
    if path := getattr(args, f"{implementation(config)}_path", None):
        command += [f"--{implementation(config)}-path", path]

    start = time.perf_counter()
    try:
        done = subprocess.run(command, cwd=os.path.dirname(example) or ".", text=True,
                              capture_output=True, timeout=args.timeout)
        output, returncode = done.stdout + done.stderr, done.returncode
    except subprocess.TimeoutExpired as expired:
        output = (expired.stdout or b"").decode(errors="replace")
        output, returncode = output + f"\ntimed out after {args.timeout} s\n", -1
    wall = time.perf_counter() - start

    row = dict(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"), host=socket.gethostname(),
               device=device, config=config, batch=batch, rep=rep, status="ok",
               returncode=returncode,
               error="", wall_s=round(wall, 1), steps=args.steps, warmup=args.warmup,
               train_samples=max(args.train_samples, batch),
               example=os.path.basename(example), **environ)
    if match := SUMMARY.search(output):
        row.update(first_step_ms=match[1], median_ms=match[2], mean_ms=match[3],
                   total_s=match[4])
    if match := VAL_LOSS.search(output):
        row.update(parameters=match[1], val_loss=match[2])
    if match := MEMORY.search(output):
        row.update(memory_forward_mib=match[1], memory_step_mib=match[2])
    if returncode or "median_ms" not in row:
        row.update(status="failed", error=classify(output))
        with open(f"{args.output}.{config}_{device}_b{batch}.log", "w",
                  encoding="utf-8") as log:  # Keep the whole thing, for the post mortem.
            log.write(" ".join(command) + "\n\n" + output)
    return row


def write(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(row)


def completed(path):
    """(config, device, batch, rep) of every run already in the csv, failures included."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8", newline="") as handle:
        return {(row["config"], row["device"], int(row["batch"]), int(row["rep"] or 1)):
                row["status"] for row in csv.DictReader(handle)}


def skip(config, device, example):
    """
    There is no cpu triton: the flag is ignored and the run is a second plain rotorch run. And
    the reference columns of the other papers are not this example's to run.
    """
    if device == "cpu" and "triton" in CONFIGS[config]:
        return True
    return implementation(config) not in ("rotorch", reference(example))


def plan(args, has_cuda):
    """Cheap and informative first, then the ones that compile for minutes, by batch size."""
    devices = args.devices or (["cpu", "cuda"] if has_cuda else ["cpu"])
    for device in devices:
        batches = args.cpu_batches if device == "cpu" else args.cuda_batches
        configs = [c for c in CONFIGS if c in args.configs and not skip(c, device, args.example)]
        for stage in (FIRST, tuple(c for c in configs if c not in FIRST)):
            for batch in batches:
                for config in [c for c in configs if c in stage]:
                    for rep in range(1, args.reps + 1):
                        yield config, device, batch, rep


def table(best, oom, device, metric, title, ratio, example):
    """
    One metric over the configurations and batch sizes of one device, each cell against the
    reference cell beside it. :param ratio: how to turn the two into the figure in brackets --
    speedup for time, so that more is better, and the ratio itself for memory, so that less is,
    which is how flash-clifford reports it.
    """
    against = reference(example)
    batches = sorted({key[2] for key in best if key[0] == device})
    configs = [c for c in CONFIGS if not skip(c, device, example)
               and any(key[:2] == (device, c) and metric in best[key] for key in best)]
    if not configs:
        return

    print(f"\n{device}: {title}")
    print("batch".rjust(7) + "".join(name.rjust(22) for name in configs))
    for batch in batches:
        cells = []
        for config in configs:
            value = best.get((device, config, batch), {}).get(metric)
            baseline = best.get((device, against, batch), {}).get(metric)
            if value is None:
                cells.append(("OOM" if (device, config, batch) in oom else "-").rjust(22))
            elif baseline and config != against:
                cells.append(f"{value:.1f} ({ratio(baseline, value):.2f}x)".rjust(22))
            else:
                cells.append(f"{value:.1f}".rjust(22))
        print(str(batch).rjust(7) + "".join(cells))


def summarize(path, example):
    """
    What the docs want: the faster of the reps, the speedup over the reference, and what each of
    them asked the card for, as flash-clifford reports it -- a forward on its own and a forward
    and backward together, each as a ratio to the reference, where below one is a saving.
    """
    best, fastest, oom = {}, {}, set()
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["device"], row["config"], int(row["batch"]))
            if row["error"] == "out-of-memory":
                oom.add(key)
            if row["status"] != "ok" or not row["median_ms"]:
                continue
            if float(row["median_ms"]) < fastest.get(key, float("inf")):
                fastest[key] = float(row["median_ms"])  # Memory comes from the run we quote.
                best[key] = {name: float(row[name]) for name in
                             ("median_ms", "memory_forward_mib", "memory_step_mib")
                             if row.get(name)}

    against = reference(example)
    for device in dict.fromkeys(key[0] for key in best):
        table(best, oom, device, "median_ms", f"ms/step, and the speedup over {against}",
              lambda baseline, value: baseline / value, example)
        table(best, oom, device, "memory_forward_mib",
              f"peak MiB of a forward, and the ratio to {against} (below one is a saving)",
              lambda baseline, value: value / baseline, example)
        table(best, oom, device, "memory_step_mib",
              f"peak MiB of a forward and backward, and the ratio to {against}",
              lambda baseline, value: value / baseline, example)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--example", default=os.path.join(os.path.dirname(__file__),
                                                          "hulls.py"))
    parser.add_argument("--output", default=f"sweep-{socket.gethostname()}.csv")
    parser.add_argument("--devices", nargs="*", choices=["cpu", "cuda"],
                        help="default: cpu, and cuda if torch can see one")
    parser.add_argument("--configs", nargs="*", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--cpu-batches", nargs="*", type=int,
                        default=[32, 128, 512, 2048])
    parser.add_argument("--cuda-batches", nargs="*", type=int,
                        default=[32, 128, 512, 2048, 4096, 8192, 16384])
    parser.add_argument("--reps", type=int, default=2,
                        help="the faster one is the timing; the first is also the cold compile")
    parser.add_argument("--steps", type=int, default=72)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--val-samples", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=3600, help="seconds, per run")
    parser.add_argument("--python", default=sys.executable)
    for name in dict.fromkeys(REFERENCE.values()):
        parser.add_argument(f"--{name}-path", default=None)
    parser.add_argument("--preset", choices=["quick", "full"], default="full")
    parser.add_argument("--retry-failed", action="store_true",
                        help="rerun rows the csv records as failed")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    # Each run is made from the example's own directory, so any path given here has to be
    # resolved against this one before it is handed on.
    args.example = os.path.abspath(args.example)
    for name in dict.fromkeys(REFERENCE.values()):
        if path := getattr(args, f"{name}_path"):
            setattr(args, f"{name}_path", os.path.abspath(path))

    if args.preset == "quick":  # Twenty minutes, to check the machine before the long night.
        args.configs = list(FIRST)
        args.cpu_batches = args.cuda_batches = [32, 512]
        args.reps = 1

    environ = environment(args.python)
    done = completed(args.output)
    keep = {"ok"} if args.retry_failed else {"ok", "failed", "skipped"}
    runs = [r for r in plan(args, has_cuda=bool(environ["gpu"])) if done.get(r) not in keep]

    print(f"{environ['gpu'] or 'no gpu'}, torch {environ['torch']}, "
          f"cuda {environ['cuda'] or 'none'}, python {environ['python']}")
    print(f"{len(runs)} runs to go, {len(done)} already in {args.output}")
    if (os.name == "nt" and not shutil.which("cl")
            and any(run[1] == "cpu" and run[0] not in FIRST for run in runs)):
        print("warning: cl.exe is not on the path, so inductor cannot compile for the cpu and\n"
              "         every compiled cpu run will fail. Start a x64 Native Tools Command\n"
              "         Prompt for VS, or drop those runs with --configs cgenn rotorch.\n"
              "         The cuda runs compile through triton and are unaffected.\n"
              "         Finding cl is not enough on its own: inductor compiles with /openmp,\n"
              "         so omp.h has to be on the include path as well, which a bare conda\n"
              "         prompt does not arrange.")
    if args.dry_run:
        for config, device, batch, rep in runs:
            print(f"  {device:5s} {config:20s} batch {batch:5d} rep {rep}")
        return

    blocked = set()  # (config, device) that has already failed, at a smaller batch size.
    for config, device, batch, rep in runs:
        if (config, device) in blocked:
            print(f"[{time.strftime('%H:%M:%S')}] {config} {device} b{batch}: skipped, "
                  f"this configuration already failed at a smaller batch size")
            write(args.output, dict(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                                    host=socket.gethostname(), device=device, config=config,
                                    batch=batch, rep=rep, status="skipped",
                                    error="failed at a smaller batch size", **environ))
            continue

        print(f"[{time.strftime('%H:%M:%S')}] {config} {device} b{batch} rep {rep} ...",
              flush=True)
        row = run(args.example, config, device, batch, rep, args, environ)
        write(args.output, row)
        if row["status"] == "ok":
            print(f"    {row['median_ms']} ms/step, first step "
                  f"{float(row['first_step_ms']) / 1000:.1f} s"
                  + (f", {row['memory_step_mib']} MiB" if row.get("memory_step_mib")
                     else ""), flush=True)
        else:
            print(f"    {row['error']} after {row['wall_s']} s, see "
                  f"{args.output}.{config}_{device}_b{batch}.log", flush=True)
            blocked.add((config, device))

    summarize(args.output, args.example)
    print(f"\nresults: {args.output}")


if __name__ == "__main__":
    main()
