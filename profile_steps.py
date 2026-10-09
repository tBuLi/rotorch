"""
Profile the training steps of one sweep cell: where the gpu time of a step goes, kernel by kernel.

    python profile_steps.py o3 16384 --backend triton --compile model
    FROZEN=1 python profile_steps.py hulls 16384 --backend triton --compile model

Runs examples/<example>.py as the sweep does, under torch.profiler for 10 steps after the warmup, and prints per step the
profiled wall time, the time the gpu was busy, the kernel launches, the kernels by gpu time and the total per kind.
FROZEN=1 freezes the parameters and has the input require grad instead, so that every backward still runs, without weight gradients.
"""
import os
import runpy
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile, schedule

example, batch, *flags = sys.argv[1:]
WARMUP, ACTIVE = 8, 10
here = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(here, "examples"))
sys.path.insert(0, os.getcwd())
sys.argv = [f"{example}.py", "--device", "cuda", "--batch-size", batch, "--steps", str(WARMUP + 2 + ACTIVE), "--warmup", str(WARMUP),
            "--train-samples", str(max(256, int(batch))), "--print-interval", "1000", *flags]

if os.environ.get("FROZEN") == "1":
    import benchmark
    run = benchmark.run

    def frozen(task):
        for impl, build in list(task.models.items()):
            def wrapped(args, build=build):
                model, loss_fn = build(args)
                frozen = []

                def loss(first, *rest):
                    value = loss_fn(first.detach().requires_grad_(True), *rest)
                    if not frozen:  # After the first call, which gives the lazy layers their parameters.
                        model.requires_grad_(False)
                        frozen.append(True)
                    return value
                return model, loss
            task.models[impl] = wrapped
        return run(task)

    benchmark.run = frozen

marks = []
profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], schedule=schedule(wait=WARMUP, warmup=2, active=ACTIVE, repeat=1))
adam_step = torch.optim.Adam.step


def step(self, *args, **kwargs):
    result = adam_step(self, *args, **kwargs)
    torch.cuda.synchronize()
    marks.append(time.perf_counter())
    profiler.step()
    return result


torch.optim.Adam.step = step
with profiler:
    runpy.run_path(f"{example}.py", run_name="__main__")

kernels = [e for e in profiler.events() if e.device_type == torch.autograd.DeviceType.CUDA]
intervals = sorted((e.time_range.start, e.time_range.end) for e in kernels)
busy, end = 0.0, -1.0
for start, stop in intervals:
    busy += max(0.0, stop - max(start, end))
    end = max(end, stop)
wall = (marks[WARMUP + 2 + ACTIVE - 1] - marks[WARMUP + 2 - 1]) / ACTIVE * 1e3 if len(marks) >= WARMUP + 2 + ACTIVE else float("nan")


def kind(name):
    if name.endswith(("_fwd", "_bwd")) or "_x_" in name:
        return "kingdon triton"
    if name.startswith(("triton_", "triton")):
        return "inductor"
    if any(s in name.lower() for s in ("gemm", "cutlass", "cublas", "sm80_", "sm86_", "ampere_", "xmma")):
        return "cublas"
    if "adam" in name.lower() or "multi_tensor" in name:
        return "optimizer"
    if "memset" in name.lower() or "memcpy" in name.lower() or "fill" in name.lower():
        return "fill/copy"
    return "other"


per = {}
for e in kernels:
    calls, total = per.get(e.name, (0, 0.0))
    per[e.name] = (calls + 1, total + e.time_range.end - e.time_range.start)
kinds = {}
for name, (calls, total) in per.items():
    kinds[kind(name)] = kinds.get(kind(name), 0.0) + total
print(f"\n== profile {example} batch {batch} {' '.join(flags) or '(eager torch)'}{' FROZEN' if os.environ.get('FROZEN') == '1' else ''}")
print(f"per step: wall {wall:.2f} ms (profiled), gpu busy {busy / ACTIVE / 1e3:.2f} ms, {len(kernels) / ACTIVE:.0f} launches")
print("by kind, ms/step: " + ", ".join(f"{k} {v / ACTIVE / 1e3:.2f}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])))
print(f"{'kernel':90s} {'calls/step':>10s} {'us/call':>9s} {'ms/step':>8s}")
for name, (calls, total) in sorted(per.items(), key=lambda kv: -kv[1][1])[:25]:
    print(f"{name[:90]:90s} {calls / ACTIVE:10.1f} {total / calls:9.1f} {total / ACTIVE / 1e3:8.3f}")
