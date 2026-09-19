"""
Draw the benchmark curves in docs/_static from a sweep csv.

    python docs/plot_benchmarks.py sweep-FM-LAB-0079.csv

The svgs are written by hand rather than by a plotting library so that they can carry their own
stylesheet: the colours are css custom properties that follow the page between light and dark,
which a rasterised or a matplotlib svg cannot do. They are inlined into benchmark.rst with
:code:`.. raw:: html`, so the selectors below see the document root and furo's theme toggle.
"""
import argparse
import csv
import math
import os

# The first five slots of the categorical palette, stepped for each surface, and drawn in that
# order. Validated as a set for both modes: worst adjacent CVD dE 9.1 light, 8.4 dark. Aqua,
# yellow and magenta sit under 3:1 on white, which is why every series is also labelled at its
# end and tabulated beside the figure.
SERIES = {
    "cgenn": ("#2a78d6", "#3987e5"),
    "cgenn-compiled": ("#eb6834", "#d95926"),
    "rotorch": ("#1baf7a", "#199e70"),
    "rotorch-operators": ("#eda100", "#c98500"),
    "rotorch-model": ("#8b6fc9", "#7456b3"),
    "rotorch-triton": ("#e87ba4", "#d55181"),
    "rotorch-triton-model": ("#4fa3a8", "#3d8990"),
}

LABELS = {"cgenn": "cgenn", "cgenn-compiled": "cgenn-model", "rotorch": "rotorch",
          "rotorch-operators": "rotorch-operators", "rotorch-triton": "rotorch-triton",
          "rotorch-model": "rotorch-operators-model", "rotorch-triton-model": "rotorch-triton-model"}

WIDTH, HEIGHT = 760, 430
LEFT, RIGHT, TOP, BOTTOM = 64, 158, 54, 52  # Room for the tick labels, the end labels, the legend.


def read(path):
    """(device, config, batch) -> the faster of the repetitions, as a row."""
    best = {}
    with open(path, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["status"] != "ok" or not row["median_ms"]:
                continue
            key = (row["device"], row["config"], int(row["batch"]))
            if float(row["median_ms"]) < float(best.get(key, {"median_ms": "inf"})["median_ms"]):
                best[key] = row
    return best


def ticks(low, high):
    """The 1-2-5 decade steps that enclose the data, so that the axis ends on a round number."""
    steps = [step * 10 ** power for power in range(-2, 8) for step in (1, 2, 5)]
    start = max([s for s in steps if s <= low], default=steps[0])
    stop = min([s for s in steps if s >= high], default=steps[-1])
    return [s for s in steps if start <= s <= stop]


def compact(value):
    """Short enough for an axis: 500, 2k, 10k. Only for the measured values, never a batch
    size, which is a power of two and has to be read as one."""
    return f"{value / 1000:g}k" if value >= 1000 else f"{value:g}"


def declutter(labels, spacing=14):
    """Push overlapping end labels apart, keeping their order, so none sits on another."""
    labels.sort(key=lambda item: item[0])
    for i in range(1, len(labels)):
        if labels[i][0] - labels[i - 1][0] < spacing:
            labels[i] = (labels[i - 1][0] + spacing, *labels[i][1:])
    return labels


def chart(best, device, field, unit, title, description, path, drop=()):
    configs = [c for c in SERIES if c not in drop
               and any(key[:2] == (device, c) and best[key].get(field) for key in best)]
    points = {c: sorted((int(k[2]), float(best[k][field])) for k in best
                        if k[:2] == (device, c) and best[k].get(field)) for c in configs}
    batches = sorted({b for series in points.values() for b, _ in series})
    values = [v for series in points.values() for _, v in series]
    y_ticks = ticks(min(values), max(values))

    def x_of(batch):
        lo, hi = math.log2(batches[0]), math.log2(batches[-1])
        return LEFT + (math.log2(batch) - lo) / (hi - lo) * (WIDTH - LEFT - RIGHT)

    def y_of(value):
        lo, hi = math.log10(y_ticks[0]), math.log10(y_ticks[-1])
        return HEIGHT - BOTTOM - (math.log10(value) - lo) / (hi - lo) * (HEIGHT - TOP - BOTTOM)

    out = [f'<svg class="viz" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}"'
           f' width="100%" role="img" aria-labelledby="{path}-t {path}-d">',
           f'<title id="{path}-t">{title}</title>',
           f'<desc id="{path}-d">{description}</desc>', STYLE]

    for value in y_ticks:  # Hairline grid, one shade off the surface, solid.
        y = y_of(value)
        out.append(f'<line class="grid" x1="{LEFT}" y1="{y:.1f}" x2="{WIDTH - RIGHT}" '
                   f'y2="{y:.1f}"/>')
        out.append(f'<text class="tick" x="{LEFT - 10}" y="{y + 4:.1f}" text-anchor="end">'
                   f'{compact(value)}</text>')
    for batch in batches:
        x = x_of(batch)
        out.append(f'<line class="grid" x1="{x:.1f}" y1="{TOP}" x2="{x:.1f}" '
                   f'y2="{HEIGHT - BOTTOM}"/>')
        out.append(f'<text class="tick" x="{x:.1f}" y="{HEIGHT - BOTTOM + 20}" '
                   f'text-anchor="middle">{batch}</text>')

    out.append(f'<text class="axis" x="{LEFT - 10}" y="{TOP - 14}" text-anchor="end">{unit}</text>')
    out.append(f'<text class="axis" x="{(LEFT + WIDTH - RIGHT) / 2:.0f}" y="{HEIGHT - 8}" '
               f'text-anchor="middle">batch size</text>')

    legend_x = LEFT
    for config in configs:  # The legend is always there; the end labels repeat it on the line.
        out.append(f'<rect class="swatch {config}" x="{legend_x}" y="{TOP - 40}" width="22" '
                   f'height="3" rx="1.5"/>')
        out.append(f'<text class="legend" x="{legend_x + 28}" y="{TOP - 34}">'
                   f'{LABELS[config]}</text>')
        legend_x += 34 + 7.2 * len(LABELS[config])

    ends = []
    for config in configs:
        series = points[config]
        line = " ".join(f"{x_of(b):.1f},{y_of(v):.1f}" for b, v in series)
        out.append(f'<polyline class="line {config}" points="{line}"/>')
        for b, v in series:
            out.append(f'<circle class="dot {config}" cx="{x_of(b):.1f}" cy="{y_of(v):.1f}" '
                       f'r="4"><title>{LABELS[config]} · batch {b} · '
                       f'{v:,.1f} {unit}</title></circle>')
        ends.append((y_of(series[-1][1]), x_of(series[-1][0]), config))

    for y, x, config in declutter(ends):
        out.append(f'<text class="end" x="{x + 12:.1f}" y="{y + 4:.1f}">{LABELS[config]}</text>')

    out.append("</svg>")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(out) + "\n")
    print(f"wrote {path}")


STYLE = """<style>
.viz { font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  --ink: #0b0b0b; --ink-2: #52514e; --grid: #e7e7e4; --surface: #ffffff;
  --cgenn: #2a78d6; --cgenn-compiled: #eb6834; --rotorch: #1baf7a;
  --rotorch-operators: #eda100; --rotorch-triton: #e87ba4; }
.viz .grid { stroke: var(--grid); stroke-width: 1; }
.viz .tick { fill: var(--ink-2); font-size: 11px; font-variant-numeric: tabular-nums; }
.viz .axis { fill: var(--ink-2); font-size: 11px; }
.viz .legend { fill: var(--ink); font-size: 12px; }
.viz .end { fill: var(--ink-2); font-size: 11px; }
.viz .line { fill: none; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
.viz .dot { stroke: var(--surface); stroke-width: 2; }
.viz .cgenn { stroke: var(--cgenn); } .viz circle.cgenn { fill: var(--cgenn); }
.viz rect.cgenn { fill: var(--cgenn); }
.viz .cgenn-compiled { stroke: var(--cgenn-compiled); }
.viz circle.cgenn-compiled, .viz rect.cgenn-compiled { fill: var(--cgenn-compiled); }
.viz .rotorch { stroke: var(--rotorch); }
.viz circle.rotorch, .viz rect.rotorch { fill: var(--rotorch); }
.viz .rotorch-operators { stroke: var(--rotorch-operators); }
.viz circle.rotorch-operators, .viz rect.rotorch-operators { fill: var(--rotorch-operators); }
.viz .rotorch-triton { stroke: var(--rotorch-triton); }
.viz circle.rotorch-triton, .viz rect.rotorch-triton { fill: var(--rotorch-triton); }
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz, body:where(:not([data-theme="light"])) .viz {
    --ink: #ffffff; --ink-2: #c3c2b7; --grid: #2b2c2f; --surface: #131416;
    --cgenn: #3987e5; --cgenn-compiled: #d95926; --rotorch: #199e70;
    --rotorch-operators: #c98500; --rotorch-triton: #d55181; } }
:root[data-theme="dark"] .viz, body[data-theme="dark"] .viz {
  --ink: #ffffff; --ink-2: #c3c2b7; --grid: #2b2c2f; --surface: #131416;
  --cgenn: #3987e5; --cgenn-compiled: #d95926; --rotorch: #199e70;
  --rotorch-operators: #c98500; --rotorch-triton: #d55181; }
</style>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="?", default="sweep-FM-LAB-0079.csv")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "_static"))
    args = parser.parse_args()

    best = read(args.results)
    os.makedirs(args.out, exist_ok=True)
    chart(best, "cuda", "median_ms", "ms / step",
          "Milliseconds per step against batch size, on the GPU",
          "Log-log lines. cgenn, its compiled form and the triton backend share a floor near "
          "25 ms up to batch 512 and rise after it; the triton line is the lowest from there "
          "on, 82 ms against cgenn's 489 at batch 16384. Eager rotorch holds a flat 228 ms "
          "until the batch overtakes its launch overhead and only passes cgenn at 8192.",
          os.path.join(args.out, "hulls-cuda-time.svg"))
    chart(best, "cuda", "memory_step_mib", "peak MiB",
          "Peak memory of a forward and backward against batch size, on the GPU",
          "Log-log lines. Both cgenn curves sit an order of magnitude above all three rotorch "
          "curves at every batch size, reaching 11.7 GiB against 2.5, 2.1 and 1.5 GiB at "
          "batch 16384.",
          os.path.join(args.out, "hulls-cuda-memory.svg"))
    chart(best, "cpu", "median_ms", "ms / step",
          "Milliseconds per step against batch size, on the CPU",
          "Log-log lines. rotorch starts a fifth behind cgenn at batch 32, crosses it before "
          "128 and is three times faster at 2048.",
          os.path.join(args.out, "hulls-cpu-time.svg"),
          drop=["rotorch-triton"])  # No cpu triton: the same line as rotorch, within 1.5%.


if __name__ == "__main__":
    main()
