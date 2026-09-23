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
# order. Validated as a set for both modes: worst adjacent CVD dE 9.1 light, 8.4 dark.
#
# Line styles:
# - Solid: Eager
# - Dashed: Compiled
# - Dotted: Triton
SERIES = {
    "cgenn": ("#2a78d6", "#3987e5", "solid"),
    "cgenn-compiled": ("#2a78d6", "#3987e5", "dashed"),
    "rotorch": ("#1baf7a", "#199e70", "solid"),
    "rotorch-operators": ("#eda100", "#c98500", "dashed"),
    "rotorch-model": ("#eb6834", "#d95926", "dashed"),
    "rotorch-triton": ("#e87ba4", "#d55181", "dotted"),
}

LABELS = {"cgenn": "cgenn", "cgenn-compiled": "cgenn-compiled", "rotorch": "rotorch",
          "rotorch-operators": "rotorch-operators", "rotorch-model": "rotorch-model",
          "rotorch-triton": "rotorch-triton"}

WIDTH, HEIGHT = 760, 440
LEFT, RIGHT, TOP, BOTTOM = 64, 158, 72, 52  # Room for the tick labels, the end labels, the legend.


def read(path):
    """(device, config, batch) -> the faster of the repetitions, as a row, and the name."""
    best = {}
    name = "hulls"
    with open(path, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("example"):
                name = os.path.splitext(row["example"])[0]
            if row["status"] != "ok" or not row["median_ms"]:
                continue
            key = (row["device"], row["config"], int(row["batch"]))
            if float(row["median_ms"]) < float(best.get(key, {"median_ms": "inf"})["median_ms"]):
                best[key] = row
    return best, name


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


def table(best, device, field, unit, decimals, caption, path):
    """The full grid behind a chart: one row per batch size, one column per run, each cell the
    measured value and, in brackets, its ratio against cgenn at that same batch size. Time
    ratios are cgenn/this (bigger is faster); memory ratios are this/cgenn (smaller is less)."""
    configs = [c for c in SERIES
               if any(key[:2] == (device, c) and best[key].get(field) for key in best)]
    batches = sorted({k[2] for k in best if k[0] == device and best[k].get(field)})

    out = [f'<div class="bench-table">', TABLE_STYLE, f'<table><caption>{caption}</caption>',
           '<thead><tr><th>batch size</th>']
    out += [f'<th class="{c}">{LABELS[c]}</th>' for c in configs]
    out.append('</tr></thead><tbody>')
    for b in batches:
        cgenn = best.get((device, "cgenn", b), {}).get(field)
        cgenn = float(cgenn) if cgenn else None
        out.append(f'<tr><td>{b}</td>')
        for c in configs:
            cell = best.get((device, c, b), {}).get(field)
            if not cell:
                out.append('<td>–</td>')
                continue
            value = float(cell)
            ratio = (cgenn / value if field == "median_ms" else value / cgenn) if cgenn else None
            brackets = f' <span class="ratio">({ratio:.2f}×)</span>' if ratio is not None else ''
            out.append(f'<td>{value:,.{decimals}f}{brackets}</td>')
        out.append('</tr>')
    out.append('</tbody></table></div>')

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(out) + "\n")
    print(f"wrote {path}")


def chart(best, device, field, unit, title, path):
    configs = [c for c in SERIES
               if any(key[:2] == (device, c) and best[key].get(field) for key in best)]
    points = {c: sorted((int(k[2]), float(best[k][field])) for k in best
                        if k[:2] == (device, c) and best[k].get(field)) for c in configs}
    description = describe(points, unit)
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

    legend_x, legend_y = LEFT, TOP - 54
    for config in configs:  # The legend is always there; the end labels repeat it on the line.
        width = 34 + 7.2 * len(LABELS[config])
        if legend_x + width > WIDTH - RIGHT:
            legend_x, legend_y = LEFT, legend_y + 18
        out.append(f'<rect class="swatch {config}" x="{legend_x}" y="{legend_y}" width="22" '
                   f'height="3" rx="1.5"/>')
        out.append(f'<text class="legend" x="{legend_x + 28}" y="{legend_y + 6}">'
                   f'{LABELS[config]}</text>')
        legend_x += width

    ends = []
    for config in configs:
        series = points[config]
        line = " ".join(f"{x_of(b):.1f},{y_of(v):.1f}" for b, v in series)
        linestyle = SERIES[config][2]
        out.append(f'<polyline class="line {config} {linestyle}" points="{line}"/>')
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
  --cgenn: #2a78d6; --cgenn-compiled: #2a78d6; --rotorch: #1baf7a;
  --rotorch-operators: #eda100; --rotorch-model: #eb6834;
  --rotorch-triton: #e87ba4; }
.viz .grid { stroke: var(--grid); stroke-width: 1; }
.viz .tick { fill: var(--ink-2); font-size: 11px; font-variant-numeric: tabular-nums; }
.viz .axis { fill: var(--ink-2); font-size: 11px; }
.viz .legend { fill: var(--ink); font-size: 12px; }
.viz .end { fill: var(--ink-2); font-size: 11px; }
.viz .line { fill: none; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
.viz .line.dashed { stroke-dasharray: 6 4; }
.viz .line.dotted { stroke-dasharray: 2 3; }
.viz .dot { stroke: var(--surface); stroke-width: 2; }
.viz .cgenn { stroke: var(--cgenn); } .viz circle.cgenn { fill: var(--cgenn); }
.viz rect.cgenn { fill: var(--cgenn); }
.viz .cgenn-compiled { stroke: var(--cgenn-compiled); }
.viz circle.cgenn-compiled, .viz rect.cgenn-compiled { fill: var(--cgenn-compiled); }
.viz .rotorch { stroke: var(--rotorch); }
.viz circle.rotorch, .viz rect.rotorch { fill: var(--rotorch); }
.viz .rotorch-operators { stroke: var(--rotorch-operators); }
.viz circle.rotorch-operators, .viz rect.rotorch-operators { fill: var(--rotorch-operators); }
.viz .rotorch-model { stroke: var(--rotorch-model); }
.viz circle.rotorch-model, .viz rect.rotorch-model { fill: var(--rotorch-model); }
.viz .rotorch-triton { stroke: var(--rotorch-triton); }
.viz circle.rotorch-triton, .viz rect.rotorch-triton { fill: var(--rotorch-triton); }
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz, body:where(:not([data-theme="light"])) .viz {
    --ink: #ffffff; --ink-2: #c3c2b7; --grid: #2b2c2f; --surface: #131416;
    --cgenn: #3987e5; --cgenn-compiled: #3987e5; --rotorch: #199e70;
    --rotorch-operators: #c98500; --rotorch-model: #d95926;
    --rotorch-triton: #d55181; } }
:root[data-theme="dark"] .viz, body[data-theme="dark"] .viz {
  --ink: #ffffff; --ink-2: #c3c2b7; --grid: #2b2c2f; --surface: #131416;
  --cgenn: #3987e5; --cgenn-compiled: #3987e5; --rotorch: #199e70;
  --rotorch-operators: #c98500; --rotorch-model: #d95926;
  --rotorch-triton: #d55181; }
</style>"""

# The same palette as STYLE, so a column header matches the colour of its line in the chart
# above it. Built from SERIES rather than repeated by hand, so the two cannot drift apart.
_LIGHT_VARS = "; ".join(f"--{c}: {SERIES[c][0]}" for c in SERIES)
_DARK_VARS = "; ".join(f"--{c}: {SERIES[c][1]}" for c in SERIES)
_HEADER_COLORS = "\n".join(f'.bench-table th.{c} {{ color: var(--{c}); }}' for c in SERIES)

TABLE_STYLE = f"""<style>
.bench-table {{ {_LIGHT_VARS}; --ink: #0b0b0b; --ink-2: #52514e; --border: #e7e7e4; }}
.bench-table table {{ border-collapse: collapse; width: 100%;
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  font-size: 13px; font-variant-numeric: tabular-nums; }}
.bench-table caption {{ caption-side: top; text-align: left; color: var(--ink-2);
  font-size: 12px; padding-bottom: 4px; }}
.bench-table th, .bench-table td {{ padding: 4px 10px; text-align: right;
  border-bottom: 1px solid var(--border); white-space: nowrap; }}
.bench-table th:first-child, .bench-table td:first-child {{ text-align: left; color: var(--ink);
  font-weight: 400; }}
.bench-table th {{ font-weight: 600; }}
{_HEADER_COLORS}
.bench-table .ratio {{ color: var(--ink-2); }}
@media (prefers-color-scheme: dark) {{
  :root:where(:not([data-theme="light"])) .bench-table, body:where(:not([data-theme="light"])) .bench-table {{
    --ink: #ffffff; --ink-2: #c3c2b7; --border: #2b2c2f; {_DARK_VARS} }} }}
:root[data-theme="dark"] .bench-table, body[data-theme="dark"] .bench-table {{
  --ink: #ffffff; --ink-2: #c3c2b7; --border: #2b2c2f; {_DARK_VARS} }}
</style>"""


def describe(points, unit):
    """Alt text from the data, so every figure says what it actually shows."""
    ends = {c: series[-1] for c, series in points.items()}
    batch = max(b for b, _ in ends.values())
    return ("Log-log lines, one per run, against batch size. At batch " + str(batch) + ": "
            + ", ".join(f"{LABELS[c]} {v:,.0f} {unit}" + ("" if b == batch else f" (at batch {b})")
                        for c, (b, v) in sorted(ends.items(), key=lambda item: item[1][1])) + ".")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "_static"))
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for results in args.results:
        best, name = read(results)
        for device, field, unit, decimals, what in [
                ("cuda", "median_ms", "ms / step", 1, "Milliseconds per step"),
                ("cuda", "memory_step_mib", "peak MiB", 1, "Peak memory of a forward and backward"),
                ("cpu", "median_ms", "ms / step", 1, "Milliseconds per step"),
                ("cpu", "memory_step_mib", "peak MiB", 1, "Peak memory of a forward and backward")]:
            if not any(k[0] == device and best[k].get(field) for k in best):
                continue
            kind = "time" if field == "median_ms" else "memory"
            title = f"{what} against batch size, {name} on the {device.upper()}"
            chart(best, device, field, unit, title, os.path.join(args.out, f"{name}-{device}-{kind}.svg"))
            table(best, device, field, unit, decimals, f"{what} ({unit})",
                  os.path.join(args.out, f"{name}-{device}-{kind}-table.html"))


if __name__ == "__main__":
    main()
