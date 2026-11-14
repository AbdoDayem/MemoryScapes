"""Run the audio pipeline on procedural example memories and report how it did.

    cd services/workers
    .venv/Scripts/python examples/run_example.py            # all scenarios
    .venv/Scripts/python examples/run_example.py park_walk  # one

For each scenario, examples/out/<name>/ gets the full stage output tree (listen to
ingest/audio.wav and separate/stems/*.wav), plus report.md and map.svg (top-down view:
camera path, true sources, estimates with their spread).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from evaluate import BAD_VERDICTS, Row, evaluate, summary  # noqa: E402
from scenes import SCENARIOS, OracleDetector, OracleSeparator, OracleTagger, Rendered, render, write_inputs  # noqa: E402

from memoryscapes_workers.context import StageContext  # noqa: E402
from memoryscapes_workers.providers.soundscape import StubProvider  # noqa: E402
from memoryscapes_workers.stages import localize, separate, soundscape  # noqa: E402
from memoryscapes_workers.storage import LocalStorage  # noqa: E402


def run_scenario(name: str, root: Path) -> tuple[Rendered, dict, list[Row], dict]:
    scenario = SCENARIOS[name]
    ctx = StageContext(memory_id=name, storage=LocalStorage(root))
    r = render(scenario)
    write_inputs(ctx, r)
    timings = {}
    t0 = time.perf_counter()
    separate.run(ctx, tagger=OracleTagger(r), separator=OracleSeparator(r))
    timings["separate"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    localized = localize.run(ctx, detector=OracleDetector(r))
    timings["localize"] = time.perf_counter() - t0
    soundscape.run(ctx, provider=StubProvider())
    return r, localized, evaluate(r, localized), timings


def _fmt(v):
    return "-" if v is None else "(" + ", ".join(f"{x:.1f}" for x in v) + ")"


def write_report(path: Path, r: Rendered, rows: list[Row], timings: dict) -> None:
    c = r.scenario.conditions
    lines = [
        f"# {r.scenario.name}",
        "",
        r.scenario.description,
        "",
        f"Conditions: {c.layout} mic (true spacing {c.spacing_m} m, metadata says {c.reported_spacing_m}), "
        f"camera {c.camera}, reverb RT60 {c.reverb_rt60_s} s, noise {c.noise_dbfs} dBFS, "
        f"separation leakage {c.leakage_db} dB, detector misses {c.detector_miss_rate:.0%}, "
        f"false boxes {c.detector_false_rate:.0%}/frame.",
        "",
        "| Sound | True position | Estimate | Error (m) | Method | Confidence | Spread (m) | Verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        err = "-" if row.error_m is None else f"{row.error_m:.2f}"
        spread = "-" if row.spread_m is None else f"{row.spread_m:.2f}"
        lines.append(
            f"| {row.label} | {_fmt(row.truth)} | {_fmt(row.estimate)} | {err} | {row.method} | {row.confidence:.2f} | {spread} | {row.verdict} |"
        )
    lines += ["", f"Timing: separate {timings['separate']:.1f} s, localize {timings['localize']:.1f} s.", "", "![map](map.svg)", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_map(path: Path, r: Rendered, rows: list[Row]) -> None:
    """Top-down SVG: x to the right, z (forward) up."""
    pts = [c.position for c in r.cameras] + [s.position for s in r.scenario.sources]
    pts += [row.estimate for row in rows if row.estimate is not None]
    xs, zs = [p[0] for p in pts], [p[2] for p in pts]
    pad = 2.0
    x0, x1, z0, z1 = min(xs) - pad, max(xs) + pad, min(zs) - pad, max(zs) + pad
    scale = 560 / max(x1 - x0, z1 - z0)
    w, h = (x1 - x0) * scale, (z1 - z0) * scale

    def X(x):
        return (x - x0) * scale

    def Z(z):
        return h - (z - z0) * scale

    el = [f'<rect width="{w:.0f}" height="{h + 40:.0f}" fill="#f7f8fa"/>']
    path_d = " ".join(f"{'M' if i == 0 else 'L'}{X(c.position[0]):.1f},{Z(c.position[2]):.1f}" for i, c in enumerate(r.cameras))
    el.append(f'<path d="{path_d}" stroke="#6b7785" stroke-width="2" fill="none" stroke-dasharray="4 3"/>')
    for c in r.cameras[::4]:
        fx, fz = c.c2w[0, 2], c.c2w[2, 2]
        el.append(f'<line x1="{X(c.position[0]):.1f}" y1="{Z(c.position[2]):.1f}" x2="{X(c.position[0] + fx * 0.8):.1f}" y2="{Z(c.position[2] + fz * 0.8):.1f}" stroke="#6b7785" stroke-width="1.5"/>')
    for row in rows:
        if row.truth is not None:
            tx, tz = X(row.truth[0]), Z(row.truth[2])
            el.append(f'<circle cx="{tx:.1f}" cy="{tz:.1f}" r="7" fill="#0d7a80"/>')
            el.append(f'<text x="{tx + 10:.1f}" y="{tz - 8:.1f}" font-size="12" fill="#15212b" font-family="sans-serif">{row.label}</text>')
        if row.estimate is not None:
            ex, ez = X(row.estimate[0]), Z(row.estimate[2])
            color = "#b42318" if row.verdict in BAD_VERDICTS else "#b45309"
            if row.spread_m:
                el.append(f'<circle cx="{ex:.1f}" cy="{ez:.1f}" r="{max(row.spread_m * scale, 3):.1f}" fill="{color}" fill-opacity="0.12" stroke="{color}" stroke-opacity="0.5"/>')
            el.append(f'<path d="M{ex - 6:.1f},{ez - 6:.1f} L{ex + 6:.1f},{ez + 6:.1f} M{ex - 6:.1f},{ez + 6:.1f} L{ex + 6:.1f},{ez - 6:.1f}" stroke="{color}" stroke-width="2.5"/>')
            if row.truth is not None:
                el.append(f'<line x1="{X(row.truth[0]):.1f}" y1="{Z(row.truth[2]):.1f}" x2="{ex:.1f}" y2="{ez:.1f}" stroke="{color}" stroke-width="1" stroke-opacity="0.6"/>')
    legend = "● true source   ✕ estimate (circle = spread)   - - camera path"
    el.append(f'<text x="8" y="{h + 26:.0f}" font-size="12" fill="#56656f" font-family="sans-serif">{legend}</text>')
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.0f}" height="{h + 40:.0f}" viewBox="0 0 {w:.0f} {h + 40:.0f}">{"".join(el)}</svg>', encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenarios", nargs="*", default=list(SCENARIOS))
    ap.add_argument("--out", default=str(HERE / "out"))
    args = ap.parse_args()
    root = Path(args.out)
    index = ["# Example runs", "", "| Scenario | Result |", "|---|---|"]
    bad = 0
    for name in args.scenarios:
        r, _, rows, timings = run_scenario(name, root)
        write_report(root / name / "report.md", r, rows, timings)
        write_map(root / name / "map.svg", r, rows)
        s = summary(rows)
        bad += sum(v for k, v in s.items() if k in BAD_VERDICTS)
        result = ", ".join(f"{v} {k}" for k, v in s.items())
        index.append(f"| [{name}]({name}/report.md) | {result} |")
        print(f"\n== {name} ({timings['separate']:.1f}s + {timings['localize']:.1f}s)")
        for row in rows:
            err = "" if row.error_m is None else f" err {row.error_m:.2f} m"
            print(f"  {row.label:10s} {row.method:8s} conf {row.confidence:.2f}{err}  -> {row.verdict}")
    (root / "README.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    print(f"\nReports in {root}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
