"""Refresh the measured examples in the JCP readiness review from saved results."""

import json
from pathlib import Path


REPORTS = Path(__file__).resolve().parent
TREES = REPORTS.parent / "worktrees"
REPORT = REPORTS / "2026-11-15-jcp-submission-readiness.md"
BEGIN = "<!-- BEGIN GENERATED EVIDENCE -->"
END = "<!-- END GENERATED EVIDENCE -->"


def read(path):
    return json.loads(path.read_text())


def plateau_rows():
    cases = read(REPORTS / "2026-11-05-quadrature-per-case" / "data.json")
    for dim, family, label in [("2D", "fibonacci", "Fibonacci"), ("3D", "lattice", "CBC lattice")]:
        panels = sorted(
            (case for case in cases if case["pde"] == "burgers" and case["dim"] == dim),
            key=lambda case: case["mesh"],
        )
        counts = {
            next(rule["mstar"] for rule in panel["families"] if rule["key"] == family)
            for panel in panels
        }
        if len(counts) != 1 or None in counts:
            raise ValueError(f"The common {family} plateau requires review.")
        yield (
            f"Burgers {dim}, {panels[0]['label']}–{panels[-1]['label']}",
            f"{counts.pop():,} {label} points across these meshes",
            "First qualifying size on the tested ladder using cohort median and worst errors. Historical held-out cases were reused.",
        )


def measured_rows():
    yield from plateau_rows()
    burgers = TREES / "2026-10-01-quadrature-burgers3d" / "experiments" / "quadrature-burgers3d"
    panel = read(burgers / "runs/ho65/code/output/result.json")
    tensor = panel["sizes"]["tensor_R512"]["bytes"] / 1e9
    gauss = panel["sizes"]["gl24_R512"]["bytes"] / 1e9
    yield (
        "Burgers 3D, width 512",
        f"{tensor:.2f} GB tensor and {gauss:.2f} GB Gauss",
        "Advection arrays only. Decimal GB. The full model also stores other data.",
    )
    ns = TREES / "2026-11-05-ns3d-offmesh-quadrature" / "experiments" / "ns3d-offmesh-quadrature"
    ratios = []
    for mesh in [32, 64, 96]:
        panel = read(ns / "runs" / f"q{mesh}" / "output" / "summary.json")
        timings = panel["timing"]["fast"]
        allowed = []
        for family in panel["config"]["families"]:
            rule = panel["mstar"][f"net/{family}/head"]["mstar_rule"]
            key = f"net_head_{rule}"
            if rule and key in timings:
                allowed.append(timings[key]["median_ms"])
        ratios.append(min(allowed) / timings["tensor_head"]["median_ms"])
    yield (
        "Navier–Stokes 3D, width 64",
        f"{min(ratios):.2f}–{max(ratios):.2f}× tensor query time",
        "Fastest timed family at its accuracy plateau, paired within each mesh allocation on development data.",
    )


def main():
    report = REPORT.read_text()
    before, section = report.split(BEGIN)
    _, after = section.split(END)
    lines = [
        "## Selected measurements",
        "",
        "| Problem and setting | Measured result | What was compared |",
        "|---|---|---|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in measured_rows())
    lines.extend([
        "",
        "[Burgers point-count data](2026-11-05-quadrature-per-case/data.json) · "
        "[Burgers storage study](2026-10-01-burgers3d-offmesh-quadrature.md) · "
        "[Navier–Stokes timing study](https://github.com/tahmidawal/Tunable-NM-ROM/blob/"
        "e10561c1d8a9cc9176a0adc8e4040c99c9818534/reports/"
        "2026-11-05-ns3d-offmesh-quadrature.md).",
    ])
    REPORT.write_text(before + BEGIN + "\n" + "\n".join(lines) + "\n" + END + after)


if __name__ == "__main__":
    main()
