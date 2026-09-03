"""Generate every table in WAVE2D-NOTES.md from runs/wav2d/*.json.  Never hand-type a number.

  $PY wav2d_tables.py            -> writes tables/wav2d-*.md and prints them
"""
from __future__ import annotations

import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs", "wav2d")
OUT = os.path.join(HERE, "tables")
os.makedirs(OUT, exist_ok=True)


def fmt(x, nd=2):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "PASS" if x else "FAIL"
    if isinstance(x, (int,)):
        return str(x)
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if xf != xf:
        return "nan"
    if xf == 0.0:
        return "0"
    if abs(xf) >= 1e3 or abs(xf) < 1e-2:
        return f"{xf:.{nd}e}"
    return f"{xf:.{nd+1}f}"


def phase1_table():
    rows = []
    files = sorted(glob.glob(os.path.join(RUNS, "wav2d_fom_gates_N*.json")))
    for f in files:
        r = json.load(open(f))
        N = r["N"]
        for bc in ("ref", "abs"):
            G = r["gates"][bc]
            for k, v in G.items():
                if not (isinstance(v, dict) and "passed" in v):
                    continue
                if k == "F2-spatial":
                    val = (f"bump order {fmt(v['order'], 3)} (errors " + ", ".join(fmt(e) for e in v["errors"]) +
                           f"); two-mode order {fmt(v.get('order_modes'), 3)}; N {v['N']} vs {v['ref']}")
                    ctrl = f"wrong-reference order {fmt(v.get('control_order_wrongref'), 3)}"
                elif k == "F2-temporal":
                    val = f"order {fmt(v['order'], 3)} (errors " + ", ".join(fmt(e) for e in v["errors"]) + ")"
                    ctrl = f"BE order {fmt(v['control_order_BE'], 2)}, separation {fmt(v['control_separation'], 1)}x"
                elif k == "V1cg":
                    val = "CG ladder " + ", ".join(fmt(x) for x in v["ladder"].values()) + f" (monotone {v['monotone']}); achieved CG resid {fmt(v['achieved_cg_relresid_1e13'])}"
                    ctrl = "—"
                elif k == "V1alg":
                    val = f"{fmt(v['value'])} over 10 steps; full horizon {fmt(v['value_full_horizon'])}"
                    ctrl = fmt(v.get("control_value"))
                elif k == "F3":
                    val = ("slope " + fmt(v["slope"], 3) + "; fraction/prediction " +
                           ", ".join(fmt(x, 3) for x in v["ratio_to_prediction"]) + f" at N {v['N']}" +
                           "; plateau/fraction " + ", ".join(fmt(dg["plateau_fraction"] / fr, 3) for dg, fr in zip(v.get("diagnostics", []), v["reflected_fraction"])) +
                           "; y-var " + ", ".join(fmt(dg["y_invariance"], 1) for dg in v.get("diagnostics", [])))
                    ctrl = f"reflective retains {fmt(v['control_reflective_fraction'], 3)}"
                elif k == "F0d-spd":
                    val = f"min eig/max(M) = {fmt(v['min_eig_over_maxM'])}"
                    ctrl = "—"
                else:
                    val = fmt(v.get("value"))
                    ctrl = fmt(v.get("control_value")) if "control_value" in v else "—"
                rows.append((N, bc, k, val, f"≤ {fmt(v['threshold'], 0)}" if "threshold" in v else "see design", ctrl, fmt(v["passed"])))
    lines = ["| N | BC | gate | value | pass rule | negative control | verdict |", "|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    # reported (not gated)
    lines2 = ["", "| N | BC | reported quantity | value |", "|---|---|---|---|"]
    for f in files:
        r = json.load(open(f)); N = r["N"]
        A = r["gates"]["abs"]
        lines2.append(f"| {N} | abs | family-blob energy ratio E(T)/E0, E(4T)/E0 (F1/F4 trajectory) | {fmt(A.get('absorbing_energy_ratio_T'))}, {fmt(A.get('absorbing_energy_ratio_4T'))} |")
        fam = A.get("F2-spatial-family-reported")
        if fam:
            lines2.append(f"| {N} | abs | family widest blob (w={fmt(fam['w'])}) spatial order, NOT gated | {fmt(fam['order'], 3)} (errors " + ", ".join(fmt(e) for e in fam["errors"]) + ") |")
        fam = r["gates"]["ref"].get("F2-spatial-family-reported")
        if fam:
            lines2.append(f"| {N} | ref | family widest blob (w={fmt(fam['w'])}) spatial order, NOT gated | {fmt(fam['order'], 3)} (errors " + ", ".join(fmt(e) for e in fam["errors"]) + ") |")
        lines2.append(f"| {N} | ref | V0 energy agreement with the frozen 08-14 FOM | {fmt(r['gates']['ref'].get('V0_energy_reldiff'))} |")
        lines2.append(f"| {N} | both | provenance | commit {r['provenance']['git_commit'][:8]}, backend {r['provenance']['jax_backend']}, jax {r['provenance']['jax_version']}, matmul {r['provenance']['matmul_precision']}, wall {fmt(r['wall_s'], 0)} s |")
    txt = "\n".join(lines + lines2) + "\n"
    open(os.path.join(OUT, "wav2d-phase1-gates.md"), "w").write(txt)
    return txt


def splice(notes_path, marker, text):
    """replace the block between <!-- {marker} --> and <!-- /{marker} --> in the notes file"""
    if not os.path.exists(notes_path):
        return
    src = open(notes_path).read()
    a, b = f"<!-- {marker} -->", f"<!-- /{marker} -->"
    if a in src and b in src:
        i, j = src.index(a) + len(a), src.index(b)
        src = src[:i] + "\n" + text + src[j:]
        open(notes_path, "w").write(src)


if __name__ == "__main__":
    t1 = phase1_table()
    print(t1)
    splice(os.path.join(HERE, "WAVE2D-NOTES.md"), "phase1-table", t1)
