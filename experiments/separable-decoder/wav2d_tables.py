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




def _g(v, key="value"):
    return fmt(v.get(key)) if isinstance(v, dict) else "—"


def phase2_table(files=None):
    import numpy as _np
    if files is None:
        files = [f for f in sorted(glob.glob(os.path.join(RUNS, "wav2d_head_gates_*_N*_R*.json"))) if "SMOKE" not in f]
    lines = ["| N | BC | head | K | params | final loss | D0 | D1 held-out/POD-K (ctrl shuffled) | D2 min cond (ctrl dup.) | G0a ratio, gap (ctrl) | G0b tangent/POD-K (ctrl random) | G0 | predicted |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    lines2 = ["", "| N | BC | head | held-out oracle median | train oracle median | POD-K median | POD-R ceiling median | G0b tangent median / POD-K median (n states) |", "|---|---|---|---|---|---|---|---|"]
    for f in files:
        r = json.load(open(f)); N, bc = r["N"], r["bc"]; D0 = r["D0"]
        d0 = f"{'PASS' if D0['passed'] else 'FAIL'} (orth {fmt(D0['orthonormality'])}, floor {fmt(D0['floor_bank_median'], 3)}, sigma_R/sigma_1 {fmt(D0['sigma_ratio_R'])})"
        for mode, H in r["heads"].items():
            G = H["gates"]
            lines.append(f"| {N} | {bc} | {mode} | {H['config']['K']} | {H['n_params']} | {fmt(H['final_loss'])} | {d0} | "
                         f"{fmt(G['D1']['value'], 3)} ({fmt(G['D1'].get('control_value'), 2)}) {fmt(G['D1']['passed'])} | "
                         f"{fmt(min(G['D2']['cond_train_min'], G['D2']['cond_test_min']))} ({fmt(-G['D2'].get('control_value', float('nan')))}) {fmt(G['D2']['passed'])} | "
                         f"{fmt(G['G0a']['ratio'], 3)}, {fmt(G['G0a']['abs_gap'], 3)} ({fmt(G['G0a'].get('control_value'), 3)}) {fmt(G['G0a']['passed'])} | "
                         f"{fmt(G['G0b']['value'], 3)} ({fmt(G['G0b'].get('control_value'), 3)}) {fmt(G['G0b']['passed'])} | "
                         f"{fmt(H['G0_passed'])} | {fmt(H['predicted_G0'])} |")
            Gb = G["G0b"]
            lines2.append(f"| {N} | {bc} | {mode} | {fmt(_np.median(H['oracle_heldout_per_traj']), 4)} | {fmt(_np.median(H['oracle_train_per_traj']), 4)} | "
                          f"{fmt(_np.median(H['podK_heldout_per_traj']), 4)} | {fmt(_np.median(H['podR_ceiling_per_traj']), 4)} | "
                          f"{fmt(Gb['tangent_median'], 4)} / {fmt(Gb['podK_median'], 4)} ({Gb['n_states']}) |")
    txt = "\n".join(lines + lines2) + "\n"
    open(os.path.join(OUT, "wav2d-phase2-gates.md"), "w").write(txt)
    return txt


def phase3_table(files=None, p2files=None):
    import numpy as _np
    if files is None:
        files = [f for f in sorted(glob.glob(os.path.join(RUNS, "wav2d_rom_gates_*_N*_R*.json"))) if "SMOKE" not in f]
    if p2files is None:
        p2files = [f for f in glob.glob(os.path.join(RUNS, "wav2d_head_gates_*_N*_R*.json")) if "SMOKE" not in f]
    lines = ["| N | BC | head | arm | RS | complete | err_T median | err_4T median | oracle floor T / 4T | POD-K T / 4T | same-dt FOM | energy ratio T (Er arm C / dyn arm A) | iters |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    glines = ["", "| N | BC | head | gate | value | threshold | control | verdict | note |", "|---|---|---|---|---|---|---|---|---|"]
    for f in files:
        r = json.load(open(f)); N, bc = r["N"], r["bc"]
        for k in ("W1", "W1-Cterm"):
            if k in r["tables"]:
                v = r["tables"][k]
                glines.append(f"| {N} | {bc} | — | {k} | {fmt(v['value'])} | {fmt(v.get('threshold'))} | {_g(v, 'control_value')} | {fmt(v['passed'])} | {v.get('note','')[:160]} |")
        for K_, w2 in r.get("W2", {}).items():
            gv = w2["gate"]
            glines.append(f"| {N} | {bc} | — | W2 POD-{K_} | {fmt(gv.get('value'))} | {fmt(gv.get('threshold'))} | {_g(gv, 'control_value')} | {fmt(gv['passed'])} | error {fmt(w2['error_median'], 4)} vs floor {fmt(w2['floor_median'], 4)}, energy ratio {fmt(w2['energy_ratio_median'], 6)} |")
        for mode, H in r["heads"].items():
            fT, f4 = _np.median(H["floor_T"]), _np.median(H["floor_4T"]); pT, p4 = _np.median(H["podK_floor_T"]), _np.median(H["podK_floor_4T"])
            for arm, arm_res in H["arms"].items():
                for rs, agg in arm_res.items():
                    if agg["n_complete"] == len(agg["per_traj"]):
                        e = agg.get("Er_ratio_T_median", agg.get("Edyn_ratio_T_median", float("nan")))
                        lines.append(f"| {N} | {bc} | {mode} | {arm} | {rs} | {agg['n_complete']}/{len(agg['per_traj'])} | {fmt(agg['err_T_median'], 4)} | {fmt(agg['err_4T_median'], 4)} | "
                                     f"{fmt(fT, 4)} / {fmt(f4, 4)} | {fmt(pT, 4)} / {fmt(p4, 4)} | {fmt(_np.median(r['samedt_fom_error'][rs]))} | {fmt(e, 4)} | {fmt(agg.get('iters_mean_median'), 1)} |")
                    else:
                        lines.append(f"| {N} | {bc} | {mode} | {arm} | {rs} | {agg['n_complete']}/{len(agg['per_traj'])} | INCOMPLETE | | {fmt(fT, 4)} / {fmt(f4, 4)} | {fmt(pT, 4)} / {fmt(p4, 4)} | | | |")
            for k, v in H["gates"].items():
                glines.append(f"| {N} | {bc} | {mode} | {k} | {fmt(v.get('value'))} | {fmt(v.get('threshold'))} | {_g(v, 'control_value')} | {fmt(v['passed'])} | {v.get('note','')[:200]} |")
    dl = ["", "| N | BC | head | G0 (phase 2) | predicted G0 | W3 arm A | W3 arm C | reading |", "|---|---|---|---|---|---|---|---|"]
    p2 = {}
    for f in p2files:
        r = json.load(open(f))
        for mode, H in r["heads"].items():
            p2[(r["N"], r["bc"], mode)] = (H["G0_passed"], H["predicted_G0"])
    for f in files:
        r = json.load(open(f)); N, bc = r["N"], r["bc"]
        for mode, H in r["heads"].items():
            g0, pred = p2.get((N, bc, mode), (None, None))
            wa = H["gates"].get("W3-A", {}).get("passed"); wcc = H["gates"].get("W3-C", {}).get("passed")
            if bc == "ref":
                if g0 is None or wcc is None: reading = "incomplete"
                elif g0 and wcc: reading = "G0 pass + reflective arm C pass: universal structural failure REFUTED"
                elif (not g0) and (not wcc): reading = "G0 fail + reflective fail: consistent with the manifold-quality diagnosis (not alone decisive)"
                elif g0 and not wcc: reading = "G0 pass + reflective arm C fail: INCONCLUSIVE (structural not refuted on this head; check W4/W6/D2)"
                else: reading = "G0 fail + reflective pass: INCONCLUSIVE (G0 is a proxy)"
            else:
                reading = "absorbing: dissipative comparator, not used for the causal verdict"
            dl.append(f"| {N} | {bc} | {mode} | {fmt(g0) if g0 is not None else '—'} | {fmt(pred) if pred is not None else '—'} | {fmt(wa) if wa is not None else '—'} | {fmt(wcc) if wcc is not None else '—'} | {reading} |")
    txt = "\n".join(lines + glines + dl) + "\n"
    open(os.path.join(OUT, "wav2d-phase3-gates.md"), "w").write(txt)
    return txt


if __name__ == "__main__":
    t1 = phase1_table()
    print(t1)
    splice(os.path.join(HERE, "WAVE2D-NOTES.md"), "phase1-table", t1)
    t2 = phase2_table(); print(t2); splice(os.path.join(HERE, "WAVE2D-NOTES.md"), "phase2-table", t2)
    t3 = phase3_table(); print(t3); splice(os.path.join(HERE, "WAVE2D-NOTES.md"), "phase3-table", t3)
