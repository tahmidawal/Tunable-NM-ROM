"""Every table of this lane, generated from the pulled summary.json / verify.json files.

Usage: write_tables.py <attempt_job> [<attempt_job> ...]   -> results/<attempt_job>.md
                                                             + results/ladder.md (all)
No number is typed by hand; every one below is read from a job's JSON.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
RESULTS = HERE / "results"
PARENT = HERE.parent / "ns3d-shift" / "runs"


def pct(x, nd=3):
    return "—" if x is None else f"{100 * x:.{nd}f}%"


def ms(x):
    return "—" if x is None else f"{x:.3f}"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(job):
    d = RUNS / job / "output"
    s = json.loads((d / "summary.json").read_text())
    v = json.loads((d / "verify.json").read_text()) if (d / "verify.json").exists() else None
    return s, v, d


def fom_rows(s):
    """Uniform list of FOM settings: family, label, worst, median, stable, ms."""
    t = s["timing"]
    rows = []
    for key, e in s["cnab2"].items():
        st = e["stats"]
        rows.append(dict(family="CNAB2", key=f"CNAB2_s{e['steps']}",
                         label=f"CNAB2 {e['steps']} steps (dt={e['dt']:.5g})",
                         worst=None if st is None else st["evolved_worst"],
                         median=None if st is None else st["evolved_median"],
                         stable=not e["unstable"], reference=e.get("reference_itself", False),
                         ms=t["fast"].get(f"CNAB2_s{e['steps']}", {}).get("median_ms")))
    for key, e in s["fd_cg"].items():
        st = e["stats"]
        rows.append(dict(family="FD-CG", key=key,
                         label=f"FD-CG {e['mesh']}^3, {e['steps']} steps, rtol {e['rtol']:g}",
                         worst=None if st is None else st["evolved_worst"],
                         median=None if st is None else st["evolved_median"],
                         stable=not e["unstable"], reference=False,
                         ms=t["slow"].get(key, {}).get("median_ms"),
                         cg_p=e.get("cg_pressure_mean_iters"),
                         cg_v=e.get("cg_viscous_max_iters"),
                         cg_ratio=e.get("cg_worst_final_ratio"),
                         cg_hit=e.get("cg_hit_maxiter")))
    return rows


def comparator(foms, family, worst):
    """Fastest stable setting of `family` whose evolved worst <= worst."""
    ok = [f for f in foms if f["family"] == family and f["stable"] and f["worst"] is not None
          and f["ms"] is not None and f["worst"] <= worst and not f["reference"]]
    if not ok:
        ok = [f for f in foms if f["family"] == family and f["stable"] and f["worst"] is not None
              and f["ms"] is not None and f["worst"] <= worst]
    return min(ok, key=lambda f: f["ms"]) if ok else None


def rom_rows(s):
    t = s["timing"]["fast"]
    gate = s["timing"].get("neighbour_gate", {})
    rows = []
    for key, e in s.get("span", {}).items():
        if not e["finite"]:
            continue
        rows.append(dict(kind="span", key=key, label=f"bank span R'={e['rank']}",
                         unknowns=e["rank"] + 3, dt=e["dt"], iters=e["iters"],
                         worst=e["stats"]["evolved_worst"], median=e["stats"]["evolved_median"],
                         over=e["stats"]["cases_evolved_over_target"], cases=e["stats"]["cases"],
                         ms=t[f"query_{key}"]["median_ms"], rank=e["rank"]))
    for key, e in s["frontier"].items():
        if not e["finite"]:
            continue
        rows.append(dict(kind="head", key=key, label=f"head k={e['k']}",
                         unknowns=e["k"] + e["q"] + 3, dt=e["dt"], iters=e["iters"],
                         worst=e["stats"]["evolved_worst"], median=e["stats"]["evolved_median"],
                         over=e["stats"]["cases_evolved_over_target"], cases=e["stats"]["cases"],
                         ms=t[f"query_{key}"]["median_ms"], k=e["k"]))
    for key, e in s["linear"].items():
        rows.append(dict(kind="parent", key=key, label="parent linear arm (unrotated R=64)",
                         unknowns=s["config"]["rank"] + 3, dt=e["dt"], iters=e["iters"],
                         worst=e["stats"]["evolved_worst"], median=e["stats"]["evolved_median"],
                         over=e["stats"]["cases_evolved_over_target"], cases=e["stats"]["cases"],
                         ms=t[f"query_{key}"]["median_ms"]))
    for r in rows:
        g = gate.get(f"query_{r['key']}", {})
        r["ms_heavy"] = g.get("after_heavy_ms")
    return rows


def speed_cells(row, foms, best_worst):
    def ratio(c, fam):
        text = f"{c['ms'] / row['ms']:.2f}x"
        if fam == "FD-CG" and row.get("ms_heavy"):
            text += f" / {c['ms'] / row['ms_heavy']:.2f}x"
        return text
    cells = []
    for fam in ("CNAB2", "FD-CG"):
        c = comparator(foms, fam, row["worst"])
        if c is None:
            cells += ["none as accurate", "—"]
        else:
            cells += [f"{c['label']} ({pct(c['worst'])}, {ms(c['ms'])} ms)",
                      f"**{ratio(c, fam)}**"]
    for fam in ("CNAB2", "FD-CG"):
        c = comparator(foms, fam, best_worst)
        cells.append("—" if c is None else ratio(c, fam))
    return cells


def mesh_section(job):
    s, v, d = load(job)
    cfg = s["config"]
    n = cfg["n"]
    ldt, lit = cfg["ladder_dt"], cfg["ladder_iters"]
    foms = fom_rows(s)
    roms = rom_rows(s)
    ksel = s["k_selected"]
    ladder = [r for r in roms if r["dt"] == ldt and r["iters"] == lit
              and (r["kind"] == "span" or (r["kind"] == "head"))]
    ladder.sort(key=lambda r: (r["kind"] != "span", -(r.get("rank") or 0), r.get("k") or 0))
    main_rows = [r for r in ladder if r["kind"] == "span" or r.get("k") == ksel]
    best_worst = min(r["worst"] for r in main_rows)
    best_label = min(main_rows, key=lambda r: r["worst"])["label"]
    L = []
    L.append(f"## {n}^3 — job {s['job_id']}\n")
    L.append(f"GPU `{s['gpu']}`, commit `{s['source_commit']}`, backend {s['device']}, "
             f"elapsed {s['elapsed_seconds']:.0f} s. Development seed {cfg['dev_seed']}, "
             f"{cfg['dev_cases']} cases. Selected head k = **{ksel}** (rule: smallest "
             f"development worst at the ladder setting; within 1 %, the smaller k).\n")
    L.append(f"### Tunability ladder (dt={ldt}, {lit} sweeps), one frozen bank + one frozen head\n")
    L.append("Speedup = comparator median / arm median, both from this job's timing block. "
             "Comparator = fastest tested stable setting of that FOM family whose development "
             "evolved worst is no larger than the arm's. The last two columns divide by the "
             f"fastest setting at least as accurate as the most accurate ladder arm "
             f"({best_label}, {pct(best_worst)}). FD-CG cells give two ratios: against the "
             "arm's fast-block median / against its median measured immediately after the "
             "heaviest FD-CG arm (conservative; timing protocol v2).\n")
    L.append("| arm | unknowns | evolved worst | evolved median | over 5 % | GPU ms | "
             "CNAB2 comparator | vs CNAB2 | FD-CG comparator | vs FD-CG | "
             "vs CNAB2 @ best arm | vs FD-CG @ best arm |")
    L.append("|---|---:|---:|---:|---:|---:|---|---:|---|---:|---:|---:|")
    for r in ladder:
        name = r["label"] + (" (selected)" if r["kind"] == "head" and r["k"] == ksel else "")
        L.append("| " + " | ".join([name, str(r["unknowns"]), pct(r["worst"]), pct(r["median"]),
                                    f"{r['over']}/{r['cases']}", ms(r["ms"])]
                                   + speed_cells(r, foms, best_worst)) + " |")
    # monotonicity on R'
    spans = sorted([r for r in ladder if r["kind"] == "span"], key=lambda r: r["rank"])
    err_mono = all(a["worst"] >= b["worst"] for a, b in zip(spans, spans[1:]))
    cost_mono = all(a["ms"] <= b["ms"] for a, b in zip(spans, spans[1:]))
    L.append("")
    L.append(f"R' ladder monotone in error (worst non-increasing as R' grows): **{err_mono}**; "
             f"in cost (GPU ms non-decreasing as R' grows): **{cost_mono}**.\n")
    # head vs span at matched unknowns
    L.append("Head vs bank span at equal latent size (ladder setting):\n")
    L.append("| size | head evolved worst | head ms | span evolved worst | span ms |")
    L.append("|---:|---:|---:|---:|---:|")
    for r in [x for x in ladder if x["kind"] == "head"]:
        m = [x for x in spans if x["rank"] == r["k"]]
        if m:
            L.append(f"| {r['k']} | {pct(r['worst'])} | {ms(r['ms'])} | {pct(m[0]['worst'])} | "
                     f"{ms(m[0]['ms'])} |")
    L.append("")
    # floors
    L.append("### Representation floors (oracle shift, development)\n")
    L.append("| trial set | evolved worst | evolved median |")
    L.append("|---|---:|---:|")
    bf = s["bank_floor"]["stats"]
    L.append(f"| full bank R={cfg['rank']} (POD order) | {pct(bf['evolved_worst'])} | "
             f"{pct(bf['evolved_median'])} |")
    for key, st in sorted(s["span_floors"].items(), key=lambda kv: -int(kv[0])):
        L.append(f"| span R'={key} (importance order) | {pct(st['evolved_worst'])} | "
                 f"{pct(st['evolved_median'])} |")
    for k, fl in sorted(s["head_floors"].items(), key=lambda kv: int(kv[0])):
        st = fl["0"]
        L.append(f"| head k={k} (query encoder) | {pct(st['evolved_worst'])} | "
                 f"{pct(st['evolved_median'])} |")
    L.append("")
    if "heads" in s:
        L.append("### Head training (training seed only)\n")
        hd = s["head_data"]
        L.append(f"{hd['cases']} trajectories × {hd['frames_per_case']} frames; "
                 f"{hd['states_train']} training / {hd['states_val']} validation states.\n")
        L.append("| k | seconds | final weighted fit | train median | train worst | "
                 "val median | val worst |")
        L.append("|---:|---:|---:|---:|---:|---:|---:|")
        for k, h in sorted(s["heads"].items(), key=lambda kv: int(kv[0])):
            f = h["fit"]
            L.append(f"| {k} | {h['train_seconds']:.0f} | {h['curve'][-1][1]:.3e} | "
                     f"{pct(f['train_q0']['median'])} | {pct(f['train_q0']['worst'])} | "
                     f"{pct(f['val_q0']['median'])} | {pct(f['val_q0']['worst'])} |")
        L.append("")
    # full frontier
    L.append("### Full frontier (every dt × sweeps)\n")
    L.append("| arm | dt | sweeps | evolved worst | over 5 % | GPU ms | vs CNAB2 | vs FD-CG |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(roms, key=lambda r: (r["kind"], r["label"], r["dt"], r["iters"])):
        cells = []
        for fam in ("CNAB2", "FD-CG"):
            c = comparator(foms, fam, r["worst"])
            cells.append("—" if c is None else f"{c['ms'] / r['ms']:.2f}x")
        L.append(f"| {r['label']} | {r['dt']} | {r['iters']} | {pct(r['worst'])} | "
                 f"{r['over']}/{r['cases']} | {ms(r['ms'])} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("### FOM grids\n")
    L.append("| setting | evolved worst | evolved median | stable | GPU ms | CG pressure iters (mean) | "
             "CG viscous iters (max) | worst CG residual ratio |")
    L.append("|---|---:|---:|---|---:|---:|---:|---:|")
    for f in foms:
        extra = ([f"{f['cg_p']:.1f}", str(f["cg_v"]), f"{f['cg_ratio']:.1e}"]
                 if f["family"] == "FD-CG" and f.get("cg_p") is not None else ["—", "—", "—"])
        L.append(f"| {f['label']}{' (the reference itself)' if f['reference'] else ''} | "
                 f"{pct(f['worst'])} | {pct(f['median'])} | {'yes' if f['stable'] else '**no**'} | "
                 f"{ms(f['ms'])} | " + " | ".join(extra) + " |")
    L.append("")
    # controls and gates
    L.append("### Controls and gates\n")
    t = s["timing"]
    fz = s["control_frame_zero"]["stats"]
    L.append(f"- Frame frozen at delta=0 (head k={ksel}): evolved worst {pct(fz['evolved_worst'])}, "
             f"{fz['cases_evolved_over_target']}/{fz['cases']} over 5 % (must fail).")
    hp = s["head_parity_vs_lm"]
    L.append(f"- Head fixed-sweep driver vs generic LM ({hp['cases']} cases, q={hp['q']}): "
             + ", ".join(f"{k} sweeps {v:.2e}" for k, v in hp["parity"].items())
             + f"; LM exit reasons [budget, tol, tiny, rejected, stationary] = {hp['lm_reasons']}.")
    rot = s["rotation"]
    L.append(f"- Rotated operators vs a fresh build of G V: {rot['rotated_vs_rebuilt_operators']:.2e}; "
             f"V orthogonality {rot['orthogonality']:.1e}.")
    oc = s["operator_checks"]
    L.append(f"- Operator checks (subset of dense tests): A {oc['A_relative']:.1e}, D {oc['derivative_relative']:.1e}, "
             f"T {oc['tensor_relative']:.1e}, diffusion {oc['diffusion_relative']:.1e}, shift sign {oc['shift_sign_relative']:.1e}.")
    if t.get("protocol") == "v2":
        L.append(f"- Timing neighbour gate (v2; heavy neighbour `{t['heavy_neighbour']}`, "
                 f"cool-down {t['cooldown_seconds']} s, bound {cfg['neighbour_gate_ratio']}×): "
                 f"fast block vs solo **{t['neighbour_gate_fast_block_passed']}**, after "
                 f"neighbour + cool-down vs fast block **{t['neighbour_gate_after_cooldown_passed']}**.")
        L.append("")
        L.append("| ladder arm | fast-block ms | after heavy (no cool-down) | ratio | "
                 "after heavy + cool-down | ratio |")
        L.append("|---|---:|---:|---:|---:|---:|")
        for k, g in t["neighbour_gate"].items():
            if "after_heavy_ms" in g:
                L.append(f"| {k.replace('query_', '')} | {ms(g['fast_block_ms'])} | "
                         f"{ms(g['after_heavy_ms'])} | {g['after_heavy_ratio']:.3f} | "
                         f"{ms(g['after_heavy_cooldown_ms'])} | {g['after_cooldown_ratio']:.3f} |")
        L.append("")
        L.append("GPU state at phase boundaries: " + "; ".join(
            f"{x['label']}: {x['smi']}" for x in t.get("gpu_states", [])) + "\n")
    else:
        L.append(f"- Timing neighbour gate (v1; sentinel median after the job's longest arm, "
                 f"`{t['long_neighbour']}`, and inside the big block, each ≤ "
                 f"{cfg['neighbour_gate_ratio']}× its solo median): **{t['neighbour_gate_passed']}** — "
                 + "; ".join(f"{k.replace('query_', '')}: after-long {g['after_long_ratio']:.3f}, "
                             f"in-block {g['fast_block_ratio']:.3f}"
                             for k, g in t["neighbour_gate"].items()) + ".")
    L.append(f"- Errors of the timed outputs vs the accuracy pass: {t['timed_output_error_agreement']:.1e}.")
    if v:
        L.append(f"- Independent NumPy audit (separate process, saved fields): worst gap float64 "
                 f"{v['worst_gap_float64']:.1e}, float32 {v['worst_gap_float32']:.1e} over "
                 f"{len(v['settings'])} field sets; perturbed-copy control gap {v['control_gap']:.1e}, "
                 f"rejected = {v['control_rejected']}.")
    # parent reproduction
    parent = PARENT / f"ladder{n}" / "summary.json"
    if parent.exists():
        ps = json.loads(parent.read_text())
        gaps = []
        for key, e in s["linear"].items():
            pk = f"r64_dt{e['dt']}_it{e['iters']}"
            if pk in ps.get("frontier", {}):
                gaps.append((key, e["stats"]["evolved_worst"],
                             ps["frontier"][pk]["stats"]["evolved_worst"]))
        pf = ps["floors"]["ranks"]["64"]["stats"]["evolved_worst"]
        worst_gap = max(abs(a - b) for _, a, b in gaps) if gaps else None
        L.append(f"- Reproduction of the parent lane (`ns3d-shift/runs/ladder{n}`): bank floor "
                 f"{pct(bf['evolved_worst'], 4)} vs {pct(pf, 4)}; linear-arm evolved worst over "
                 f"{len(gaps)} shared settings, largest absolute gap "
                 f"{'—' if worst_gap is None else f'{worst_gap:.1e}'}.")
    L.append("")
    L.append(f"Summary JSON: `{(d / 'summary.json').relative_to(HERE.parents[1])}` "
             f"sha256 `{sha(d / 'summary.json')}`.\n")
    return "\n".join(L), s


def headline(jobs):
    """Cross-mesh ladder table: one row per arm per mesh, the ladder setting only."""
    L = ["## Headline — the tunability ladder at every mesh\n",
         "Ladder setting (dt and sweeps fixed in DESIGN.md). Worst/median = development "
         "evolved worst / median. Comparator = fastest stable FOM setting of the family no "
         "less accurate than the arm; FD-CG ratios are fast-block / after-heavy "
         "(conservative).\n",
         "| mesh | arm | unknowns | worst | median | GPU ms | CNAB2 comparator | vs CNAB2 | "
         "FD-CG comparator | vs FD-CG | job |",
         "|---:|---|---:|---:|---:|---:|---|---:|---|---:|---|"]
    for job in jobs:
        s, _, _ = load(job)
        cfg = s["config"]
        foms = fom_rows(s)
        rows = [r for r in rom_rows(s) if r["dt"] == cfg["ladder_dt"]
                and r["iters"] == cfg["ladder_iters"]
                and (r["kind"] == "span" or (r["kind"] == "head" and r["k"] == s["k_selected"]))]
        rows.sort(key=lambda r: (r["kind"] != "span", -(r.get("rank") or 0)))
        for r in rows:
            cells = []
            for fam in ("CNAB2", "FD-CG"):
                c = comparator(foms, fam, r["worst"])
                if c is None:
                    cells += ["none as accurate", "—"]
                else:
                    txt = f"{c['ms'] / r['ms']:.2f}x"
                    if fam == "FD-CG" and r.get("ms_heavy"):
                        txt += f" / {c['ms'] / r['ms_heavy']:.2f}x"
                    cells += [f"{c['label']} ({pct(c['worst'])}, {ms(c['ms'])} ms)", txt]
            L.append(f"| {cfg['n']}^3 | {r['label']} | {r['unknowns']} | {pct(r['worst'])} | "
                     f"{pct(r['median'])} | {ms(r['ms'])} | " + " | ".join(cells)
                     + f" | {s['job_id']} |")
    return "\n".join(L) + "\n"


def main():
    jobs = sys.argv[1:]
    RESULTS.mkdir(exist_ok=True)
    parts = []
    for job in jobs:
        text, _ = mesh_section(job)
        (RESULTS / f"{job}.md").write_text(f"# {job}\n\nGenerated by `write_tables.py` from "
                                           f"`runs/{job}/output/summary.json`.\n\n" + text)
        parts.append(text)
    if len(jobs) > 1:
        (RESULTS / "ladder.md").write_text(
            "# NS3D head and bank span inside the co-moving frame — development ladder\n\n"
            "Generated by `write_tables.py` from each job's `summary.json`; nothing typed "
            "by hand. Development cohort only.\n\n" + headline(jobs) + "\n" + "\n".join(parts))
    print("wrote", [f"{j}.md" for j in jobs])


if __name__ == "__main__":
    main()
