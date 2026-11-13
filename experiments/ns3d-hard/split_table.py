"""Tables for the flat-timing lane, generated from the job JSONs (no hand-typed numbers).

  python split_table.py split <split.json> [...]       per-arm, per-mesh stage times + flatness
  python split_table.py rerun <compare.json> [...]     gate status + any-mesh solve-only ratios
"""
import json, sys
from pathlib import Path

ARMS = ["r64_head", "r96_head", "r128_head", "r128_head_fast", "r192_head", "r192_head_fast"]


def split_tables(paths):
    out = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        meshes = [m for m in d["meshes"] if any("ms" in a for a in d["meshes"][m]["arms"].values())]
        out.append(f"### {Path(p).parent.parent.name}: job {d.get('job_id')}, {d.get('gpu', '').split(',')[0]}, "
                   f"host {d.get('host')}, status {d.get('status')}\n")
        if d.get("failures"):
            out.append("Failures: " + "; ".join(d["failures"]) + "\n")
        gates = []
        for m in meshes:
            for g, t in d["meshes"][m].get("timing", {}).items():
                gates.append(f"{m}^3 [{g}]: {'pass' if t['gate_passed'] else 'FAIL'} (attempt {t['attempts']})")
        out.append("Timing gate: " + "; ".join(gates) + "\n")
        out.append("| arm | mesh | M | steps | ENC ms | FIT ms | SOLVE ms | DEC ms | sum ms | FULL ms | DECB ms | "
                   "solver any-mesh (solve-only) ms | solver same-mesh ms | FULL ratio | SOLVE-only ratio |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        flat = []
        for arm in ARMS:
            solves, fulls = {}, {}
            for m in meshes:
                e = d["meshes"][m]["arms"].get(arm)
                if not e or "ms" not in e: continue
                s = e["ms"]; c = e["config"]; ref = d["meshes"][m]["fom_refs"][arm]
                any_tag = "/".join(str(x) for x in ref["any"])
                same = f"{s['FOM_same_solve']:.1f} ({ref['same'][0]} {ref['same'][1]}^3 x{ref['same'][2]})" if "FOM_same_solve" in s else "-"
                out.append(f"| {arm} | {m}{'' if d['meshes'][m]['native_bank'] else '*'} | {c['M']} | {c['steps']} | "
                           f"{s['ENC']:.2f} | {s['FIT']:.2f} | {s['SOLVE']:.2f} | {s['DEC']:.2f} | {s['SUM']:.2f} | {s['FULL']:.2f} | "
                           f"{s.get('DECB', float('nan')):.2f} | {s['FOM_any_solve']:.1f} ({any_tag}) | {same} | "
                           f"{s['FOM_any_solve'] / s['FULL']:.2f} | {s['FOM_any_solve'] / s['SOLVE']:.2f} |")
                solves[m] = s["SOLVE"]; fulls[m] = s["FULL"]
            if solves:
                same_cfg = len({(d["meshes"][m]["arms"][arm]["config"]["M"], d["meshes"][m]["arms"][arm]["config"]["steps"])
                                for m in solves}) == 1
                flat.append(f"| {arm} | {'yes' if same_cfg else 'no'} | {min(solves.values()):.2f} | {max(solves.values()):.2f} | "
                            f"{max(solves.values()) / min(solves.values()):.3f} | "
                            f"{max(fulls.values()) / min(fulls.values()):.2f} | {fulls[meshes[-1]] / fulls[meshes[0]]:.2f} |")
        out.append("\n`*` = mesh above the bank meshes (128^3 bank resampled, 128^3 heads and settings).\n")
        out.append(f"| arm | same (M, steps) at every mesh | SOLVE min ms | SOLVE max ms | SOLVE max/min | FULL max/min | "
                   f"FULL {meshes[-1]}/{meshes[0]} |")
        out.append("|---|---|---|---|---|---|---|")
        out += flat
        chk = []
        for m in meshes:
            for arm in ARMS:
                e = d["meshes"][m]["arms"].get(arm)
                if e:
                    chk.append(f"{arm}@{m}: worst {e['timing_case_evolved_worst'] * 100:.3f}% vs source@{e['source_errors_same_cases_mesh']} "
                               f"{e['source_evolved_worst_same_cases'] * 100:.3f}% (max |diff| {e['max_abs_diff_vs_source']:.1e}; split=full {e['split_vs_full_relative']:.0e})")
        out.append("\nError sanity (8 timing cases): " + "; ".join(chk) + "\n")
        if d.get("asm"):
            for b, v in d["asm"].get("banks", {}).items():
                for n, x in v["meshes"].items():
                    out.append(f"asm {b} @ {n}: checks A {x['operator_checks']['A_relative']:.1e}, tensor {x['operator_checks']['tensor_relative']:.1e}; "
                               f"vs 128 ops: " + ", ".join(f"{q} {y['relative_difference']:.1e}" for q, y in x["vs_128_operators"].items()))
        sol = []
        for m in meshes:
            for k, v in d["meshes"][m].get("solvers", {}).items():
                w = v.get("timing_case_evolved_worst", {})
                if w: sol.append(f"{k} -> {m}: {max(w.values()) * 100:.3f}%")
        if sol:
            out.append("\nReference-solver error on the timing cases (worst): " + "; ".join(sol) + "\n")
    return "\n".join(out)


def rerun_table(paths):
    out = ["| run | job | node/GPU | mesh | gate | attempts | " + " | ".join(ARMS) + " |",
           "|---|---|---|---|---|---|" + "---|" * len(ARMS)]
    for p in paths:
        d = json.loads(Path(p).read_text())
        for m, mr in d["meshes"].items():
            t = mr.get("timing", {})
            cells = []
            for arm in ARMS:
                e = mr["arms"].get(arm, {})
                x = e.get("any_mesh_solve")
                cells.append(f"{x['ratio']:.2f}x ({e['stats']['evolved_worst'] * 100:.2f}%)" if x else "-")
            out.append(f"| {Path(p).parent.name if Path(p).parent.name != 'output' else Path(p).parent.parent.name} | {d.get('job_id')} | "
                       f"{d.get('gpu', '').split(',')[0]} | {m} | {'pass' if t.get('gate_passed') else 'FAIL'} | "
                       f"{mr.get('timing_attempts')} | " + " | ".join(cells) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    kind, paths = sys.argv[1], sys.argv[2:]
    print(split_tables(paths) if kind == "split" else rerun_table(paths))
