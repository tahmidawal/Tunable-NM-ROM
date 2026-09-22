"""Generate results/<name>.md from a run's summary.json. No number is typed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def pct(x):
    return f"{100.0 * float(x):.3f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True, help="runs/<name>/output")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    s = json.loads((args.run / "summary.json").read_text())
    cfg = s["config"]
    verify = None
    if (args.run / "verify.json").exists():
        verify = json.loads((args.run / "verify.json").read_text())
    L = []
    L.append(f"# {cfg['name']} development pilot: the frame as an online unknown\n")
    L.append(f"Generated from `{args.run}/summary.json` by `write_pilot_table.py`. "
             f"Job {s.get('job_id')}, commit `{s.get('source_commit')}`, "
             f"{s.get('gpu')}, device {s.get('device')}. "
             f"Development seed {cfg['dev_seed']} only; the final cohort was not opened. "
             f"Evolved worst is the worst case of the worst evolved time.\n")

    L.append("## Floors\n")
    L.append("| rank | A0 fixed uncentred bank | over 5% | A1 centered bank, oracle shift | over 5% |")
    L.append("|---:|---:|---:|---:|---:|")
    a0, a1 = s["A0_fixed_bank_floor"], s["A1_centered_oracle_floor"]["ranks"]
    cases = None
    for rank in sorted({int(k) for k in a0} | {int(k) for k in a1}):
        left = a0.get(str(rank))
        right = a1.get(str(rank))
        cases = (left or right["stats"])["cases"]
        L.append("| {} | {} | {} | {} | {} |".format(
            rank,
            pct(left["evolved_worst"]) if left else "-",
            f"{left['cases_evolved_over_target']}/{left['cases']}" if left else "-",
            pct(right["stats"]["evolved_worst"]) if right else "-",
            f"{right['stats']['cases_evolved_over_target']}/{right['stats']['cases']}"
            if right else "-"))
    L.append(f"\nCentered POD available rank: {s['A1_centered_oracle_floor']['available_rank']}. "
             f"Worst centroid travel over the horizon: "
             f"{a1[str(cfg['base_rank'])]['travel']['centroid_travel_worst']:.6f} of the box.\n")

    L.append("## Harness checks\n")
    oc = s["operator_checks"]
    L.append("| check | value | meaning |")
    L.append("|---|---:|---|")
    L.append(f"| `delta == 0` vs `ns3d_rom.make_run` | {s['zero_delta_parity_vs_ns3d_rom']:.3e} "
             "| the co-moving residual reduces exactly to the existing weak ROM |")
    L.append(f"| complete-query equivariance | {s['query_equivariance']:.3e} "
             "| the solved query on a shifted input is the shift of the solved query |")
    L.append(f"| advection tensor vs full grid | {oc['tensor_relative']:.3e} | |")
    L.append(f"| derivative projection vs full grid | {oc['derivative_relative']:.3e} | |")
    L.append(f"| diffusion identity | {oc['diffusion_relative']:.3e} | "
             "`Phi^T Lap G == -diag(lam) A` |")
    L.append(f"| finite-difference shift sign | {oc['shift_sign_relative']:.3e} | "
             "independent sign check on `-D_d a` |")
    L.append(f"| `S_d` skew symmetry | {oc['S_skew']:.3e} | the phase row needs it |")
    L.append(f"| bank orthonormality | {oc['bank_orthonormality']:.3e} | |")
    L.append(f"| test orthogonality | {oc['test_orthogonality']:.3e} | |")
    L.append("")

    L.append("## Identifiability of the frame\n")
    L.append("| quantity | no gauge | with gauge |")
    L.append("|---|---:|---:|")
    ng, wg = s["conditioning"]["no_gauge"], s["conditioning"]["gauge"]
    rows = [("smallest / largest singular value of the full Jacobian",
             "smallest_over_largest", "{:.3e}"),
            ("condition number of the full Jacobian", "condition", "{:.3e}"),
            ("largest singular value of the coefficient block",
             "coefficient_block_largest", "{:.3e}"),
            ("**deflated** smallest sv of `(I - Ja Ja+) Jdelta`, over largest sv of `Ja`",
             "deflated_smallest_over_Ja_largest", "{:.3e}"),
            ("analytic vs AD delta column", "analytic_delta_column_relative", "{:.3e}")]
    for label, key, fmt in rows:
        L.append(f"| {label} | {fmt.format(ng[key])} | {fmt.format(wg[key])} |")
    tt = s["translation_tangents"]
    L.append(f"\nTranslation tangents `d_d(G a)` captured inside `span(G)`: "
             f"{', '.join(f'{x:.4f}' for x in tt['captured_fraction'])} "
             f"(rank {tt['tangent_rank_in_span']} of 3). The pre-registered "
             "identifiability threshold was a deflated ratio above 1e-6.\n")

    L.append("## Solved arms, development cohort\n")
    L.append("| arm | rank | M | dt | gauge | budget | evolved worst | evolved median "
             "| over 5% | median LM iters | on budget | centroid gap worst |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, entry in s["arms"].items():
        if "failed" in entry:
            L.append(f"| {label} | | | | | | FAILED: {entry['failed']} | | | | | |")
            continue
        sp, st, sv = entry["spec"], entry["stats"], entry["solver"]
        L.append("| {} | {} | {} | {} | {} | {} | {} | {} | {}/{} | {} | {:.1%} | {:.5f} |".format(
            label, sp["rank"], sp["modes"], sp["dt"], sp["gauge"], sp["budget"],
            pct(st["evolved_worst"]), pct(st["evolved_median"]),
            st["cases_evolved_over_target"], st["cases"],
            sv["median_iterations"], sv["fraction_on_budget"],
            entry["centroid_gap"]["worst"]))
    c = s["C_tracker"]
    L.append("| C tracker | {} | - | {} | - | - | {} | {} | {}/{} | - | - | - |".format(
        c["rank"], c["dt"], pct(c["stats"]["evolved_worst"]),
        pct(c["stats"]["evolved_median"]), c["stats"]["cases_evolved_over_target"],
        c["stats"]["cases"]))
    for dtv, st in s["E_cnab2"].items():
        L.append("| CNAB2 dt={} | - | - | {} | - | - | {} | {} | {}/{} | - | - | - |".format(
            dtv, dtv, pct(st["evolved_worst"]), pct(st["evolved_median"]),
            st["cases_evolved_over_target"], st["cases"]))
    L.append("")

    L.append("## Complete-query cost, paired\n")
    L.append(f"{s['timing']['note']}. Case {s['timing']['case']}, "
             f"{cfg['timing_repetitions']} retained repetitions.\n")
    L.append("| arm | median ms | min ms | max ms | worst evolved error of the timed output |")
    L.append("|---|---:|---:|---:|---:|")
    errs = s["timing"]["errors_from_timed_outputs"]
    for name, t in s["timing"]["arms"].items():
        e = errs.get(name)
        L.append("| {} | {:.3f} | {:.3f} | {:.3f} | {} |".format(
            name, t["median_ms"], t["min_ms"], t["max_ms"],
            pct(max(e[1:])) if e else "-"))
    # comparator rule: fastest CNAB2 whose worst evolved error <= the ROM's
    rom_names = [k for k in s["timing"]["arms"] if k.startswith("B")]
    L.append("")
    for rom in rom_names:
        rw = max(errs[rom][1:])
        eligible = [(s["timing"]["arms"][k]["median_ms"], k) for k in s["timing"]["arms"]
                    if k.startswith("CNAB2") and max(errs[k][1:]) <= rw]
        if not eligible:
            L.append(f"- **{rom}**: no tested CNAB2 step is as accurate; comparator undefined "
                     "at this ladder.")
            continue
        ms, name = min(eligible)
        L.append(f"- **{rom}**: comparator {name} at {ms:.3f} ms, paired speedup "
                 f"{ms / s['timing']['arms'][rom]['median_ms']:.3f}x "
                 f"(worst evolved {pct(rw)} vs {pct(max(errs[name][1:]))}).")
    L.append("")

    if verify:
        L.append("## Independent verification\n")
        L.append(f"`verify_pilot.py` recomputed every saved arm from the fields in NumPy "
                 f"with a different reduction. Worst disagreement "
                 f"{verify['worst_gap']:.3e}.\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
