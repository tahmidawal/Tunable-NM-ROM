"""Generate the resolution-ladder tables from one or more ladder summaries.

Usage: write_ladder_table.py --run <dir> [--run <dir> ...] --out results/ladder.md
Every number comes from a summary.json read here; nothing is typed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def pct(x):
    return f"{100.0 * float(x):.3f}%"


def ms(x):
    return f"{float(x):.3f}"


def load(run):
    s = json.loads((run / "summary.json").read_text())
    v = json.loads((run / "verify.json").read_text()) if (run / "verify.json").exists() else None
    return s, v


def comparators(summary, timing):
    """(matched-accuracy chooser, stability-limited row). Unstable settings never win."""
    fom = summary["cnab2"]
    stable = {d: v for d, v in fom.items() if not v["unstable"] and v["stats"]}

    def matched(worst):
        ok = [(timing[f"CNAB2_dt{d}"]["median_ms"], d) for d, v in stable.items()
              if v["stats"]["evolved_worst"] <= worst]
        return min(ok) if ok else (None, None)

    target = float(summary["config"]["target_relative"])
    ok = [(timing[f"CNAB2_dt{d}"]["median_ms"], d) for d, v in stable.items()
          if v["stats"]["evolved_worst"] <= target]
    return matched, (min(ok) if ok else (None, None))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    runs = [load(r) for r in args.run]
    runs.sort(key=lambda sv: sv[0]["config"]["n"])
    L = []
    W = L.append
    W("# Resolution ladder: does the shift ROM stay accurate and get fast?\n")
    W("Generated from each job's `summary.json` by `write_ladder_table.py`. Accuracy is the "
      "full development cohort; timing is one case in a single interleaved block per job, "
      "complete queries. **Comparator rule:** the fastest tested *stable* CNAB2 setting whose "
      "evolved worst is no larger than the ROM's. A CNAB2 setting counts as unstable when its "
      "evolved worst exceeds 100 %; unstable settings are never comparators.\n")

    W("## Headline per mesh\n")
    W("| mesh | best setting meeting 5 % | evolved worst | over 5 % | ROM ms | comparator "
      "| stable? | comparator error | comparator ms | paired speedup | stability-limited FOM "
      "| that speedup |")
    W("|---:|---|---:|---:|---:|---|---|---:|---:|---:|---|---:|")
    for s, _ in runs:
        n = s["config"]["n"]
        t = s["timing"]["arms"]
        matched, (slim_ms, slim_dt) = comparators(s, t)
        # gate membership comes from the MEASURED parity at this mesh, not a fixed
        # sweep count: 3 sweeps met 1e-8 at 32^3 and missed it at 64^3 and 96^3.
        parity_by_sweep = {v["iters"]: v["parity_vs_reference_lm"]
                           for v in s["frontier"].values()
                           if "parity_vs_reference_lm" in v}
        ok_sweeps = {i for i, pv in parity_by_sweep.items() if pv <= 1e-8}
        gated = {k: v for k, v in s["frontier"].items()
                 if v["stats"]["cases_evolved_over_target"] == 0
                 and v["iters"] in ok_sweeps}
        if not gated:
            W(f"| {n}^3 | *no sweep count met the 1e-8 parity gate at this mesh "
              f"(best {min(parity_by_sweep.values()):.2e})* | | | | | | | | | | |")
            continue
        best = max(gated, key=lambda k: (matched(gated[k]["stats"]["evolved_worst"])[0] or 0)
                   / t[f"query_{k}"]["median_ms"])
        v = gated[best]
        q = t[f"query_{best}"]["median_ms"]
        cms, dname = matched(v["stats"]["evolved_worst"])
        W("| {}^3 | {} | {} | {}/{} | {} | CNAB2 dt={} | stable | {} | {} | **{:.2f}x** | {} | {} |".format(
            n, best.replace("_", " "), pct(v["stats"]["evolved_worst"]),
            v["stats"]["cases_evolved_over_target"], v["stats"]["cases"], ms(q),
            dname, pct(s["cnab2"][dname]["stats"]["evolved_worst"]), ms(cms), cms / q,
            f"CNAB2 dt={slim_dt}" if slim_dt else "none",
            f"{slim_ms / q:.2f}x" if slim_dt else "-"))
        W(f"| | *parity at {n}^3: "
          + ", ".join(f"{i} sweeps {pv:.2e}" for i, pv in sorted(parity_by_sweep.items()))
          + "* | | | | | | | | | | |")
    W("")
    W("The **stability-limited FOM** is the cheapest stable CNAB2 that itself meets the 5 % "
      "target, i.e. the cheapest the FOM can honestly be run at that mesh. Where it equals the "
      "matched-accuracy comparator, the speedup is pure throughput; where it is coarser, the "
      "extra margin comes from the reduced model being more accurate than the FOM at that "
      "step, not from taking a step the FOM cannot.\n")

    for s, verify in runs:
        n = s["config"]["n"]
        t = s["timing"]["arms"]
        matched, _ = comparators(s, t)
        W(f"## {n}^3\n")
        W("### Representation floor (is accuracy limited by the bank?)\n")
        W("| rank | oracle-shift floor, evolved worst | over 5 % |")
        W("|---:|---:|---:|")
        for rank, entry in sorted(s["floors"]["ranks"].items(), key=lambda kv: int(kv[0])):
            W("| {} | {} | {}/{} |".format(rank, pct(entry["stats"]["evolved_worst"]),
                                           entry["stats"]["cases_evolved_over_target"],
                                           entry["stats"]["cases"]))
        W("")
        W("### Accuracy-cost frontier\n")
        W("| rank | dt | steps | sweeps | evolved worst | over 5 % | query ms | comparator "
          "| comparator ms | paired speedup |")
        W("|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|")
        rows = []
        for key, v in s["frontier"].items():
            q = t[f"query_{key}"]["median_ms"]
            cms, dname = matched(v["stats"]["evolved_worst"])
            rows.append(((cms / q) if dname else 0.0, key, v, q, cms, dname))
        for ratio, key, v, q, cms, dname in sorted(rows, reverse=True):
            W("| {} | {} | {} | {} | {} | {}/{} | {} | {} | {} | {} |".format(
                v["rank"], v["dt"], v["steps"], v["iters"], pct(v["stats"]["evolved_worst"]),
                v["stats"]["cases_evolved_over_target"], v["stats"]["cases"], ms(q),
                f"CNAB2 dt={dname}" if dname else "none stable is as accurate",
                ms(cms) if dname else "-", f"{ratio:.3f}x" if dname else "-"))
        W("")
        finite = [float(d) for d, v in s["cnab2"].items() if not v["unstable"]]
        blew = [float(d) for d, v in s["cnab2"].items() if v["unstable"]]
        usable = [float(d) for d, v in s["cnab2"].items()
                  if not v["unstable"] and v["stats"]
                  and v["stats"]["evolved_worst"] <= float(s["config"]["target_relative"])]
        if finite:
            rom_dts = [v["dt"] for v in s["frontier"].values()]
            rom_steps = min(v["steps"] for v in s["frontier"].values())
            blew_text = (f"steps at or above {min(blew)} blew up (evolved worst over 100 %)"
                         if blew else "no tested step blew up")
            W(f"**Stability and step size.** For CNAB2 at this mesh, {blew_text}; the coarsest "
              f"step that both stays finite and meets the 5 % target is {max(usable)} "
              f"({s['cnab2'][repr(max(usable))]['steps']} steps) -- that is the "
              "stability-limited comparator. The reduced model runs at "
              f"{max(rom_dts)} ({rom_steps} steps), because its step is solved implicitly at "
              "the midpoint instead of advanced explicitly. Where the two comparator columns "
              "differ, the gap between them is the part of the margin that comes from a step "
              "the FOM cannot take rather than from throughput.\n")
        W("### Baselines and the FOM ladder\n")
        W("| arm | evolved worst | over 5 % | median ms | note |")
        W("|---|---:|---:|---:|---|")
        ref = s["reference_lm"]
        W("| reference LM arm (rank {}, dt {}) | {} | {}/{} | {} | the pre-fix solver, "
          "same job |".format(ref["rank"], ref["dt"], pct(ref["stats"]["evolved_worst"]),
                              ref["stats"]["cases_evolved_over_target"],
                              ref["stats"]["cases"], ms(t["reference_lm"]["median_ms"])))
        if "tracker" in s:
            tr = s["tracker"]
            W("| centroid tracker (rank {}, dt {}) | {} | {}/{} | {} | the arm to beat |".format(
                tr["rank"], tr["dt"], pct(tr["stats"]["evolved_worst"]),
                tr["stats"]["cases_evolved_over_target"], tr["stats"]["cases"],
                ms(t["tracker"]["median_ms"])))
        for dtv in sorted(s["cnab2"], key=float):
            entry = s["cnab2"][dtv]
            st = entry["stats"]
            note = "**unstable**" if entry["unstable"] else ""
            if float(dtv) == float(s["config"]["dt_truth"]):
                note = (note + " this is the reference itself").strip()
            W("| CNAB2 dt={} ({} steps) | {} | {}/{} | {} | {} |".format(
                dtv, entry["steps"], pct(st["evolved_worst"]) if st else "nonfinite",
                st["cases_evolved_over_target"] if st else "-", st["cases"] if st else "-",
                ms(t[f"CNAB2_dt{dtv}"]["median_ms"]), note))
        W("")
        W("### Cost breakdown\n")
        pieces = {k: v["median_ms"] for k, v in t.items() if k.startswith("piece_")}
        W("| piece | median ms |")
        W("|---|---:|")
        for name, value in sorted(pieces.items()):
            W(f"| {name.replace('piece_', '').replace('_', ' ')} | {ms(value)} |")
        outs = [(k, v) for k, v in t.items() if k.startswith("one_output_")]
        for name, value in sorted(outs):
            key = name.replace("one_output_", "")
            if f"query_{key}" in t:
                W("| four extra output reconstructions ({}) | {} |".format(
                    key, ms(t[f"query_{key}"]["median_ms"] - value["median_ms"])))
        rank0 = sorted(s["floors"]["ranks"], key=int)[0]
        init = pieces.get(f"piece_initial_r{rank0}")
        one = pieces.get(f"piece_output_r{rank0}")
        if init and one:
            contract = init + 5 * one
            matched_any, (slim_ms, slim_dt) = comparators(s, t)
            W("")
            W(f"The six-output contract plus the initial projection is {ms(contract)} ms at "
              f"rank {rank0}. A solve of zero cost would therefore cap the speedup at "
              f"**{(slim_ms / contract):.2f}x** against the stability-limited FOM "
              f"(CNAB2 dt={slim_dt}, {ms(slim_ms)} ms) at this mesh.\n")
        if verify:
            W(f"Independent NumPy recomputation: worst disagreement {verify['worst_gap']:.3e}; "
              f"driver-fix parity against the LM arm "
              f"{min(verify['parity_vs_reference_lm']):.3e}.\n")
        W(f"GPU: {s['gpu']}. Job {s['job_id']}, commit `{s['source_commit']}`. "
          f"Operator checks and the GPU-centering cross-check are in the summary.\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
