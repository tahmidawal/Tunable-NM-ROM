"""Generate results/<name>.md from a cost sweep's summary.json. No number typed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def pct(x):
    return f"{100.0 * float(x):.3f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    s = json.loads((args.run / "summary.json").read_text())
    cfg = s["config"]
    verify = json.loads((args.run / "verify.json").read_text()) \
        if (args.run / "verify.json").exists() else None
    L = [f"# {cfg['name']}: where the co-moving query's time goes\n",
         f"Generated from `{args.run}/summary.json` by `write_cost_table.py`. "
         f"Job {s.get('job_id')}, commit `{s.get('source_commit')}`, {s.get('gpu')}, "
         f"device {s.get('device')}. Rank {cfg['rank']}, gauge {cfg['gauge']}, "
         f"development seed {cfg['dev_seed']} only. Accuracy is the full 16-case "
         f"cohort; timing is case {cfg['timing_case']}, "
         f"{cfg['timing_repetitions']} interleaved repetitions after burn-in.\n"]

    settings = {k: v for k, v in s["settings"].items() if "failed" not in v}
    any_key = next(iter(settings))
    common = settings[any_key]["timing"]

    L.append("## Grid-sized pieces the ROM cannot avoid\n")
    L.append("| piece | median ms |")
    L.append("|---|---:|")
    for name in ("piece_initial_centering_projection", "piece_one_output_reconstruction"):
        L.append(f"| {name.replace('piece_', '').replace('_', ' ')} "
                 f"| {common[name]['median_ms']:.3f} |")
    L.append("")

    L.append("## CNAB2 comparators, same cohort and allocation\n")
    L.append("| dt | evolved worst | evolved median | over 5% | median ms |")
    L.append("|---:|---:|---:|---:|---:|")
    for dtv, st in s["cnab2"].items():
        L.append("| {} | {} | {} | {}/{} | {:.3f} |".format(
            dtv, pct(st["evolved_worst"]), pct(st["evolved_median"]),
            st["cases_evolved_over_target"], st["cases"],
            common[f"CNAB2_dt{dtv}"]["median_ms"]))
    L.append("")

    L.append("## Co-moving ROM: accuracy and complete-query cost\n")
    L.append("| M | dt | steps | evolved worst | over 5% | median LM iters | query ms "
             "| one-output ms | outputs ms | comparator | paired speedup |")
    L.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|")
    for key in sorted(settings, key=lambda k: (settings[k]["modes"], -settings[k]["dt"])):
        v = settings[key]
        t = v["timing"]
        q = t[f"query_{key}"]["median_ms"]
        one = t[f"query_one_output_{key}"]["median_ms"]
        worst = v["stats"]["evolved_worst"]
        eligible = [(s["cnab2"][d], d) for d in s["cnab2"]
                    if s["cnab2"][d]["evolved_worst"] <= worst]
        if eligible:
            _, dname = min(eligible, key=lambda x: common[f"CNAB2_dt{x[1]}"]["median_ms"])
            cms = common[f"CNAB2_dt{dname}"]["median_ms"]
            comparator, speed = f"CNAB2 dt={dname}", f"{cms / q:.3f}x"
        else:
            comparator, speed = "none tested is as accurate", "-"
        L.append("| {} | {} | {} | {} | {}/{} | {} | {:.3f} | {:.3f} | {:.3f} | {} | {} |".format(
            v["modes"], v["dt"], v["steps"], pct(worst),
            v["stats"]["cases_evolved_over_target"], v["stats"]["cases"],
            v["solver"]["median_iterations"], q, one, q - one, comparator, speed))
    for key, v in s["settings"].items():
        if "failed" in v:
            L.append(f"| {v['modes']} | {v['dt']} | | FAILED: {v['failed']} | | | | | | | |")
    L.append("\nThe *outputs* column is the whole query minus the same trajectory asked for "
             "one output instead of five, so it isolates the four extra laboratory-frame "
             "reconstructions. The comparator is the fastest tested CNAB2 step whose "
             "evolved worst is no larger than that row's.\n")

    if verify:
        L.append("## Independent verification\n")
        L.append(f"`verify_cost.py` recomputed every saved setting from its fields in NumPy "
                 f"with a different reduction. Worst disagreement {verify['worst_gap']:.3e}.\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
