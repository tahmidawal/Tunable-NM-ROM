"""Freeze the sealed-draw config from a development ladder summary.

The setting is *read* from the development result, never typed. Among frontier
settings whose sweep count met the 1e-8 parity gate at that mesh:

  1. prefer those meeting the **stretch** accuracy target (evolved worst <= 1 %)
     on every development case; fall back to the 5 % target only if none does;
  2. among those, take the largest paired speedup against the **stability-limited**
     comparator -- the cheapest stable CNAB2 that itself meets the 5 % target,
     which is the conservative of the two comparators.

Accuracy is therefore never traded away for speed: the ladder is searched for a
setting that satisfies both bars at once, and speed only breaks ties inside that
set.

Usage: make_sealed_config.py --run <ladder output dir> --out configs/sealed09.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SEALED_SEED = 202609221
SEALED_CASES = 32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    s = json.loads((args.run / "summary.json").read_text())
    cfg, timing = s["config"], s["timing"]["arms"]
    target = float(cfg["target_relative"])

    stable = {d: v for d, v in s["cnab2"].items() if not v["unstable"] and v["stats"]}
    usable = [(timing[f"CNAB2_dt{d}"]["median_ms"], d) for d, v in stable.items()
              if v["stats"]["evolved_worst"] <= target]
    if not usable:
        raise SystemExit("no stable CNAB2 setting meets the target; nothing to seal against")
    slim_ms, slim_dt = min(usable)

    parity = {v["iters"]: v["parity_vs_reference_lm"] for v in s["frontier"].values()
              if "parity_vs_reference_lm" in v}
    gated = {i for i, p in parity.items() if p <= 1e-8}
    if not gated:
        raise SystemExit(f"no sweep count met the 1e-8 parity gate here: {parity}")

    stretch = float(cfg.get("stretch_relative", 0.01))
    gate_ok = {k: v for k, v in s["frontier"].items()
               if v["stats"]["cases_evolved_over_target"] == 0 and v["iters"] in gated}
    pool = {k: v for k, v in gate_ok.items() if v["stats"]["evolved_worst"] <= stretch}
    rule = f"stretch <= {stretch:.0%}"
    if not pool:
        pool, rule = gate_ok, f"target <= {target:.0%} (stretch not reachable here)"
    if not pool:
        raise SystemExit("no gate-passing setting meets the target on every case")
    best = max(pool, key=lambda k: slim_ms / timing[f"query_{k}"]["median_ms"])
    v = pool[best]

    out = dict(cfg)
    out.update(
        name="sealed09",
        question="one sealed draw of the frozen co-moving shift ROM on a new cohort",
        sealed_cohort=True,
        dev_seed=SEALED_SEED,
        dev_cases=SEALED_CASES,
        disjoint_against=[[cfg["dev_seed"], cfg["dev_cases"]], [202609203, 32],
                          [202609211, 32]],
        ranks=[v["rank"]],
        rom_dt_ladder=[v["dt"]],
        iters_ladder=[v["iters"]],
        reference_rank=v["rank"],
        reference_dt=v["dt"],
        include_tracker=True,
        frozen_from=dict(job=s["job_id"], mesh=cfg["n"], setting=best,
                         development_evolved_worst=v["stats"]["evolved_worst"],
                         parity=parity[v["iters"]],
                         development_query_ms=timing[f"query_{best}"]["median_ms"],
                         stability_limited_comparator=slim_dt,
                         stability_limited_ms=slim_ms,
                         development_speedup=slim_ms / timing[f"query_{best}"]["median_ms"],
                         selection_rule=rule),
    )
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}")
    print(json.dumps(out["frozen_from"], indent=2))


if __name__ == "__main__":
    main()
