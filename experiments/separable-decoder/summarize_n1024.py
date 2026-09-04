"""Generate the N=1024 scaling-round summary tables from the run JSONs.

Usage:  python summarize_n1024.py [runs_glob ...] > SUMMARY-N1024.md

Reads every sep_poisson_*.json / sep_burgers_*.json under the given globs
(default: runs/sepdec_n1024_*/out/*.json) and emits markdown tables.  All
numbers come from the JSONs -- nothing is hand-typed (project rule).
"""
from __future__ import annotations

import glob
import json
import sys


def fmt(x, spec=".3e"):
    if x is None:
        return "--"
    try:
        return format(float(x), spec)
    except (TypeError, ValueError):
        return str(x)


def load(globs):
    paths = []
    for g in globs:
        paths += glob.glob(g)
    out = []
    for p in sorted(set(paths)):
        try:
            d = json.load(open(p))
            d["_path"] = p
            out.append(d)
        except Exception as e:                                    # noqa: BLE001
            print(f"<!-- SKIPPED {p}: {e} -->")
    return out


def poisson_tables(docs):
    docs = [d for d in docs if d.get("config", {}).get("pde") == "poisson2d"]
    if not docs:
        return
    print("\n## Poisson 2D, N=1024\n")
    print("### Cells\n")
    print("| K | R | steps | train s | recon mean | oracle held-out | "
          "oracle fresh | gate0 | EQ rel fit | EQ row p95 | EQ row max | "
          "complete |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for d in docs:
        c, t, e = d["config"], d.get("train", {}), d.get("eq", {})
        oh = d.get("oracle_held_out_seed0", {})
        of = d.get("oracle_fresh_seed", {})
        print(f"| {c['k']} | {c['r']} | {c['steps']} | "
              f"{fmt(t.get('seconds'), '.0f')} | "
              f"{fmt(t.get('recon_rel_l2_mean'))} | {fmt(oh.get('mean'))} | "
              f"{fmt(of.get('mean'))} | {fmt(d.get('gate0_max_rel_dev'), '.1e')} | "
              f"{fmt(e.get('rel_fit'), '.1e')} | {fmt(e.get('row_rel_p95'), '.1e')} | "
              f"{fmt(e.get('row_rel_max'), '.1e')} | {d.get('complete')} |")
    print("\n### ROM solve rows (end-to-end: source -> projection -> LM -> "
          "full decode)\n")
    print("| K | R | cohort | arm | tau | time ms | err mean | err max | "
          "jac | censored | stop reasons | timed-vs-err dev |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for d in docs:
        c = d["config"]
        for r in d.get("rows", []):
            print(f"| {c['k']} | {c['r']} | {r['cohort']} | {r['method']} | "
                  f"{fmt(r['tau'], '.0e')} | {fmt(r['time_ms'], '.3f')} | "
                  f"{fmt(r['err_rel_l2'])} | {fmt(r['err_rel_l2_max'])} | "
                  f"{fmt(r['jac_evals'], '.1f')} | "
                  f"{fmt(100 * r['censored_frac'], '.0f')}% | "
                  f"{r.get('stop_reasons')} | "
                  f"{fmt(r.get('timed_vs_error_max_dev'), '.1e')} |")
    print("\n### Classical baselines (same job, same GPU)\n")
    print("| K | R | cohort | solver | tol | time ms | err mean | err max |")
    print("|---|---|---|---|---|---|---|---|")
    for d in docs:
        c = d["config"]
        for r in d.get("fom", []):
            print(f"| {c['k']} | {c['r']} | {r['cohort']} | {r['fom']} | "
                  f"{fmt(r.get('fom_tol'), '.0e') if r.get('fom_tol') else '--'} | "
                  f"{fmt(r['time_ms'], '.3f')} | {fmt(r['err_rel_l2'])} | "
                  f"{fmt(r['err_rel_l2_max'])} |")
    print("\n### Balanced AB/BA paired blocks (cached ROM vs baseline)\n")
    print("| K | R | cohort | ROM | baseline | ROM ms | base ms | ratio |")
    print("|---|---|---|---|---|---|---|---|")
    for d in docs:
        c = d["config"]
        for r in d.get("paired", []):
            ratio = (r["base_ms"] / r["rom_ms"]) if r["rom_ms"] else None
            bl = r["baseline"] + (f" tol={fmt(r['fom_tol'], '.0e')}"
                                  if r.get("fom_tol") else "")
            print(f"| {c['k']} | {c['r']} | {r['cohort']} | {r['rom']} | {bl} | "
                  f"{fmt(r['rom_ms'], '.3f')} | {fmt(r['base_ms'], '.3f')} | "
                  f"{fmt(ratio, '.3f')} |")


def burgers_tables(docs):
    docs = [d for d in docs if d.get("config", {}).get("pde") == "burgers2d"]
    if not docs:
        return
    print("\n## Burgers 2D, N=1024 (50 implicit steps)\n")
    print("### Cells\n")
    print("| K | R | steps | states | train s | recon mean | gate0 | "
          "EQ rel fit | EQ row p95 | EQ row max | complete |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for d in docs:
        c, t, e = d["config"], d.get("train", {}), d.get("eq", {})
        print(f"| {c['k']} | {c['r']} | {c['steps']} | "
              f"{d.get('data', {}).get('n_states_trained')} | "
              f"{fmt(t.get('seconds'), '.0f')} | "
              f"{fmt(t.get('recon_rel_l2_mean'))} | "
              f"{fmt(d.get('gate0_max_rel_dev'), '.1e')} | "
              f"{fmt(e.get('rel_fit'), '.1e')} | {fmt(e.get('row_rel_p95'), '.1e')} | "
              f"{fmt(e.get('row_rel_max'), '.1e')} | {d.get('complete')} |")
    print("\n### ROM rollout rows (cached arm timed END-TO-END: u0 -> IC fit "
          "-> 50 LM steps -> full 51-state decode)\n")
    print("| K | R | arm | err mean | err max | e2e ms | (ic ms | roll+dec ms) "
          "| jac | blowups | stop reasons | timed-vs-err dev |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for d in docs:
        c = d["config"]
        for r in d.get("rows", []):
            dev = None
            if r.get("per_traj"):
                devs = [p.get("timed_vs_error_max_latent_dev")
                        for p in r["per_traj"]
                        if p.get("timed_vs_error_max_latent_dev") is not None]
                dev = max(devs) if devs else None
            print(f"| {c['k']} | {c['r']} | {r['method']} | "
                  f"{fmt(r.get('err_traj_rel_mean'))} | "
                  f"{fmt(r.get('err_traj_rel_max'))} | "
                  f"{fmt(r.get('e2e_ms_median'), '.2f')} | "
                  f"{fmt(r.get('icfit_ms_median'), '.2f')} | "
                  f"{fmt(r.get('rolldec_ms_median'), '.2f')} | "
                  f"{fmt(r.get('jac_total_mean'), '.0f')} | "
                  f"{r.get('n_blowups')} | {r.get('stop_reasons')} | "
                  f"{fmt(dev, '.1e')} |")
    print("\n### IC fits (per trajectory, cached span-split fitter)\n")
    print("| K | R | traj | ic rel | est dev | incumbent check |")
    print("|---|---|---|---|---|---|")
    for d in docs:
        c = d["config"]
        row0 = next((r for r in d.get("rows", [])
                     if r["method"] == "sep_cached"), None)
        for p in (row0 or {}).get("per_traj", []):
            print(f"| {c['k']} | {c['r']} | {p['traj']} | {fmt(p['ic_rel'])} | "
                  f"{fmt(abs(p['ic_rel'] - p['ic_rel_est']), '.1e')} | "
                  f"{fmt(p.get('ic_rel_incumbent_mean_init_b30'))} |")
    print("\n### Classical baselines (same job, same GPU, full-grid outputs)\n")
    print("| K | R | solver | setting | time ms/traj | err mean | err max | "
          "newton/traj |")
    print("|---|---|---|---|---|---|---|---|")
    for d in docs:
        c = d["config"]
        ft = d.get("fom_truth")
        if ft:
            print(f"| {c['k']} | {c['r']} | truth Newton (OVER-SOLVED) | "
                  f"fixed {ft['newton_iters']} it, lin {fmt(ft['lin_tol'], '.0e')} | "
                  f"{fmt(ft['time_ms_median'], '.2f')} | ~0 (truth) | -- | "
                  f"{ft['newton_iters'] * c['num_steps']} |")
        for r in d.get("fom_tolnewton", []):
            print(f"| {c['k']} | {c['r']} | tol-Newton | "
                  f"ntol {fmt(r['ntol'], '.0e')}, lin {fmt(r['lin_tol'], '.0e')} | "
                  f"{fmt(r['time_ms_median'], '.2f')} | "
                  f"{fmt(r['err_traj_rel_mean'])} | {fmt(r['err_traj_rel_max'])} | "
                  f"{fmt(r['newton_total_mean'], '.0f')} |")
    print("\n### Balanced AB/BA paired blocks (ROM end-to-end vs tol-Newton)\n")
    print("| K | R | baseline | ROM ms | base ms | ratio (base/ROM) |")
    print("|---|---|---|---|---|---|")
    for d in docs:
        c = d["config"]
        for r in d.get("paired", []):
            ratio = (r["base_ms"] / r["rom_ms"]) if r["rom_ms"] else None
            print(f"| {c['k']} | {c['r']} | {r['baseline']} | "
                  f"{fmt(r['rom_ms'], '.2f')} | {fmt(r['base_ms'], '.2f')} | "
                  f"{fmt(ratio, '.3f')} |")


def main():
    globs = sys.argv[1:] or ["runs/sepdec_n1024_*/out/*.json"]
    docs = load(globs)
    print("# Separable EQ-decoder NM-ROM at N=1024 -- generated summary")
    print(f"\nGenerated by summarize_n1024.py from {len(docs)} run JSON(s); "
          "no hand-typed numbers.")
    for d in docs:
        c = d["config"]
        print(f"- `{d['_path']}`: {c['pde']} K={c['k']} R={c['r']} "
              f"job={c.get('slurm_job')} gpu={c.get('gpu')} "
              f"complete={d.get('complete')}")
    poisson_tables(docs)
    burgers_tables(docs)


if __name__ == "__main__":
    main()
