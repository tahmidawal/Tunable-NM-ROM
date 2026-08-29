"""Constant-time-verification tables (never hand-type numbers).  Run from
experiments/separable-decoder:

    /home/tahmid/Dev/.venv/bin/python runs/b1dtensor/gen_ladder.py

(1) N x arm table of ic / solve / dec / e2e ms and ms per LM attempt from
    JOB A alone (runs/b1dtensor/ladder1/out/sep_b1d_ladder.json, one GPU,
    one process, interleaved);
(2) the same quantities from the separate large-N jobs
    (runs/b1dtensor/tensor_n{16384,65536}/out/sep_b1d_tensor_n*.json),
    flagged "different job/GPU";
(3) log-log slopes of solve ms vs N for the tensor and oracle arms.
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np

HERE = os.environ.get("TABLES_DIR", os.path.dirname(os.path.abspath(__file__)))
ARMS = ["oracle", "base_tight", "tensor"]


def slope(ns, ys):
    return float(np.polyfit(np.log(ns), np.log(ys), 1)[0])


def main():
    lp = os.path.join(HERE, "ladder1", "out", "sep_b1d_ladder.json")
    L = json.load(open(lp)) if os.path.exists(lp) else None
    big = {}
    for p in sorted(glob.glob(os.path.join(HERE, "tensor_n*", "out",
                                           "sep_b1d_tensor_n*.json"))):
        n = int(re.search(r"_n(\d+)\.json$", p).group(1))
        if n > 4096:
            big[n] = json.load(open(p))
    bigscale = {}
    for n in big:
        p = os.path.join(HERE, f"tensor_n{n}", "out", f"sep_b1d_scale_n{n}.json")
        if os.path.exists(p):
            bigscale[n] = json.load(open(p))

    if L is not None:
        c = L["config"]
        print(f"### JOB A — ladder on one GPU\n\nJob {c['slurm_job']}, node "
              f"{c['node']}, GPU **{c['gpu']}**, commit {str(c['commit'])[:10]}, "
              f"jax {c['jax_version']}, {c['time_reps']} timed reps + {c['burn']} "
              f"burn, order: {c['order']}; complete={L['complete']}, "
              f"{L.get('secs_total', 0):.0f} s.\n")
        print("| N | arm | ic ms | solve ms | dec ms | e2e ms | LM attempts / traj "
              "| us per LM attempt (median over traj) | err | vs committed err | "
              "committed solve ms (own job) |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        NS = [int(n) for n in L["cells"]]
        for n in NS:
            for arm in ARMS:
                v = L["cells"][str(n)][arm]
                print(f"| {n} | {arm} | {v['ic_ms']:.2f} | {v['roll_ms']:.2f} | "
                      f"{v['dec_ms']:.2f} | {v['e2e_ms']:.2f} | "
                      f"{v['lm_attempts_total_mean']:.1f} | "
                      f"{v['ms_per_lm_attempt_median']*1e3:.1f} | "
                      f"{v['err_mean']:.6e} | "
                      f"{v.get('parity_err_rel_diff_vs_committed', float('nan')):.1e} | "
                      f"{v.get('committed_roll_ms', float('nan')):.2f} |")
        print("\n#### Tensor vs oracle inside JOB A\n")
        print("| N | max per-traj abs err diff | stop hist identical | LM attempt "
              "counts identical | solve ratio tensor/oracle | solve ratio "
              "tensor/NNLS-32 | e2e ratio tensor/oracle | e2e ratio tensor/NNLS-32 |")
        print("|---|---|---|---|---|---|---|---|")
        for n in NS:
            cl = L["cells"][str(n)]
            d = cl["tensor_vs_oracle"]
            print(f"| {n} | {d['err_abs_diff_max']:.2e} | {d['stop_hist_identical']} | "
                  f"{d['attempts_identical']} | "
                  f"{cl['tensor']['roll_ms']/cl['oracle']['roll_ms']:.3f} | "
                  f"{cl['tensor']['roll_ms']/cl['base_tight']['roll_ms']:.3f} | "
                  f"{cl['tensor']['e2e_ms']/cl['oracle']['e2e_ms']:.3f} | "
                  f"{cl['tensor']['e2e_ms']/cl['base_tight']['e2e_ms']:.3f} |")
        print("\n#### Slopes from JOB A alone: exponent p in ms ~ N^p "
              "(least-squares fit of log ms vs log N)\n")
        print("| arm | N range | solve ms | us per LM attempt | ic ms | e2e ms |")
        print("|---|---|---|---|---|---|")
        for arm in ARMS:
            f = L["fits"]
            print(f"| {arm} | {min(NS)}..{max(NS)} | "
                  f"{f[f'{arm}:roll_ms']['exponent']:+.4f} | "
                  f"{f[f'{arm}:ms_per_lm_attempt_median']['exponent']:+.4f} | "
                  f"{f[f'{arm}:ic_ms']['exponent']:+.4f} | "
                  f"{f[f'{arm}:e2e_ms']['exponent']:+.4f} |")

    if big:
        print("\n### JOBs B/C — large N, each its own job and GPU (DIFFERENT "
              "JOB/GPU from JOB A: compare with care)\n")
        print("| N | job | node | GPU | arm | ic ms | solve ms | dec ms | e2e ms "
              "| err | stop hist | trained in-job (recon rel-L2 / train s) |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for n, d in big.items():
            c = d["config"]
            tr = bigscale.get(n, {}).get("train", {})
            trs = (f"{tr.get('recon_rel_l2_mean', float('nan')):.3e} / "
                   f"{tr.get('seconds', float('nan')):.0f}") if tr else "n/a"
            for arm in ARMS + ["nodes_tight", "tensor_nolean"]:
                v = d["variants"].get(arm)
                if v is None:
                    continue
                print(f"| {n} | {c['slurm_job']} | {c['node']} | {c['gpu']} | {arm} | "
                      f"{v['ic_fast_ms']:.2f} | {v['roll_fast_ms']:.2f} | "
                      f"{v['dec_ms']:.2f} | {v['e2e_fast_ms']:.2f} | "
                      f"{v['err_fast_mean']:.6e} | {v['stop_reasons_fast']} | {trs} |")
        print("\n#### Large-N tensor vs oracle (within each job)\n")
        print("| N | max per-traj abs err diff | stop hist identical | latent dev "
              "max | solve ratio tensor/oracle | solve ratio tensor/NNLS-32 | e2e "
              "ratio tensor/oracle | e2e ratio tensor/NNLS-32 | FOM 1e-3 ms/traj "
              "(err) | FOM 1e-8 ms/traj (err) |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for n, d in big.items():
            cm = d["comparison"]["tensor"]
            fom = bigscale.get(n, {}).get("fom", [])
            fs = [f"{f['ms_median']:.2f} ({f['err_mean']:.2e})" for f in fom]
            while len(fs) < 2:
                fs.append("n/a")
            print(f"| {n} | {cm['err_abs_diff_max']:.2e} | {cm['stop_hist_identical']} | "
                  f"{cm['lat_dev_fast_max']:.1e} | {cm['roll_ratio_vs_oracle']:.3f} | "
                  f"{cm['roll_ratio_vs_base_tight']:.3f} | {cm['e2e_ratio_vs_oracle']:.3f} "
                  f"| {cm['e2e_ratio_vs_base_tight']:.3f} | {fs[0]} | {fs[1]} |")
        print("\n#### Large-N gates\n")
        print("| N | J (at N) | T2 (at N, traj) | E | F | C | G rel | V rel | TB | "
              "TA | T0 (n states) | TQ r rel med / max | oracle parity vs in-job "
              "scale run |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for n, d in big.items():
            g = d["gates"]
            tq = g["TQ"]
            po = d["variants"]["oracle"]["parity"]
            print(f"| {n} | {g['gateJ']:.1e} ({g.get('gateJ_N', n)}) | "
                  f"{g['gateT2']:.1e} ({g.get('gateT2_N', n)}, {g.get('gateT2_traj', 4)}) "
                  f"| {g['gateE']:.1e} | {g['gateF']:.1e} | {g['gateC']:.1e} | "
                  f"{g['gateG']['rel_diff']:.1e} | {g['gateV']['rel_diff']:.1e} | "
                  f"{g['TB_build_order_rel']:.1e} | {g['TA_algebraic_identity_max_rel']:.1e} | "
                  f"{g['T0_all_positive_states_max_rel']:.1e} ({g['T0_n_all_positive_states']}) | "
                  f"{tq['r_rel']['median']:.1e} / {tq['r_rel']['max']:.1e} | "
                  f"{po.get('err_rel_diff_ref_vs_base', float('nan')):.1e} |")

    if L is not None and big:
        print("\n### Slopes across the FULL range (JOB A points + large-N points "
              "from DIFFERENT jobs/GPUs — cross-job, indicative only)\n")
        NS = [int(n) for n in L["cells"]]
        print("| arm | N range | solve ms exponent | e2e ms exponent | solve ms at "
              "each N |")
        print("|---|---|---|---|---|")
        for arm in ARMS:
            ns = NS + sorted(big)
            ys = [L["cells"][str(n)][arm]["roll_ms"] for n in NS] + \
                 [big[n]["variants"][arm]["roll_fast_ms"] for n in sorted(big)]
            es = [L["cells"][str(n)][arm]["e2e_ms"] for n in NS] + \
                 [big[n]["variants"][arm]["e2e_fast_ms"] for n in sorted(big)]
            print(f"| {arm} | {min(ns)}..{max(ns)} | {slope(ns, ys):+.4f} | "
                  f"{slope(ns, es):+.4f} | "
                  f"{', '.join(f'{n}:{y:.1f}' for n, y in zip(ns, ys))} |")
        print("\n#### Same, large-N jobs only (N=16384..65536, two separate jobs)\n")
        if len(big) >= 2:
            print("| arm | solve ms exponent | e2e ms exponent |")
            print("|---|---|---|")
            for arm in ARMS:
                ns = sorted(big)
                ys = [big[n]["variants"][arm]["roll_fast_ms"] for n in ns]
                es = [big[n]["variants"][arm]["e2e_fast_ms"] for n in ns]
                print(f"| {arm} | {slope(ns, ys):+.4f} | {slope(ns, es):+.4f} |")


if __name__ == "__main__":
    main()
