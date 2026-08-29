"""Generate the TENSOR-NOTES.md tables from the run JSONs (never hand-type
numbers).  Run from experiments/separable-decoder:

    /home/tahmid/Dev/.venv/bin/python runs/b1dtensor/gen_tables.py

Reads runs/b1dtensor/tensor_n*/out/sep_b1d_tensor_n*.json (A100 jobs) and
runs/b1dtensor/audit/audit_n*.json (E1 CPU audit); prints markdown.
"""
from __future__ import annotations

import glob
import json
import os
import re

HERE = os.environ.get("TABLES_DIR", os.path.dirname(os.path.abspath(__file__)))


def load(pattern):
    out = {}
    for p in sorted(glob.glob(pattern)):
        n = int(re.search(r"_n(\d+)\.json$", p).group(1))
        out[n] = json.load(open(p))
    return dict(sorted(out.items()))


def e(x, f="{:.3e}"):
    return "n/a" if x is None else f.format(x)


def main():
    jobs = load(os.path.join(HERE, "tensor_n*", "out", "sep_b1d_tensor_n*.json"))
    # the first four-job experiment (N <= 1024); the large-N jobs belong to
    # the constant-time verification (gen_ladder.py)
    jobs = {n: d for n, d in jobs.items() if n <= 1024}
    audits = load(os.path.join(HERE, "audit", "audit_n*.json"))

    print("### Provenance (A100 jobs)\n")
    print("| N | job | node | GPU | backend | commit | jax | complete | secs |")
    print("|---|---|---|---|---|---|---|---|---|")
    for n, d in jobs.items():
        c = d["config"]
        print(f"| {n} | {c['slurm_job']} | {c['node']} | {c['gpu']} | "
              f"{c['backend']} | {str(c['commit'])[:10]} | {c['jax_version']} | "
              f"{d['complete']} | {d.get('secs_total', 0):.0f} |")

    print("\n### Key table: rollout error (mean rel-L2 over 8 test trajectories, "
          "51 states each) and e2e ms (median over 8 traj x 5 reps; optimized "
          "path)\n")
    print("| N | oracle err | tensor err | max per-traj abs diff | NNLS-32 err | "
          "learned-32 err | e2e oracle | e2e tensor | e2e tensor_nolean | "
          "e2e NNLS-32 | e2e learned-32 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for n, d in jobs.items():
        v = d["variants"]
        cmp = d["comparison"]
        print(f"| {n} | {e(v['oracle']['err_fast_mean'], '{:.6e}')} | "
              f"{e(v['tensor']['err_fast_mean'], '{:.6e}')} | "
              f"{e(cmp['tensor']['err_abs_diff_max'], '{:.2e}')} | "
              f"{e(v['base_tight']['err_fast_mean'], '{:.6e}')} | "
              f"{e(v['nodes_tight']['err_fast_mean'], '{:.6e}')} | "
              f"{v['oracle']['e2e_fast_ms']:.2f} | {v['tensor']['e2e_fast_ms']:.2f} | "
              f"{v.get('tensor_nolean', {}).get('e2e_fast_ms', float('nan')):.2f} | "
              f"{v['base_tight']['e2e_fast_ms']:.2f} | "
              f"{v['nodes_tight']['e2e_fast_ms']:.2f} |")

    print("\n### Pass criteria per N (tensor vs oracle, optimized path)\n")
    print("| N | (i) T0 exact on all-positive states | (ii) stop-reason hist "
          "identical (total / per traj) | (iii) max abs err diff (<= 1e-5?) | "
          "latent dev max | (iv) tensor vs NNLS-32 / learned-32 (err ratio) | "
          "(v) e2e ratio vs oracle / vs NNLS-32 | roll ratio vs oracle / NNLS-32 |")
    print("|---|---|---|---|---|---|---|---|")
    for n, d in jobs.items():
        v = d["variants"]
        g = d["gates"]
        c = d["comparison"]["tensor"]
        print(f"| {n} | {e(g['T0_all_positive_states_max_rel'], '{:.1e}')} "
              f"({g['T0_n_all_positive_states']} states) | "
              f"{c['stop_hist_identical']} / {c['stop_hist_identical_per_traj']} | "
              f"{c['err_abs_diff_max']:.2e} ({'yes' if c['err_abs_diff_max'] <= 1e-5 else 'NO'}) | "
              f"{c['lat_dev_fast_max']:.1e} | "
              f"{v['tensor']['err_fast_mean']/v['base_tight']['err_fast_mean']:.3f} / "
              f"{v['tensor']['err_fast_mean']/v['nodes_tight']['err_fast_mean']:.3f} | "
              f"{c['e2e_ratio_vs_oracle']:.3f} / {c['e2e_ratio_vs_base_tight']:.3f} | "
              f"{c['roll_ratio_vs_oracle']:.3f} / {c['roll_ratio_vs_base_tight']:.3f} |")

    print("\n### Stop-reason histograms (optimized path, 8 traj x 50 steps; "
          "0=budget, 1=tol, 2=stall, 3=lambda-max, 4=already converged, 5=nonfinite)\n")
    print("| N | oracle | tensor | tensor_nolean | NNLS-32 | learned-32 | "
          "committed baseline oracle |")
    print("|---|---|---|---|---|---|---|")
    for n, d in jobs.items():
        v = d["variants"]
        print(f"| {n} | {v['oracle']['stop_reasons_fast']} | "
              f"{v['tensor']['stop_reasons_fast']} | "
              f"{v.get('tensor_nolean', {}).get('stop_reasons_fast', 'n/a')} | "
              f"{v['base_tight']['stop_reasons_fast']} | "
              f"{v['nodes_tight']['stop_reasons_fast']} | "
              f"{v['oracle']['parity'].get('base_stop_reasons', 'n/a')} |")

    print("\n### Per-trajectory tensor vs oracle (optimized path)\n")
    print("| N | traj | nu | oracle err | tensor err | abs diff | latent dev | "
          "reasons equal | NNLS-32 err | learned-32 err |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for n, d in jobs.items():
        v = d["variants"]
        for r in d["comparison"]["tensor"]["per_traj"]:
            t = r["traj"]
            print(f"| {n} | {t} | {v['oracle']['rollout'][t]['nu']:.4f} | "
                  f"{r['err_oracle']:.6e} | {r['err_arm']:.6e} | "
                  f"{r['abs_diff']:.2e} | {r['lat_dev_fast']:.2e} | "
                  f"{r['reasons_equal']} | "
                  f"{v['base_tight']['rollout'][t]['err_fast']:.6e} | "
                  f"{v['nodes_tight']['rollout'][t]['err_fast']:.6e} |")

    print("\n### Timing split per arm (ms, medians; ref = verbatim jacfwd/LU "
          "reference rollout, fast = optimized path)\n")
    print("| N | arm | ic ref->fast | roll ref->fast | dec | e2e ref | e2e fast | "
          "committed baseline e2e (A100, scale job) |")
    print("|---|---|---|---|---|---|---|---|")
    for n, d in jobs.items():
        for arm, v in d["variants"].items():
            print(f"| {n} | {arm} | {v['ic_ref_ms']:.2f}->{v['ic_fast_ms']:.2f} | "
                  f"{v['roll_ref_ms']:.2f}->{v['roll_fast_ms']:.2f} | "
                  f"{v['dec_ms']:.2f} | {v['e2e_ref_ms']:.2f} | {v['e2e_fast_ms']:.2f} | "
                  f"{e(v['parity'].get('base_e2e_ms_median'), '{:.2f}')} |")

    print("\n### Parity of the re-run arms against the committed baseline JSONs "
          "(rel diff of the 8-trajectory mean error; ref path vs committed, "
          "fast path vs ref)\n")
    print("| N | arm | ref vs committed | max per-traj ref vs committed | "
          "fast vs ref | latent dev ref vs fast |")
    print("|---|---|---|---|---|---|")
    for n, d in jobs.items():
        for arm, v in d["variants"].items():
            p = v["parity"]
            print(f"| {n} | {arm} | {e(p.get('err_rel_diff_ref_vs_base'), '{:.1e}')} | "
                  f"{e(p.get('per_traj_rel_diff_ref_vs_base_max'), '{:.1e}')} | "
                  f"{p['err_rel_diff_fast_vs_ref']:.1e} | {p['lat_dev_max']:.1e} |")

    print("\n### Gates (in-job, A100)\n")
    print("| N | J | T2 | E | F | C | G rel | V rel (tensor) | TB | TX | TA | "
          "T0 | TQ r rel med / max | TQ J rel max | TQ states with u<=0 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for n, d in jobs.items():
        g = d["gates"]
        tq = g["TQ"]
        print(f"| {n} | {g['gateJ']:.1e} | {g['gateT2']:.1e} | {g['gateE']:.1e} | "
              f"{g['gateF']:.1e} | {g['gateC']:.1e} | {g['gateG']['rel_diff']:.1e} | "
              f"{e(g.get('gateV', {}).get('rel_diff'), '{:.1e}')} | "
              f"{g['TB_build_order_rel']:.1e} | {e(g.get('TX_vs_audit_Q_rel'), '{:.1e}')} | "
              f"{g['TA_algebraic_identity_max_rel']:.1e} | "
              f"{e(g['T0_all_positive_states_max_rel'], '{:.1e}')} | "
              f"{tq['r_rel']['median']:.1e} / {tq['r_rel']['max']:.1e} | "
              f"{tq['J_rel']['max']:.1e} | {tq['n_states_with_neg']}/{tq['r_rel']['n']} |")

    print("\n### E1 CPU audit (local GB10, JAX_PLATFORMS=cpu, committed checkpoints)\n")
    print("| N | TB | TA | T0 (all-positive states) | TS mismatch med / mean / max | "
          "frac points u<=0 | min u (train) | all-positive states | TC max / median | "
          "TJ J rel max (unpert / pert) | TJ grad cos min |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for n, a in audits.items():
        g = a["gates"]
        ts = a["TS_train_states"]
        tcx = a["TC_contraction"]
        print(f"| {n} | {g['TB_build_order_rel']:.1e} | "
              f"{g['TA_algebraic_identity_max_rel']:.1e} | "
              f"{e(g['T0_all_positive_states_max_rel'], '{:.1e}')} | "
              f"{ts['mismatch_rel']['median']:.1e} / {ts['mismatch_rel']['mean']:.1e} / "
              f"{ts['mismatch_rel']['max']:.1e} | {ts['frac_points_u_le_0']:.2%} | "
              f"{ts['min_u']:.2e} | {ts['frac_states_all_positive']:.1%} | "
              f"{tcx['per_entry_max']:.1e} / {tcx['per_entry_median']:.1e} | "
              f"{a['TJ_unperturbed']['J_rel']['max']:.1e} / "
              f"{a['TJ_perturbed_0.05']['J_rel']['max']:.1e} | "
              f"{min(a['TJ_unperturbed']['g_cos_min'], a['TJ_perturbed_0.05']['g_cos_min']):.6f} |")

    print("\n### E1 sign audit at every LM candidate of the oracle host-loop "
          "rollout (8 test trajectories)\n")
    print("| N | candidates (rejected) | inits | cands with a u<=0 point "
          "(accepted) | inits with u<=0 | min u | q rel med / max | r rel med / max | "
          "J rel max | grad cos min (non-stationary) | scaled grad mismatch max | "
          "host-vs-device latdev |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for n, a in audits.items():
        t = a["TL_candidates"]
        print(f"| {n} | {t['n_candidates']} ({t['n_rejected']}) | {t['n_init']} | "
              f"{t['n_with_neg']} ({t['n_accepted_with_neg']}) | {t['n_init_with_neg']} | "
              f"{t['min_u']:.2e} | {t['q_rel']['median']:.1e} / {t['q_rel']['max']:.1e} | "
              f"{t['r_rel']['median']:.1e} / {t['r_rel']['max']:.1e} | "
              f"{t['J_rel']['max']:.1e} | {t['g_cos_min_nonstationary']:.6f} "
              f"({t['n_nonstationary']}) | {t['g_scaled']['max']:.1e} | "
              f"{t['latdev_host_vs_device_max']:.1e} |")


if __name__ == "__main__":
    main()
