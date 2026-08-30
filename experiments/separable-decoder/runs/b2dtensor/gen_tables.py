"""Generate the TENSOR2D-NOTES.md tables from the run JSONs (never hand-type
numbers).  Run from experiments/separable-decoder:

    /home/tahmid/Dev/.venv/bin/python runs/b2dtensor/gen_tables.py > runs/b2dtensor/tables.md

Reads runs/b2dtensor/n*/out/sep_b2d_tensor_n*.json (one job per N); prints
markdown.  Slopes are fitted ONLY within one job (one GPU); across N the jobs
are different GPUs/nodes, so cross-N numbers are presented as ratios and
labelled with their GPU.
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np

HERE = os.environ.get("TABLES_DIR", os.path.dirname(os.path.abspath(__file__)))


def load():
    out = {}
    for p in sorted(glob.glob(os.path.join(HERE, "n*", "out", "sep_b2d_tensor_n*.json"))):
        n = int(re.search(r"_n(\d+)\.json$", p).group(1))
        d = json.load(open(p))
        if d.get("complete"):
            out[n] = d
    return dict(sorted(out.items()))


def e(x, f="{:.3e}"):
    return "n/a" if x is None else f.format(x)


def main():
    jobs = load()
    arms_all = []
    for d in jobs.values():
        for a in d["config"]["arms_run"]:
            if a not in arms_all:
                arms_all.append(a)

    print("### T-1 Provenance (one job per N; the GPU differs across N)\n")
    print("| N | job | node | GPU | backend | commit | jax | checkpoint | trained in-job | complete | secs |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for n, d in jobs.items():
        c = d["config"]
        print(f"| {n} | {c['slurm_job']} | {c['node']} | {c['gpu']} | {c['backend']} | "
              f"{str(c['commit'])[:10]} | {c['jax_version']} | {os.path.basename(c['ckpt'])} | "
              f"{c['train_in_job']} | {d['complete']} | {d.get('secs_total', 0):.0f} |")

    print("\n### T-2 Positivity audit: truth (training + test states, interior points) and decoded states\n")
    print("| N | train traj / states | truth min u (train) | truth frac<0 (train) | truth min u (test) | "
          "truth frac<0 (test) | assert ok | decoded train states: min u / frac points u<=0 / frac all-positive | "
          "full-arm rollout: min u / frac states touching u<=0 / frac points u<=0 | tensor-arm rollout: min u / frac states touching u<=0 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for n, d in jobs.items():
        tr, te = d["data"]["train"], d["data"]["test"]
        ts = d["TS_train_states"]
        vf, vt = d["variants"].get("full"), d["variants"].get("tensor")
        print(f"| {n} | {tr['n_traj']} / {tr['n_states']} | {tr['min_u']:.2e} | {tr['frac_points_lt0']:.1e} | "
              f"{te['min_u']:.2e} | {te['frac_points_lt0']:.1e} | {d['data']['positivity_assert']['ok']} | "
              f"{ts['min_u']:.2e} / {ts['frac_points_u_le0']:.3%} / {ts['frac_states_all_positive']:.1%} | "
              f"{e(vf and vf['decoded_min_u'], '{:.2e}')} / {e(vf and vf['decoded_frac_states_with_u_le0'], '{:.1%}')} / "
              f"{e(vf and vf['decoded_frac_points_le0'], '{:.2%}')} | "
              f"{e(vt and vt['decoded_min_u'], '{:.2e}')} / {e(vt and vt['decoded_frac_states_with_u_le0'], '{:.1%}')} |")

    print("\n### T-3 Gates (asserted unless marked recorded)\n")
    print("| N | bank==meshfree | gate 0 | L | A | FOMR | TB | TA (states) | T0 (all-positive states) | "
          "TQ r rel med / max (recorded) | TQ J rel max (recorded) | TQ states with u<=0 | STEP | ROLL | "
          "IC gram vs full dz | R-lite recon mean | test FOM res | train FOM res |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for n, d in jobs.items():
        g = d["gates"]
        tq = g["TQ"]
        print(f"| {n} | {g['bank_vs_meshfree']:.1e} | {g['gate0']:.1e} | {g['gateL']:.1e} | {g['gateA']:.1e} | "
              f"{g['gateFOMR']:.1e} | {g['TB_build_order_rel']:.1e} | {g['TA_algebraic_identity_max_rel']:.1e} "
              f"({d['config']['ckpt_cfg'].get('max_snaps', '?')}) | "
              f"{e(g['T0_all_positive_states_max_rel'], '{:.1e}')} ({g['T0_n_all_positive_states']}) | "
              f"{tq['r_rel']['median']:.1e} / {tq['r_rel']['max']:.1e} | {tq['J_rel']['max']:.1e} | "
              f"{tq['n_states_with_neg']}/{tq['r_rel']['n']} | {g['STEP_aux_vs_make_step_lspg_var']:.0e} | "
              f"{g['ROLL_aux_vs_make_rollout_v2']:.0e} | {g['ic_gram_vs_full_latent_dev']:.1e} | "
              f"{g['R_lite_recon_on_regenerated_states']['mean']:.2e} | "
              f"{d['data']['test']['max_fom_rel_residual']:.1e} | {d['data']['train']['max_fom_rel_residual']:.1e} |")

    print("\n### T-4 Accuracy per arm per N (mean rel-L2 over 8 test trajectories x 51 states; "
          "fused e2e output of the last timed rep) and tensor-vs-full per-trajectory max |diff|\n")
    hdr = "| N | " + " | ".join(f"{a} err" for a in arms_all) + \
        " | tensor-full max abs diff | tensor/full err ratio | tensor/ex err ratio | " \
        "tensor-full latent dev max | stop hist identical tensor/full (per traj) | attempts identical |"
    print(hdr)
    print("|---|" + "---|" * (len(arms_all) + 6))
    for n, d in jobs.items():
        v = d["variants"]
        c = d["comparison"].get("tensor_vs_full", {})
        ce = d["comparison"].get("tensor_vs_ex", {})
        print(f"| {n} | " + " | ".join(e(v[a]['err_traj_rel_mean'], '{:.6e}') if a in v else "n/a"
                                        for a in arms_all)
              + f" | {e(c.get('err_abs_diff_max'), '{:.2e}')} | {e(c.get('err_ratio'), '{:.5f}')} | "
              f"{e(ce.get('err_ratio'), '{:.4f}')} | {e(c.get('lat_dev_max'), '{:.1e}')} | "
              f"{c.get('stop_hist_identical')} ({c.get('stop_hist_identical_per_traj')}) | "
              f"{c.get('attempts_identical_per_traj')} |")

    print("\n### T-5 Stop-reason histograms and LM counts (8 traj x 50 steps)\n")
    print("| N | arm | stop reasons | LM attempts / traj | accepted Jacobians / traj | IC rel err |")
    print("|---|---|---|---|---|---|")
    for n, d in jobs.items():
        for a in d["config"]["arms_run"]:
            v = d["variants"][a]
            print(f"| {n} | {a} | {v['stop_reasons']} | {v['attempts_total_mean']:.1f} | "
                  f"{v['jac_total_mean']:.1f} | {v['ic_rel_mean']:.3e} |")

    print("\n### T-6 Cost split per arm per N (ms per trajectory, median over 8 traj x 5 timed reps; "
          "arms interleaved AB/BA; ic / solve / dec are separately blocked phases, e2e is one fused jit)\n")
    print("| N | GPU | arm | ic ms | latent solve ms | decode ms | split sum ms | fused e2e ms | "
          "solve ratio vs ex | solve ratio vs full | e2e ratio vs ex | e2e ratio vs full |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for n, d in jobs.items():
        for a in d["config"]["arms_run"]:
            v = d["variants"][a]
            cx = d["comparison"].get(f"{a}_vs_ex", {})
            cf = d["comparison"].get(f"{a}_vs_full", {})
            print(f"| {n} | {d['config']['gpu']} | {a} | {v['ic_ms_median']:.2f} | {v['roll_ms_median']:.2f} | "
                  f"{v['dec_ms_median']:.2f} | {v['split_sum_ms_median']:.2f} | {v['e2e_ms_median']:.2f} | "
                  f"{e(cx.get('roll_ratio'), '{:.3f}')} | {e(cf.get('roll_ratio'), '{:.3f}')} | "
                  f"{e(cx.get('e2e_ratio'), '{:.3f}')} | {e(cf.get('e2e_ratio'), '{:.3f}')} |")

    print("\n### T-7 FOM cost per N (standardised tol-Newton ladder, same GPU as the ROM arms; "
          "matched = cheapest rung at least as accurate as the tensor arm; closest = rung with error "
          "closest to the tensor arm's in log; tightest = most accurate rung)\n")
    print("| N | GPU | tensor err / e2e ms | matched rung (nt, lt) err / ms | paired tensor vs matched: "
          "ROM ms / FOM ms / speedup | closest rung err / ms | tightest rung (nt, lt) err / ms | "
          "paired ex vs matched speedup | paired full vs matched speedup |")
    print("|---|---|---|---|---|---|---|---|---|")
    for n, d in jobs.items():
        m = d["matched"]
        t = m["arms"].get("tensor", {})
        mt = t.get("matched")
        pr = t.get("paired", {})
        tg = m["tightest"]
        cl = t.get("closest", {})
        ex = m["arms"].get("ex", {}).get("paired", {})
        fu = m["arms"].get("full", {}).get("paired", {})
        print(f"| {n} | {d['config']['gpu']} | {t.get('rom_err', float('nan')):.3e} / {t.get('rom_e2e_ms', float('nan')):.2f} | "
              + (f"({mt['newton_tol']:.0e}, {mt['lin_tol']:.0e}) {mt['err']:.2e} / {mt['ms']:.2f}" if mt else "none")
              + f" | {e(pr.get('rom_ms'), '{:.2f}')} / {e(pr.get('fom_ms'), '{:.2f}')} / {e(pr.get('speedup'), '{:.2f}x')} | "
              f"{e(cl.get('err'), '{:.2e}')} / {e(cl.get('ms'), '{:.2f}')} | "
              f"({tg['newton_tol']:.0e}, {tg['lin_tol']:.0e}) {tg['err']:.2e} / {tg['ms']:.2f} | "
              f"{e(ex.get('speedup'), '{:.2f}x')} | {e(fu.get('speedup'), '{:.2f}x')} |")

    print("\n### T-8 Full FOM ladder per N (err = mean rel-L2 vs the 8-Newton truth; ms median over 8 traj x 5 reps)\n")
    print("| N | newton_tol | lin_tol | err | ms | Newton iters / traj |")
    print("|---|---|---|---|---|---|")
    for n, d in jobs.items():
        for f_ in d["fom"]:
            print(f"| {n} | {f_['newton_tol']:.0e} | {f_['lin_tol']:.0e} | {f_['err_traj_rel_mean']:.3e} | "
                  f"{f_['time_ms_median']:.2f} | {f_['newton_iters_mean']:.0f} |")

    print("\n### T-9 Per-trajectory tensor vs full vs ex\n")
    print("| N | traj | nu | full err | tensor err | abs diff | latent dev | reasons equal | ex err | "
          + ("ex_learned err |" if "ex_learned" in arms_all else ""))
    print("|---|---|---|---|---|---|---|---|---|" + ("---|" if "ex_learned" in arms_all else ""))
    for n, d in jobs.items():
        v = d["variants"]
        if "tensor_vs_full" not in d["comparison"]:
            continue
        for r in d["comparison"]["tensor_vs_full"]["per_traj"]:
            t = r["traj"]
            line = (f"| {n} | {t} | {v['full']['per_traj'][t]['nu']:.4f} | {r['err_ref']:.6e} | "
                    f"{r['err_arm']:.6e} | {r['abs_diff']:.2e} | {r['lat_dev']:.2e} | {r['reasons_equal']} | "
                    f"{v['ex']['per_traj'][t]['traj_rel']:.6e} |")
            if "ex_learned" in arms_all:
                line += (f" {v['ex_learned']['per_traj'][t]['traj_rel']:.6e} |" if "ex_learned" in v
                         else " n/a |")
            print(line)

    print("\n### T-10 Cross-N cost ratios (DIFFERENT jobs and GPUs per N -- ratios, not exponents; "
          "the GPU is named per row)\n")
    print("| N | GPU | tensor solve ms | ex solve ms | full solve ms | tensor/ex | tensor/full | "
          "tensor ic ms | tensor dec ms | tensor e2e ms |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for n, d in jobs.items():
        v = d["variants"]
        print(f"| {n} | {d['config']['gpu']} | {v['tensor']['roll_ms_median']:.2f} | {v['ex']['roll_ms_median']:.2f} | "
              f"{v['full']['roll_ms_median']:.2f} | {v['tensor']['roll_ms_median']/v['ex']['roll_ms_median']:.3f} | "
              f"{v['tensor']['roll_ms_median']/v['full']['roll_ms_median']:.3f} | {v['tensor']['ic_ms_median']:.2f} | "
              f"{v['tensor']['dec_ms_median']:.2f} | {v['tensor']['e2e_ms_median']:.2f} |")
    same_gpu = {}
    for n, d in jobs.items():
        same_gpu.setdefault(d["config"]["gpu"], []).append(n)
    for gpu, ns in same_gpu.items():
        if len(ns) >= 3:
            print(f"\nLeast-squares slope of log(solve ms) vs log(N) over N={ns} (same GPU model "
                  f"{gpu}, but DIFFERENT jobs/nodes -- indicative only):")
            for a in ("tensor", "ex", "full"):
                y = np.log([jobs[n]["variants"][a]["roll_ms_median"] for n in ns])
                x = np.log(ns)
                p = np.polyfit(x, y, 1)[0]
                print(f"- {a}: {p:+.3f}")

    print("\n### T-11 Tensor build and NNLS fit (offline costs)\n")
    print("| N | Q shape | Q MiB | build s (one chunking) | T asymmetry rel | TB | NNLS m | NNLS rel fit | NNLS s |")
    print("|---|---|---|---|---|---|---|---|---|")
    for n, d in jobs.items():
        t = d["tensor"]
        q = d["eq"]
        print(f"| {n} | {t['shape']} | {t['bytes']/2**20:.1f} | {t['build_secs']:.2f} | {t['T_asym_rel']:.2f} | "
              f"{d['gates']['TB_build_order_rel']:.1e} | {q['m']} | {q.get('rel_fit', float('nan')):.2e} | "
              f"{q.get('secs', float('nan')):.0f} |")

    for n, d in jobs.items():
        if "train" in d:
            print(f"\nIn-job training at N={n}: recon rel-L2 mean {d['train']['recon_rel_l2_mean']:.3e} "
                  f"max {d['train']['recon_rel_l2_max']:.3e}, {d['train']['steps']} steps, "
                  f"{d['train']['n_snapshots']} states, {d['train']['seconds']:.0f} s.")


if __name__ == "__main__":
    main()
