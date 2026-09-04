#!/usr/bin/env python3
"""Generate every Burgers-3D table from the run JSONs (nothing hand-typed).

    python runs/b3dtensor/gen_tables.py [--out runs/b3dtensor/tables.md]

Reads: runs/b3dtensor/*/out/b3d_fom_gates_n*.json (phase 0), sep_b3d_pilot_*.json
(pilot), sep_b3d_tensor_n*.json (phases 1-3), sep_b3d_kernels.json (C1).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def f(x, fmt="{:.3e}"):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "True" if x else "False"
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return str(x)
    try:
        return fmt.format(float(x))
    except Exception:
        return str(x)


def load(pattern):
    out = []
    for p in sorted(glob.glob(os.path.join(HERE, pattern))):
        if "withdrawn" in p or "superseded" in p or "/smoke/" in p:
            continue
        try:
            out.append((p, json.load(open(p))))
        except Exception as e:  # pragma: no cover
            out.append((p, dict(error=str(e))))
    return out


def gates_table(runs):
    lines = ["| N | gate | value | pass | control | fired | note |", "|---|---|---|---|---|---|---|"]
    for p, r in runs:
        N = r.get("config", {}).get("N")
        for name, g in r.get("gates", {}).items():
            if not isinstance(g, dict):
                continue
            lines.append(f"| {N} | {name} | {f(g.get('value'))} | {f(g.get('passed'))} | "
                         f"{f(g.get('control'))} | {f(g.get('control_fired'))} | {g.get('control_note', g.get('note', ''))} |")
    return "\n".join(lines)


def pilot_table(runs):
    lines = ["| N | K | R | M | m | recon (pool) | D3 σmin/σmax | M-stab tail/head | D4 mean | D4 worst | D4 mean k>0 | pool/full max | opt max | control |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p, r in runs:
        c = r.get("config", {}); g = r.get("gates", {}); d4 = g.get("D4_heldout_oracle_validation", {}); d3 = g.get("D3_rank_of_A", {})
        tr = r.get("train", {})
        lines.append(f"| {c.get('N')} | {c.get('k')} | {c.get('r')} | {c.get('M')} | {c.get('m_nnls')} | {f(tr.get('recon_rel_l2_mean'))} | "
                     f"{f(d3.get('value'))} | {f(d3.get('M_stability_tail_over_head'))} | {f(d4.get('mean'))} | {f(d4.get('worst'))} | "
                     f"{f(d4.get('mean_k_gt0'))} | {f(d4.get('pool_to_full_ratio_max'))} | {f(d4.get('optimality_max'))} | {f(d4.get('control'))} |")
    return "\n".join(lines)


def arms_table(runs):
    lines = ["| N | GPU | arm | row allowed | err mean | err max | e2e ms | ic ms | solve ms | dec ms | attempts | censored | opt max | stop reasons | decoded states u<=0 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p, r in runs:
        c = r.get("config", {})
        allowed = r.get("preconditions", {}).get("result_rows_allowed", {})
        for a, v in r.get("variants", {}).items():
            if "err_traj_rel_mean" not in v:
                continue
            if not allowed.get(a, False):
                lines.append(f"| {c.get('N')} | {c.get('gpu')} | {a} | **NOT ALLOWED** (censored {v['censored_steps_total']}, gates {r.get('preconditions', {}).get('gates_all_pass')}) | | | | | | | | | | | |")
                continue
            lines.append(f"| {c.get('N')} | {c.get('gpu')} | {a} | {f(allowed.get(a))} | {f(v['err_traj_rel_mean'], '{:.6e}')} | {f(v['err_traj_rel_max'])} | "
                         f"{f(v['e2e_ms_median'], '{:.2f}')} | {f(v['ic_ms_median'], '{:.2f}')} | {f(v['roll_ms_median'], '{:.2f}')} | "
                         f"{f(v['dec_ms_median'], '{:.2f}')} | {f(v['attempts_total_mean'], '{:.0f}')} | {v['censored_steps_total']} | "
                         f"{f(v['optimality_max'])} | {v['stop_reasons']} | {f(v['decoded_frac_states_with_u_le0'], '{:.1%}')} |")
    return "\n".join(lines)


def cmp_table(runs):
    lines = ["| N | pair | err ratio | field rel diff max | latent dev max | stop hist identical | attempts identical | path r rel max | path J rel max | E1 | e2e ratio | solve ratio |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p, r in runs:
        c = r.get("config", {})
        for k, v in r.get("comparison", {}).items():
            pf = v.get("path_fidelity", {})
            lines.append(f"| {c.get('N')} | {k} | {f(v['err_ratio'], '{:.5f}')} | {f(v['field_rel_diff_max'])} | {f(v['lat_dev_max'])} | "
                         f"{f(v['stop_hist_identical_per_traj'])} | {f(v['attempts_identical_per_traj'])} | "
                         f"{f(pf.get('r_rel', {}).get('max'))} | {f(pf.get('J_rel', {}).get('max'))} | {f(v.get('E1_pass'))} | "
                         f"{f(v['e2e_ratio'], '{:.3f}')} | {f(v['roll_ratio'], '{:.3f}')} |")
    return "\n".join(lines)


def fom_table(runs):
    lines = ["| N | arm | ntol | lin_tol / k | err mean | ms | iters | stalled |", "|---|---|---|---|---|---|---|---|"]
    for p, r in runs:
        c = r.get("config", {})
        for row in r.get("fom", []):
            p2 = row["lin_tol"] if row["arm"] == "newton" else row["max_iter"]
            lines.append(f"| {c.get('N')} | {row['arm']} | {f(row['ntol'], '{:.0e}')} | {p2} | {f(row['err_traj_rel_mean'])} | "
                         f"{f(row['time_ms_median'], '{:.2f}')} | {f(row['iters_mean'], '{:.1f}')} | {row.get('stalled_steps_total', 0)} |")
    return "\n".join(lines)


def matched_table(runs):
    lines = ["| N | arm | ROM err | ROM e2e ms | bracket | matched arm | ntol | matched err | matched ms | paired ROM ms | paired FOM ms | speedup median | min | boot lower95 | all>1 | speed win |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p, r in runs:
        c = r.get("config", {})
        for a, e in r.get("matched", {}).get("arms", {}).items():
            m = e.get("matched") or {}
            pr = e.get("paired") or {}
            lines.append(f"| {c.get('N')} | {a} | {f(e['rom_err'])} | {f(e['rom_e2e_ms'], '{:.2f}')} | {f(e['bracket'])} | {m.get('arm', '—')} | "
                         f"{f(m.get('ntol'), '{:.0e}')} | {f(m.get('err'))} | {f(m.get('ms'), '{:.2f}')} | {f(pr.get('rom_ms'), '{:.2f}')} | "
                         f"{f(pr.get('fom_ms'), '{:.2f}')} | {f(pr.get('speedup'), '{:.2f}')} | {f(pr.get('speedup_min'), '{:.2f}')} | "
                         f"{f(pr.get('boot_lower95'), '{:.2f}')} | {f(pr.get('all_gt1'))} | {f(e.get('speed_win'))} |")
    return "\n".join(lines)


def tr_table(runs):
    lines = ["| N | TR candidates | r rel max | J rel max | decision agreement | concern | TQ r rel median | TQ r rel max | TQ J rel max | T0-decoded all-positive states | decoded frac points u<=0 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p, r in runs:
        c = r.get("config", {}); g = r.get("gates", {})
        tr = g.get("TR_candidate_path", {}); tq = g.get("TQ", {}); t0 = g.get("T0_decoded", {})
        lines.append(f"| {c.get('N')} | {tr.get('n_candidates', '—')} | {f(tr.get('r_rel_max'))} | {f(tr.get('J_rel_max'))} | "
                     f"{f(tr.get('decision_agreement'), '{:.4f}')} | {f(tr.get('concern'))} | {f(tq.get('r_rel', {}).get('median'))} | "
                     f"{f(tq.get('r_rel', {}).get('max'))} | {f(tq.get('J_rel', {}).get('max'))} | {t0.get('n_all_positive_states', '—')} | "
                     f"{f(t0.get('frac_points_u_le0'), '{:.3%}')} |")
    return "\n".join(lines)


def kernels_table(runs):
    lines = ["| N | K | R | M | kernel ms median | per-traj ms | LM attempts per traj |", "|---|---|---|---|---|---|---|"]
    for p, r in runs:
        for N, v in r.get("kernels", {}).items():
            att = [v.get(f"traj{i}_attempts") for i in range(8) if f"traj{i}_attempts" in v]
            lines.append(f"| {N} | {v.get('K')} | {v.get('R')} | {v.get('M')} | {f(v.get('kernel_ms_median'), '{:.2f}')} | "
                         f"{', '.join(f'{x:.2f}' for x in v.get('per_traj_ms', []))} | {att} |")
        c1 = r.get("C1", {})
        lines.append(f"| C1 | | | | max/min {f(c1.get('max_over_min'), '{:.3f}')} | pass {f(c1.get('passed'))} ({r.get('config', {}).get('gpu')}) |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "tables.md"))
    a = ap.parse_args()
    g = load("*/out/b3d_fom_gates_n*.json") + load("local/fom_gates_n*.json")
    pil = load("*/out/sep_b3d_pilot_*.json")
    main_ = load("*/out/sep_b3d_tensor_n*.json")
    ker = load("*/out/sep_b3d_kernels*.json")
    parts = ["<!-- GENERATED by runs/b3dtensor/gen_tables.py -- do not hand-edit -->",
             "### B-0 Phase 0 gates (per N; controls must fire)", gates_table(g),
             "### B-1 Capacity pilot (N=33, validation rows only)", pilot_table(pil),
             "### B-2 Arms per N", arms_table(main_),
             "### B-3 Comparisons (E1 oracle-equivalence)", cmp_table(main_),
             "### B-4 Tensor fidelity diagnostics (TQ, TR, T0-decoded)", tr_table(main_),
             "### B-5 Classical ladders (newton, defect)", fom_table(main_),
             "### B-6 Matched / bracket / paired AB/BA / bootstrap", matched_table(main_),
             "### B-7 Same-GPU kernels (C1)", kernels_table(ker)]
    open(a.out, "w").write("\n\n".join(parts) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
