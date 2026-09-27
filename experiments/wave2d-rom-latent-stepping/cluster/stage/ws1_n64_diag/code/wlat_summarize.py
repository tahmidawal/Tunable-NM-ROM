"""Regenerate every README table from the pulled run JSONs (runs/<cell>/out/*.json).

Usage: python wlat_summarize.py [runs_dir] > SUMMARY_TABLES.md
"""
from __future__ import annotations

import glob
import json
import os
import sys

RUNS = sys.argv[1] if len(sys.argv) > 1 else "runs"


def e(x, d=2):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "-"
    return "-" if x != x else f"{x:.{d}e}"


def load(pat):
    out = {}
    for f in sorted(glob.glob(os.path.join(RUNS, pat))):
        try:
            out[f] = json.load(open(f))
        except Exception as ex:                                   # noqa: BLE001
            print(f"<!-- unreadable {f}: {ex} -->")
    return out


def cell_of(f):
    return f.split(os.sep)[1] if os.sep in f else f


def main():
    print("# Wave-2D latent-stepping ROM — tables regenerated from the run JSONs\n")
    print(f"Source: `{RUNS}/<cell>/out/*.json`; regenerate with `python wlat_summarize.py`.\n")

    # ---------------- verification ----------------
    ver = load("*/out/wlat_verify_N*.json")
    for f, r in ver.items():
        print(f"## Verification ({cell_of(f)}, N={r['config']['N']})\n")
        print(f"- FOM energy drift **{e(r['V4_fom_energy_drift'])}**; "
              f"u-only Newmark(RS=80) energy drift {e(r.get('V4b_newmark_rs80_energy_drift'))}")
        print(f"- V1 u-only Newmark(RS=80) vs the (u,v) CN FOM: **{e(r.get('V1_newmark_rs80_vs_cn_fom'))}** "
              f"(CG-tolerance limited; the v-elimination is exact)")
        v2 = r["V2_residual_ops"]
        print(f"- V2 residual operators on exact Newmark states: strong-full {e(v2['strong_full'])}, "
              f"strong-subset {e(v2['strong_rand'])}, weak {e(v2['weak_full'])}; "
              f"a NON-solution gives {e(v2['weak_nonsolution'])} (guard against a trivially-zero residual)")
        print(f"- V2b the FOM's own states through the residual formula: {e(r['V2b_fom_traj_residual_rel'])}")
        v3 = r["V3_weak_allmodes_vs_strong"]
        print(f"- V3 weak form with ALL {v3['M']} modes and WEAK_ALPHA=0 vs the strong full-grid "
              f"residual: {e(v3['strong_norm'], 6)} vs {e(v3['weak_norm'], 6)}, rel diff "
              f"**{e(v3['rel_diff'])}**\n")
        print("| ROM sub-steps RS | dt | latent steps | u-only Newmark FOM vs the 80-substep FOM (traj-RMS) | max | energy drift |")
        print("|---|---|---|---|---|---|")
        for rs, v in sorted(r["V5_samedt_newmark_fom"].items(), key=lambda kv: int(kv[0])):
            print(f"| {rs} | {0.02/int(rs):.2e} | {v['n_latent_steps']} | "
                  f"**{e(v['traj_rel_vs_fom_mean'])}** | {e(v['traj_rel_vs_fom_max'])} | "
                  f"{e(v['energy_drift_max'])} |")
        print()
        if "V6_pod_floors_val" in r:
            print("POD projection floors (val, traj-RMS): "
                  + ", ".join(f"r{k}={e(v)}" for k, v in sorted(r["V6_pod_floors_val"].items(),
                                                                key=lambda kv: int(kv[0])))
                  + f"  (ortho dev {e(r['V6_pod_ortho_dev'])})\n")

    # ---------------- stage 1 ----------------
    s1 = load("*/**/wlat_stage1_N*.json") or load("*/out/wlat_stage1_N*.json")
    if s1:
        print("## Stage 1 — space-time LSPG on the (z,t) sweep decoder\n")
        print("| cell | IC_W | arm | traj-RMS mean | median | max | mean \\|z-z*\\| |")
        print("|---|---|---|---|---|---|---|")
        for f, r in s1.items():
            icw = r["config"]["ic_w"]
            o = r["oracle_true_z"]
            print(f"| {cell_of(f)} | — | oracle (true z) | **{e(o['traj_rel_mean'])}** | — | "
                  f"{e(o['traj_rel_max'])} | 0 |")
            for arm in ("ic", "resid", "both"):
                if arm in r:
                    a = r[arm]
                    print(f"| {cell_of(f)} | {icw:g} | `{arm}` | {e(a['traj_rel_mean'])} | "
                          f"{e(a['traj_rel_median'])} | {e(a['traj_rel_max'])} | {e(a['z_err_mean'])} |")
        for f, r in s1.items():
            print(f"\n`{cell_of(f)}`: Newmark-state residual check {e(r['newmark_states_residual_maxabs'])}; "
                  f"sweep-decoder residual at the true z (rel) {e(r['resid_at_true_z_rel'])}.")
        print()

    # ---------------- stage 2 ----------------
    rom = load("*/out/wlat_rom_*.json")
    if not rom:
        return
    print("## Stage 2 — auto-decoder latent-stepping ROM\n")
    print("### Floors\n")
    print("| cell | N | K | RS | AD train recon (traj-RMS) | oracle held-out latents | IC fit | "
          "same-dt Newmark FOM | POD proj floors (test) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for f, r in rom.items():
        c = r["config"]
        pf = r.get("oracle_pod_projection_floor_test", {})
        pfs = " ".join(f"k{k}={e(v)}" for k, v in sorted(pf.items(), key=lambda kv: int(kv[0])))
        print(f"| {cell_of(f)}{'/' + c['tag'] if c.get('tag') else ''} | {c['N']} | {c['k_lat']} | "
              f"{c['rom_substeps']} | {e(r.get('train_traj_rel_mean'))} | "
              f"{e(r['oracle_inferred_latent_test']['traj_rel_mean'])} | {e(r['ic_fit']['rel_mean'])} | "
              f"{e(r['samedt_fom']['traj_rel_vs_fom_mean'])} | {pfs} |")

    print("\n### ROM variants (traj-RMS vs the 80-substep FOM; `samedt` = vs the same-dt Newmark FOM)\n")
    for f, r in rom.items():
        c = r["config"]
        tag = f"{cell_of(f)}" + (f" / {c['tag']}" if c.get("tag") else "")
        print(f"\n**{tag}** — N={c['N']}, K={c['k_lat']}, RS={c['rom_substeps']}, "
              f"{c['n_test']} test trajectories, backend={r['backend']}\n")
        if not r.get("finished"):
            print("> **incomplete run** (the cell had not written its final JSON yet; the "
                  "variants below did finish)\n")
        print("| variant | m | M | traj-RMS | median | max | vs same-dt FOM | incomplete | "
              "E-drift dyn | E-drift kin | E_T/E_0 (dyn) | v defect | iters cold/warm | "
              "ms/step | EQ rel fit | cond(J) |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for label, s in list(r.get("rom", {}).items()) + list(r.get("pod_rom", {}).items()):
            ei = s.get("eq_info"); js = s.get("jac_svals") or {}
            print(f"| `{label}` | {s.get('m', '-')} | {s.get('M') or '-'} | **{e(s['traj_rel_mean'])}** | "
                  f"{e(s['traj_rel_median'])} | {e(s['traj_rel_max'])} | "
                  f"{e(s.get('traj_rel_vs_samedt_fom_mean'))} | {s['n_blowup']}/{s['n_total']} | "
                  f"{e(s.get('energy_dyn_drift_max_mean'))} | {e(s.get('energy_drift_max_mean'))} | "
                  f"{s.get('energy_dyn_final_ratio_mean', float('nan')):.4f} | "
                  f"{e(s.get('v_kin_dyn_defect_max_mean'))} | "
                  f"{s['iters_cold_step0']:.1f}/{s['iters_warm_mean']:.2f} | "
                  f"{s['step_time_ms_median']:.2f} | {e(ei['rel_fit']) if ei else '-'} | "
                  f"{js.get('cond', float('nan')):.1e} |")

    print("\n### Per-time error (snapshot indices 0/10/20/30/40/50)\n")
    for f, r in rom.items():
        c = r["config"]
        tag = f"{cell_of(f)}" + (f"/{c['tag']}" if c.get("tag") else "")
        idx = [0, 10, 20, 30, 40, 50]
        print(f"\n**{tag}**\n")
        print("| arm | " + " | ".join(f"t{i}" for i in idx) + " |")
        print("|---|" + "---|" * len(idx))
        o = r["oracle_inferred_latent_test"]["per_time_mean"]
        print("| oracle held-out latents | " + " | ".join(e(o[i]) for i in idx) + " |")
        for label, s in list(r.get("rom", {}).items()) + list(r.get("pod_rom", {}).items()):
            p = s["per_time_mean"]
            print(f"| `{label}` | " + " | ".join(e(p[i]) for i in idx) + " |")

    print("\n### Timing (median of 7 after 2 warm-ups, block_until_ready, one device per cell)\n")
    for f, r in rom.items():
        c = r["config"]
        t = r.get("timing", {})
        if not t:
            continue
        tag = f"{cell_of(f)}" + (f"/{c['tag']}" if c.get("tag") else "")
        print(f"\n**{tag}** (N={c['N']}, K={c['k_lat']}, RS={c['rom_substeps']}, "
              f"{c['num_steps']*c['rom_substeps']} latent steps)\n")
        print("| what | rollout (ms) | ms/latent step | speedup vs 80-substep FOM | "
              "vs same-dt Newmark FOM | + IC solve (ms) | + 51-slice decode (ms) | end-to-end |")
        print("|---|---|---|---|---|---|---|---|")
        for k, v in t.items():
            ms = 1e3 * v["rollout_s_median"]
            row = [f"`{k}`", f"{ms:.0f}", f"{v.get('ms_per_step', float('nan')):.2f}",
                   f"{v.get('speedup_vs_fom_rollout_only', float('nan')):.2f}x",
                   f"{v.get('speedup_vs_samedt_fom_rollout_only', float('nan')):.2f}x"]
            if "ic_fit_s" in v:
                row += [f"{1e3*v['ic_fit_s']:.1f}", f"{1e3*v['decode_all_slices_s']:.1f}",
                        f"{v.get('speedup_vs_fom_end_to_end', float('nan')):.2f}x"]
            else:
                row += ["-", "-", "-"]
            print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
