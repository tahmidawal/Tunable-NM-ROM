"""Markdown tables from runs/*/hlat_rom_*.json (+ stage-1 JSONs)."""
import glob, json, os, sys
import numpy as np

def f(x): return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.2e}"
rows = []
cells = sorted(glob.glob("runs/ad_*/hlat_rom_*.json"))
print("## Stage 2 — latent-stepping ROM (held-out TEST_SEED trajectories)\n")
for c in cells:
    r = json.load(open(c)); cfg = r["config"]; K = cfg["ad_config"]["k_lat"]; N = cfg["N"]
    log = open(glob.glob(os.path.dirname(c) + "/*.out")[0]).read()
    gpu = [l for l in log.splitlines() if l.startswith("host=")][0].split("gpu=")[1]
    fom = r["timing"]["fom_rollout"]["rollout_s_median"]
    print(f"### N={N}, K={K}, n_test={cfg['n_test']}  ({gpu}; job "
          f"{os.path.basename(glob.glob(os.path.dirname(c)+'/*.out')[0]).split('.')[0]})\n")
    print(f"auto-decoder TRAIN recon {f(r['train_rel_mean'])} · ORACLE inferred-latent floor (held-out) {f(r['oracle_inferred_latent_test']['traj_rel_mean'])} · IC-fit misfit (u0, cold start) {f(r['ic_fit']['rel_mean'])} (jit LM {f(r['ic_fit_jit']['rel_mean'])}) · max FOM rel residual {r['max_fom_rel_residual']:.1e} · FOM rollout {fom*1e3:.0f} ms\n")
    print("| variant (solver:colloc:objective) | M | m | traj rel-L2 mean | median | max | blow-ups | iters cold / warm | step ms | rollout ms | speedup (rollout) | end-to-end speedup (py IC / jit IC) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for v, s in r["rom"].items():
        t = r["timing"].get(v, {})
        print(f"| `{v}` | {s.get('M') or '—'} | {s['m']} | {f(s['traj_rel_mean'])} | {f(s['traj_rel_median'])} | {f(s['traj_rel_max'])} | {s['n_blowup']}/{s['n_total']} | {s['iters_cold_step0']:.1f} / {s['iters_warm_mean']:.1f} | {s['step_time_ms_median']:.1f} | {t.get('rollout_s_median',float('nan'))*1e3:.0f} | {t.get('speedup_vs_fom_rollout_only',float('nan')):.2f}x | {t.get('speedup_vs_fom_end_to_end_py_ic',float('nan')):.2f}x / {t.get('speedup_vs_fom_end_to_end_jit_ic',float('nan')):.2f}x |")
    if r.get("pod_direct"):
        print("\nDirect reduced POD-Galerkin (k x k solve per step, the production linear ROM):\n")
        print("| k | traj rel-L2 mean | median | max | rollout ms | speedup vs FOM |")
        print("|---|---|---|---|---|---|")
        for kk, v in r["pod_direct"].items():
            t = r["timing"].get(f"pod_direct_{kk}", {})
            print(f"| {kk[1:]} | {f(v['traj_rel_mean'])} | {f(v['traj_rel_median'])} | "
                  f"{f(v['traj_rel_max'])} | {t.get('rollout_s_median',float('nan'))*1e3:.1f} | "
                  f"{t.get('speedup_vs_fom_rollout_only',float('nan')):.1f}x |")
    print("\nPOD control (same solver), projection floors " + ", ".join(f"k{k}={f(v)}" for k, v in r["oracle_pod_projection_floor_test"].items()) + ":\n")
    print("| k | variant | traj rel-L2 mean | median | iters warm | step ms | rollout ms | speedup |")
    print("|---|---|---|---|---|---|---|---|")
    for kv, s in r["pod_rom"].items():
        k, var = kv.split(":", 1)
        t = r["timing"].get(f"pod_{k}:{var}", {})
        print(f"| {k[1:]} | `{var}` | {f(s['traj_rel_mean'])} | {f(s['traj_rel_median'])} | {s['iters_warm_mean']:.1f} | {s['step_time_ms_median']:.1f} | {t.get('rollout_s_median',float('nan'))*1e3:.0f} | {t.get('speedup_vs_fom_end_to_end',float('nan')):.2f}x |")
    pt = r["oracle_inferred_latent_test"]["per_time_mean"]
    print("\nper-time (t-index 0/10/20/30/40/50): oracle " + " / ".join(f(pt[i]) for i in (0,10,20,30,40,50)))
    for v in ("lspg:full:fd", "lspg:full:weak64", "lspg:eq256:weak64", "lspg:eqoff256:weak64", "lspg:full:weakc64"):
        if v in r["rom"]:
            p = r["rom"][v]["per_time_mean"]; print(f"; `{v}` " + " / ".join(f(p[i]) for i in (0,10,20,30,40,50)))
    print()
print("\n## Stage 1 — space-time LSPG with the (z,t) sweep decoder (N=64, 16 test, budget 100)\n")
print("| IC_W | arm | traj rel-L2 mean | median | max | |z-z*| |")
print("|---|---|---|---|---|---|")
for arm, lab in (("icw_sqrt50", "sqrt(50)"), ("icw1", "1")):
    r = json.load(open(f"runs/s1_n64/{arm}/hlat_stage1_N64.json"))
    for k in ("ic", "resid", "both"):
        x = r[k]; print(f"| {lab} | `{k}` | {f(x['traj_rel_mean'])} | {f(x['traj_rel_median'])} | {f(x['traj_rel_max'])} | {f(x['z_err_mean'])} |")
print(f"\noracle (true z) {f(r['oracle_true_z']['traj_rel_mean'])} mean / {f(r['oracle_true_z']['traj_rel_max'])} max; FOM trajectory residual through the FOM residual {r['fom_traj_rel_res']:.1e}; ours-vs-FOM residual max|diff| {r['ours_vs_fom_traj_res_maxabs_diff']:.1e}; decoder residual at true z (rel) {f(r['resid_at_true_z_rel'])}")
