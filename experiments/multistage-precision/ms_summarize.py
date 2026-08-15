"""Markdown tables from the report JSONs.  Refuses (marks) incomplete runs."""
import glob, json, os, sys
d = sys.argv[1] if len(sys.argv) > 1 else "."
load = lambda p: json.load(open(p))

def cfgline(c):
    return (f"N={c['N']}, n_train {c['n_train']}, n_val {c['n_val']}, {c['steps']} Adam steps/stage, "
            f"batch {c['batch']}, P_SUB {c['p_sub']}, hidden {c['hidden']}x{c['n_layers']}, "
            f"const_lr {c['const_lr']}, lbfgs {c['lbfgs_steps']}, z_ff {c['z_ff']}, seed {c['seed']}")

p = os.path.join(d, "ms_function_report.json")
if os.path.exists(p):
    r = load(p)
    print(f"### (A0) single-function fit — N={r['N']}, tanh {r['hidden']}x{r['layers']}, Adam {r['adam_steps']} + L-BFGS {r['lbfgs_steps']}\n")
    print("| stage | n_freq | eps_in | rel residual after | secs |\n|---|---|---|---|---|")
    for s in r["stages"]:
        print(f"| {s['stage']} | {s['n_freq']} | {s['eps_in']:.2e} | {s['resid_rel']:.2e} | {s['secs']:.0f} |")
    print()

for p in sorted(glob.glob(os.path.join(d, "runs/**/ms_parametric_report.json"), recursive=True)):
    r = load(p); c = r["config"]
    flag = "" if r.get("complete") else "  **INCOMPLETE — not for citation**"
    print(f"### (1) true-z parametric decoder — {os.path.dirname(p)}{flag}\n{cfgline(c)}\n")
    print("| stage | n_freq | f_d (cyc) | eps_in | TRAIN global / mean-rel | VAL global / mean-rel | VAL by amp quartile (low..high) | final batch loss |\n|---|---|---|---|---|---|---|---|")
    for s in r["stages"]:
        q = " / ".join(f"{v:.1e}" for v in s["val_rel_l2_by_amp_quartile"])
        print(f"| {s['stage']} | {s['n_freq']} | {s['f_d_cyc']:.1f} | {s['eps_in']:.2e} | "
              f"{s['train_global_rel']:.2e} / {s['train_mean_rel_l2']:.2e} | "
              f"{s['val_global_rel']:.2e} / {s['val_mean_rel_l2']:.2e} | {q} | {s['adam_final_batch_loss']:.1e} |")
    print()

for p in sorted(glob.glob(os.path.join(d, "runs/**/ms_autodecoder_K*_report.json"), recursive=True)):
    r = load(p); c = r["config"]; K = c["K_LAT"]
    flag = "" if r.get("complete") else f"  **INCOMPLETE (phase {r.get('phase')}) — not for citation**"
    print(f"### (2) auto-decoder K_LAT={K} — {os.path.dirname(p)}{flag}\n{cfgline(c)}; "
          f"LM budget {c['gn_iters']} attempts, m_eq {c['m_eq']}, n_test {c['n_test']}\n")
    print("| stage | n_freq | eps_in | TRAIN global / mean-rel (learned latents) | VAL stage-0 latents fixed | VAL LM-inferred best / mean-start / nearest-start | VAL by amp quartile |\n|---|---|---|---|---|---|---|")
    for s in r["stages"]:
        li = s["val_lm_inferred_mean_rel_l2"]
        q = " / ".join(f"{v:.1e}" for v in s["val_lm_inferred_by_amp_quartile"])
        print(f"| {s['stage']} | {s['n_freq']} | {s['eps_in']:.2e} | {s['train_global_rel']:.2e} / {s['train_mean_rel_l2']:.2e} | "
              f"{s['val_fixed_stage0_latents_mean_rel_l2']:.2e} | {li['best']:.2e} / {li['mean']:.2e} / {li['nearest']:.2e} | {q} |")
    if r["stages"] and "val_adam_inferred_mean_rel_l2" in r["stages"][0]:
        print(f"\n(stage-0 Adam-inferred val latents, secondary: {r['stages'][0]['val_adam_inferred_mean_rel_l2']:.2e})")
    print()
    if r["rom"]:
        print("ROM (LM Gauss-Newton on the ghost-zero FD residual, held-out sources):\n")
        print("| colloc | stages | init | ROM rel-L2 mean / med / max | oracle (same start) | oracle best-of | ‖r‖ LM / oracle / ‖f‖ | bnd block | acc/rej | reasons | z-norm LM / oracle | NN-lat dist |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
        for w in r["rom"]:
            print(f"| {w['colloc']} | {w['n_stages']} | {w['init']} | {w['rom_rel_l2_mean']:.2e} / {w['rom_rel_l2_med']:.2e} / {w['rom_rel_l2_max']:.2e} | "
                  f"{w['oracle_rel_l2_mean']:.2e} | {w['oracle_best_of_starts_rel_l2_mean']:.2e} | "
                  f"{w['resid_lm_med']:.1e} / {w['resid_oracle_med']:.1e} / {w['f_norm_med']:.1e} | {w['boundary_block_lm_med']:.1e} | "
                  f"{w['lm_accepted_med']:.0f}/{w['lm_rejected_med']:.0f} | {w['lm_reasons']} | {w['z_norm_med']:.2f} / {w['z_norm_oracle_med']:.2f} | {w['z_nn_dist_med']:.2f} |")
        print()

p = os.path.join(d, "ms_diag_report.json")
if os.path.exists(p):
    r = load(p)
    print("### diag — residual smoothness in the conditioning variable (whitened NN corr) vs fields\n")
    print("| object | NN1 corr (true z) | NN5 corr (true z) | NN1 corr (latent) | spectral centroid (cyc/unit) | global rel |\n|---|---|---|---|---|---|")
    for k, v in r.items():
        if k == "config": continue
        print(f"| {k} | {v.get('nn1_corr', v.get('nn1_corr_in_true_z', float('nan'))):.3f} | "
              f"{v.get('nn5_corr', v.get('nn5_corr_in_true_z', float('nan'))):.3f} | "
              f"{v.get('nn1_corr_in_latent', float('nan')):.3f} | {v.get('spec_centroid', float('nan')):.2f} | "
              f"{v.get('global_rel', float('nan')):.2e} |")
