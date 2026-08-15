"""Print markdown tables from the ms_*_report.json files (for the README)."""
import glob
import json
import os
import sys

d = sys.argv[1] if len(sys.argv) > 1 else "."

def load(p):
    with open(p) as f:
        return json.load(f)

p = os.path.join(d, "ms_function_report.json")
if os.path.exists(p):
    r = load(p)
    print(f"### (A0) single-function fit, N={r['N']}, hidden {r['hidden']}x{r['layers']}, "
          f"Adam {r['adam_steps']} + L-BFGS {r['lbfgs_steps']}\n")
    print("| stage | n_freq | eps_in | rel residual after | secs |\n|---|---|---|---|---|")
    for s in r["stages"]:
        print(f"| {s['stage']} | {s['n_freq']} | {s['eps_in']:.2e} | "
              f"{s['resid_rel']:.2e} | {s['secs']:.0f} |")
    print()

p = os.path.join(d, "runs/parametric/ms_parametric_report.json")
if os.path.exists(p):
    r = load(p)
    print(f"### (1) parametric decoder, TRUE z given — N={r['N']}, n_train {r['n_train']}, "
          f"{r['steps']} steps/stage, P_SUB {r.get('p_sub', 0)}\n")
    print("| stage | n_freq | eps_in | TRAIN fit rel-RMS | VAL rel-L2 |\n|---|---|---|---|---|")
    for s in r["stages"]:
        print(f"| {s['stage']} | {s['n_freq']} | {s['eps_in']:.2e} | "
              f"{s['train_fit_rel_rms']:.2e} | {s['val_rel_l2']:.2e} |")
    print()

for p in sorted(glob.glob(os.path.join(d, "runs/autodec/ms_autodecoder_K*_report.json"))):
    r = load(p)
    K = r["K_LAT"]
    print(f"### (2) auto-decoder K_LAT={K} — N={r['N']}, n_train {r['n_train']}, "
          f"{r['steps']} steps/stage, P_SUB {r.get('p_sub', 0)}\n")
    print("| stage | n_freq | eps_in | TRAIN fit rel-RMS (manifold floor) | "
          "VAL, stage-0 latents fixed | VAL, latents re-inferred |\n|---|---|---|---|---|---|")
    for s in r["stages"]:
        print(f"| {s['stage']} | {s['n_freq']} | {s['eps_in']:.2e} | "
              f"{s['train_fit_rel_rms']:.2e} | {s['val_rel_l2_stage0_latents']:.2e} | "
              f"{s['val_rel_l2_reinferred']:.2e} |")
    print()
    if r["rom"]:
        print(f"ROM solve (LM Gauss-Newton, {len(r['rom'])} arms):\n")
        print("| colloc | stages | init | ROM rel-L2 mean | med | max | oracle-latent floor | "
              "LM accepted | secs |\n|---|---|---|---|---|---|---|---|---|")
        for w in r["rom"]:
            print(f"| {w['colloc']} | {w['n_stages']} | {w['init']} | "
                  f"{w['rom_rel_l2_mean']:.2e} | {w['rom_rel_l2_med']:.2e} | "
                  f"{w['rom_rel_l2_max']:.2e} | {w['oracle_latent_rel_l2']:.2e} | "
                  f"{w['gn_accepted_med']:.0f} | {w['secs']:.0f} |")
        print()
