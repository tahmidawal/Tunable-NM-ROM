"""Markdown tables from cn_poisson_*/cn_burgers_* report JSONs (rejects
incomplete reports unless --partial).  Usage: python cn_summarize.py <dir> [--partial]"""
import glob, json, os, sys

d = sys.argv[1] if len(sys.argv) > 1 else "."
partial = "--partial" in sys.argv
out = ["# Cascade NM-ROM summary tables (machine-generated)\n"]
for p in sorted(glob.glob(os.path.join(d, "**", "cn_*_report.json"), recursive=True)):
    r = json.load(open(p))
    if not r.get("complete") and not partial:
        out.append(f"\n_skipped incomplete: {p}_\n"); continue
    out.append(f"\n## {os.path.relpath(p, d)}  (complete={r.get('complete')})\n")
    out.append("\n### stages\n| stage | eps_in | n_freq | train global | train mean | held-out (encoded) global | mean |\n|---|---|---|---|---|---|---|")
    for s in r["stages"]:
        ho = s.get("heldout_encoded_global_rel", s.get("heldout_t0_encoded_global_rel"))
        hm = s.get("heldout_encoded_mean_rel_l2", s.get("heldout_t0_encoded_mean_rel_l2"))
        out.append(f"| {s['stage']} | {s['eps_in']:.2e} | {s['n_freq']} | {s['train_global_rel']:.3e} | "
                   f"{s['train_mean_rel_l2']:.3e} | {ho:.3e} | {hm:.3e} |")
    out.append("\n### residual probes (before next stage)\n| after stage | eff_rank | pod8 err | pod32 err | nn1 corr | nn5 corr | stop? |\n|---|---|---|---|---|---|---|")
    for q in r.get("probes", []):
        out.append(f"| {q['after_stage']} | {q['eff_rank']:.1f} | {q.get('pod8_rel_err', float('nan')):.2e} | "
                   f"{q.get('pod32_rel_err', float('nan')):.2e} | {q['nn1_corr']:.2f} | {q['nn5_corr']:.2f} | {q['stop_suggested']} |")
    if r.get("inferred"):
        out.append("\n### held-out finite-budget inferred latents (data misfit LM)\n| stages | init encoded | init mean | best |\n|---|---|---|---|")
        for q in r["inferred"]:
            out.append(f"| {q['n_stages']} | {q['init_encoded_mean_rel_l2']:.3e} | {q['init_mean_mean_rel_l2']:.3e} | {q['best_of_starts_mean_rel_l2']:.3e} |")
    if r.get("rom"):
        if r["config"].get("pde") == "poisson2d":
            out.append("\n### ROM (held-out, init = E(f))\n| objective | colloc | stages | ROM mean | med | max | encoded plug-in | inferred | r_lm | r_enc | acc/rej |\n|---|---|---|---|---|---|---|---|---|---|---|")
            for q in r["rom"]:
                out.append(f"| {q['objective']} | {q['colloc']} | {q['n_stages']} | {q['rom_rel_l2_mean']:.3e} | {q['rom_rel_l2_med']:.3e} | "
                           f"{q['rom_rel_l2_max']:.3e} | {q['encoded_plugin_rel_l2_mean']:.3e} | {q['inferred_latent_rel_l2_mean']:.3e} | "
                           f"{q['resid_lm_med']:.2e} | {q['resid_encoded_med']:.2e} | {q['lm_accepted_med']:.0f}/{q['lm_rejected_med']:.0f} |")
        else:
            out.append("\n### ROM rollout (held-out, z0 = E(u0))\n| stages | mean all-t | final-t | t0 encoded | inferred-latent all-t | acc/step |\n|---|---|---|---|---|---|")
            for q in r["rom"]:
                out.append(f"| {q['n_stages']} | {q['rom_rel_l2_mean_all_t']:.3e} | {q['rom_rel_l2_final_t']:.3e} | {q['encoded_t0_rel_l2']:.3e} | "
                           f"{q['inferred_latent_rel_l2_mean_all_t']:.3e} | {q['lm_accepted_med_per_step']:.0f} |")
open(os.path.join(d, "SUMMARY_TABLES.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
