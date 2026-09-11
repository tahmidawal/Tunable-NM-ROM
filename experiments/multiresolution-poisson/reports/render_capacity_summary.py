"""Render accepted capacity and training-only correction evidence from JSON."""
import argparse,json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);a=ap.parse_args();r=a.run;p=json.loads((r/'panel.json').read_text());d=json.loads((r/'head-correction-diagnostic.json').read_text());audit=json.loads((r/'audit.json').read_text());assert audit['passed']
    lines=['# Audited Poisson bank-capacity comparison','','These are accepted measurements on already-opened development cases. No neural endpoint meets complete-cohort physical eligibility; bank and online targets remain separate.','','| Cohort | Intervals | Method | GPU ms | Host ms | Worst field error (%) | Invalid | Physical target passes all |','| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |']
    for group in p['groups']:
        for mesh in group['meshes']:
            for name,m in mesh['methods'].items():
                if group['group']!='all' and name not in p['config']['trained_model_ids']:continue
                lines.append(f"| {group['group']} | {mesh['intervals']} | {name} | {m['gpu_median_ms']:.9f} | {m['host_median_ms']:.9f} | {100*m['worst_relative_error']:.9f} | {m['invalid_invocations']} | {m['all_cases_pass_target']} |")
    lines+=['','The `new_development` group label is retained for continuity: these later cases were already opened before capacity selection. Neither group is fresh independent confirmation for this study.','','Full-bank projections and bounded stationary full-field head fits remain diagnostic; all values and per-case decompositions are in `panel.json`. Every timing row includes its corresponding full field from the same invocation.','','| Saved head | Correction directions | Training residual energy captured (%) | Worst training reconstruction (%) |','| --- | ---: | ---: | ---: |']
    for model in d['models']:
        for q in model['correction_panels']:lines.append(f"| {model['model']} | {q['correction_directions']} | {100*q['normalized_inside_bank_energy_captured']:.9f} | {100*q['worst_fixed_code_corrected_error']:.9f} |")
    lines+=['','Correction directions are constructed only from normalized training residuals in the physical QR metric. The table uses saved training latent codes with analytically fitted linear corrections and does not measure an online PDE solve. The nonlinear training codes are not independently certified optimal fits. No measured online gain or speed gain for corrections is claimed.','','Glossary: intervals counts mesh subdivisions per axis; cohort identifies original versus later opened development cases; GPU/host ms are pooled medians of equal-count repetitions; worst field error is current-relative full-field L2 error versus restricted fine truth; invalid means a failed numerical solver gate; physical eligibility includes the declared error and refinement gates; bank is the learned spatial feature span; head maps latent coordinates to feature coefficients; QR gives the exact physical field metric; residual energy is averaged over normalized training snapshots; correction directions are nested fixed linear coefficient vectors learned offline; CG/DST are iterative/direct Poisson controls.']
    (r/'summary.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
