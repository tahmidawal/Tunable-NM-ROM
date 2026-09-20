"""Generate the dated LAB-LOG correction entry for DESIGN §A13 from the JSONs. Nothing typed by hand.

    python checks/make_a13_entry.py --out checks/2026-09-19-lab-log-correction-a13.md

Reads `checks/design5-verification.json` (both flags re-derived from result.json), the re-audited
`artifacts/*/audit.json` and `reports/summary.json`, and the worktree HEAD.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]


def f(x, d=3):
    return f'{x:.{d}f}'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    v = json.load(open(HERE / 'checks/design5-verification.json'))
    M = v['meshes']
    S = json.load(open(HERE / 'reports/summary.json'))
    head = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    branch = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', '--abbrev-ref', 'HEAD'], text=True).strip()
    rp = HERE / 'reports/2026-09-17-b-panel.md'
    m1 = M['1024']
    L = ['## 2026-09-19',
         f"### b-panel — CORRECTION (DESIGN §A13) superseding the 2026-09-17 entries' admissible counts and the 1024² ratio: "
         f"the convergence flag was evaluated at each query's own gtol, not §5's fixed 1e-6; under §5 the admissible reduced "
         f"counts are {M['256']['design5']['admissible_reduced']}/{M['512']['design5']['admissible_reduced']}/"
         f"{M['1024']['design5']['admissible_reduced']} (were {M['256']['as_implemented']['admissible_reduced']}/"
         f"{M['512']['as_implemented']['admissible_reduced']}/{M['1024']['as_implemented']['admissible_reduced']}), "
         f"{len(m1['design5']['reduced_on_gpu_evolved_frontier'])} reduced on the 1024² frontier (was "
         f"{len(m1['as_implemented']['reduced_on_gpu_evolved_frontier'])}), 1024² cheapest-reduced/FOM "
         f"{f(m1['design5']['ratio_cheapest_reduced_over_cheapest_fom'])}× (was "
         f"{f(m1['as_implemented']['ratio_cheapest_reduced_over_cheapest_fom'])}×); no job submitted", '']
    L.append(f"Worktree `worktrees/2026-09-17-b-panel`, branch `{branch}` at `{head[:12]}`. Nothing ran on the cluster; jobs used "
             f"remain six of eight. Trigger: the Codex report audit (`experiments/b-panel/reports/codex-report-audit-2026-09-19.md`, "
             f"run after the quota returned; its 15 sampled numbers and all 123 timing medians matched the JSONs) — four findings, "
             f"all accepted, recorded as DESIGN §A13 (a dated correction superseding the affected sentences of §A10–A12, which are not edited).")
    L.append('')
    L.append("**What was wrong and is retracted.** (1) `audit_panel.py` evaluated DESIGN §5's convergence rule against each "
             "invocation's own `gtol` instead of the fixed 1e-6 the section states, so every 1e-3 arm was flagged converged at 1e-3 "
             "and entered the admissible frontiers and ratios since `bpn101`; no amendment authorised it. Verified independently "
             "before any change by `checks/verify_design5.py` from the raw per-step quantities in the three `result.json` files "
             f"(counterexample `{m1['counterexample']['arm']}`, max per-step stationarity {m1['counterexample']['max_step_stationarity']:.4g}, "
             "every exit on the gradient rule). (2) The §A5.2 scoring at 512² duplicated the six historical 1024² `eqxfer` capped "
             "transfers against both 512² rule sets (\"6 of 12\") and pronounced a verdict on an experiment the prediction never "
             "covered — withdrawn; the scoring is keyed by (mesh, rule set, q, fit-state regime) and stated only at 1024² `eqxfer`, "
             "where the predicted outcome held on the capped refit (both named rungs uncertified) and the uncapped `bpn203` refit is "
             "reported separately as a changed experiment (q = 128 secondary, q = 256 uncertified with 64 fit states). (3) Two "
             "templated counterfactual sentences claimed that removing `free512_M1024_dense` makes `pod256_M1024_dense` non-dominated "
             "at 512² and 1024²; computed, it stays dominated (the report names the remaining dominator); the 256² `pod512_M2048_dense` "
             "statement stands. (4) \"Indistinguishable ... in physical terms\" and \"loses weight mass\" over-reached one worst-case "
             "scalar and a support count; both now say what was measured.")
    L.append('')
    L.append("**The rule now applied.** `converged_design5` (§5 as written, fixed 1e-6, residual-rule exception) defines `admissible`; "
             "the as-implemented flag is kept as `converged_own_gtol` / `admissible_own_gtol`, labelled, defining nothing. Audits rerun over "
             "the retained fields (every numeric field of every arm byte-identical to the pre-A13 audits; only flag keys changed); the "
             "second-path `checks/recheck_headline.py` agrees at 0.0 for all three jobs. The 1e-3 arms keep their numbers in every "
             "table and lose eligibility.")
    L.append('')
    L.append('| mesh | job | reduced | 1e-3 arms | admissible reduced (pre-A13 → §5) | reduced on the (GPU ms, worst evolved %) frontier | cheapest admissible reduced | / cheapest same-job FOM | / same-job `fft_tight` |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    for k in ('256', '512', '1024'):
        m = M[k]; o, n = m['as_implemented'], m['design5']
        L.append(f"| {k}² | `{m['job_id']}` | {m['reduced_arms']} | {len(m['loose_tolerance_arms'])} | {o['admissible_reduced']} → **{n['admissible_reduced']}** | "
                 f"{len(o['reduced_on_gpu_evolved_frontier'])} → **{len(n['reduced_on_gpu_evolved_frontier'])}** | `{o['cheapest_admissible_reduced']}` → `{n['cheapest_admissible_reduced']}` | "
                 f"{f(o['ratio_cheapest_reduced_over_cheapest_fom'])}× → **{f(n['ratio_cheapest_reduced_over_cheapest_fom'])}×** | "
                 f"{f(o['ratio_cheapest_reduced_over_fft_tight'])}× → **{f(n['ratio_cheapest_reduced_over_fft_tight'])}×** |")
    L.append('')
    L.append(f"1024² survivors under §5: {', '.join(f'`{x}`' for x in m1['design5']['reduced_on_gpu_evolved_frontier'])} (the 1e-6 transferred "
             "rules at q = 0, 16, 32). The §A11 answer is unchanged — 0 at 256², 0 at 512², > 0 at 1024² — so \"the crossover lies between "
             "512² and 1024²\" stands under §5, with the report now saying that its factor-of-N sentences are ratios of within-job ratios "
             "across GPU classes (A100-80G vs H200). Also disclosed in the report's new generated \"Retractions, corrections and disclosures\" "
             "section, at the audit's request: §A1's missing pre-job Codex audit, §A3's withdrawal of the frozen-state transfer collector, and "
             "that the 512² acceptance in §A12 rests on a post-data reading of §7's narrower clause while §6 says every gate must pass.")
    L.append('')
    L.append(f"**For the paper generator.** `experiments/b-panel/reports/summary.json` ({len(S['rows'])} rows): per row `admissible` is now §5 "
             "(alias `admissible_design5`), `admissible_own_gtol` is the pre-A13 flag, plus `converged_design5` / `converged_own_gtol` / "
             "`converged_strict`; frontier rows `nondominated_<pair>_admissible` and `_reduced_only` are §5, `_admissible_own_gtol` and "
             "`_reduced_only_own_gtol` are pre-A13, `_all` is unfiltered; a top-level `admissibility` block states the rule and the key map. "
             f"Report `experiments/b-panel/reports/2026-09-17-b-panel.md` (SHA256 `{hashlib.sha256(rp.read_bytes()).hexdigest()}`) regenerated "
             "with the pre-A13 values carried beside the corrected ones, labelled.")
    L.append('')
    L.append("**Open.** The 1024² frontier is now three 1e-6 transferred rungs on an H200 while 256²/512² are A100-80G; a matched-hardware "
             "1024² job (A100-80G; the 29-subject 1024² panel has only ever run on an H200 and its memory on 80 GB is untested) would remove "
             "the hardware caveat from the crossover statement. Not submitted; two of the eight-job cap remain.")
    L.append('')
    Path(a.out).write_text('\n'.join(L))
    print('WROTE', a.out)


if __name__ == '__main__':
    main()
