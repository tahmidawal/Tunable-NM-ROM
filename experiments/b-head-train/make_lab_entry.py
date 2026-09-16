"""Generate this cell's LAB-LOG.md entry from the audited JSONs. Nothing is typed."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def f(x, n=4):
    return '--' if x is None else f'{x:.{n}f}'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-result', required=True)
    p.add_argument('--train-audit', required=True)
    p.add_argument('--eval-result', required=True)
    p.add_argument('--eval-audit', required=True)
    p.add_argument('--report', required=True)
    p.add_argument('--report-sha256', required=True)
    p.add_argument('--branch-commit', required=True)
    p.add_argument('--archives', default='', help='name=sha256:chunks, comma separated')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    tr = json.loads(Path(a.train_result).read_text())
    ta = json.loads(Path(a.train_audit).read_text())
    ev = json.loads(Path(a.eval_result).read_text())
    ea = json.loads(Path(a.eval_audit).read_text())
    rows = [x for x in ea['arm_table'] if x['kind'] == 'rom']
    foms = [x for x in ea['arm_table'] if x['kind'] != 'rom']
    inc = next(x for x in ea['arm_table'] if x['arm'] == ev['config']['incumbent_arm'])
    sel = tr['selection']
    crit = ea['checks']['success_criteria_evaluated']['detail']['criteria']
    passing = [v for v in ea['verdicts'] if v['success']]
    quad = {x['arm']: x['quadrature'] for x in ea['arm_table']}
    incq = quad.get(ev['config']['incumbent_arm'])
    trained = [v for v in ea['verdicts']
               if v['checkpoint'] != 'incumbent' and quad.get(v['arm']) == incq]
    best = min(trained or ea['verdicts'], key=lambda v: v['best_found_percent'])
    o = []
    w = o.append

    w('## 2026-09-16')
    w(f"### b-head-train — training alone {'DOES' if passing else 'does NOT'} close the Burgers "
      f"head's gap: the best trained checkpoint reaches "
      f"best-found {best['best_found_percent']:.4f} % against the incumbent's "
      f"{next(v for v in ea['verdicts'] if v['checkpoint'] == 'incumbent')['best_found_percent']:.4f} % "
      f"and a {f(inc['worst_bank_projection_percent'])} % bank floor; the diagnostic says "
      f"{sel['diagnostic_verdict']}")
    w('')
    w("The coordinator asked whether training alone -- same separable architecture, no new head or "
      "bank families -- can move the Burgers head's best-found reconstruction from 2.54 % toward "
      "the 0.39 % bank floor at 256 intervals, and whether the online solve follows. Three levers, "
      "each isolated with the others at the incumbent recipe, then the best combination: data "
      "density, training objective (a solve-aware weak-residual term, a two-code trajectory term, "
      "and a code-smoothness term), and capacity (K, and conditionally the bank rank R). Every "
      "trained checkpoint is frozen, hashed, and evaluated through the UNCHANGED head-ablation arm "
      "(a) machinery, with the incumbent re-run in the same job as the control. Predeclared "
      "protocol and its dated amendments: `experiments/b-head-train/DESIGN.md`.")
    w('')
    w(f"Worktree `worktrees/2026-09-16-b-head-train`, branch `exp/2026-09-16-b-head-train` at "
      f"`{a.branch_commit}`, forked from `exp/2026-09-14-head-ablation` at `fee3231a`. Namespace "
      f"`/cluster/tufts/paralab/tawal01/b_head_train_20260916/`, one attempt directory per job. "
      f"Training job `{tr.get('job_id')}` on `{tr.get('gpu')}` (source `{tr.get('commit')}`, "
      f"elapsed {tr['elapsed_seconds']:.1f} s); evaluation job `{ev.get('job_id')}` on "
      f"`{ev.get('gpu')}` (source `{ev.get('commit')}`, elapsed {ev['elapsed_seconds']:.1f} s). "
      f"Both printed `jax_backend=gpu`, ran float64 with matmul precision "
      f"`{ev['matmul_precision']}`, and were checksum-collected, independently NumPy-audited and "
      f"Git-archived before their exact remote attempt directories were removed. "
      f"{ev['config']['reps']} timed repetitions with GPU burn-in before every block, randomised "
      f"subject order, every repetition array retained.")
    w('')
    w('**A correction to the brief that matters for every density number below.** The brief called '
      '128 trajectories "the incumbent". 128 is the head-ablation *configuration*\'s snapshot draw '
      '(`config-ablation.json: train_trajectories = 128`), used there for POD bases, the '
      'linear/quadratic map fits and the empirical-quadrature rule -- not for training the head. '
      'The incumbent checkpoint\'s own training set is 4608 trajectories: the canonical '
      '`sample_params(seed=0)` draw of 576 plus 4032 appended from seed 1000 (job 2837431, '
      '`runs/dn256b`). `burgers2d_film.sample_params` and `engines.params_draw` are the same '
      'sequential RNG draw over the same ranges, so that set is reproducible in this lane, and the '
      'density ladder is a NESTED PREFIX of it at '
      + str([n for n, _ in sel['density_curve']]) + ' trajectories. Density is then the only '
      'variable across the rungs and the top rung is the incumbent\'s own data.')
    w('')
    # the single most consequential comparison, computed rather than typed
    top = int(max(n for n, _ in sel['density_curve']))
    qsuf = '_' + ev['config']['incumbent_arm'].split('_')[-1]
    lname = f'd{top}k{tr["K_incumbent"]}rec'
    verd = {v['arm']: v for v in ea['verdicts']}
    lfl, iv = verd.get(lname + qsuf), verd[ev['config']['incumbent_arm']]
    lrow = next((r for r in tr['arms'] if r['arm'] == lname), None)
    ick = next((c for c in ev['checkpoints'] if c['arm'] == 'incumbent'), None)
    if lfl and lrow and ick:
        w(f"**THE HEAD GAP IS NOT A SHORTAGE OF DATA.** `{lfl['arm']}` is the like-for-like "
          f"retrain: the same {top} trajectories, the same K = {tr['K_incumbent']}, R = "
          f"{lrow['R']}, head {tr['config']['head_hidden']} wide x "
          f"{tr['config']['head_layers']} layers, the same {tr['config']['steps']}-step budget, "
          f"lr {tr['config']['lr']}, batch {tr['config']['batch']} that the incumbent's own refit "
          f"used. It reaches {lfl['best_found_percent']:.4f} % best-found against the incumbent's "
          f"{iv['best_found_percent']:.4f} % -- "
          f"{lfl['best_found_percent'] / iv['best_found_percent']:.2f}x WORSE. Given the "
          'incumbent\'s own data this pipeline does not recover the incumbent, so adding data is '
          f"not the lever. One difference is recorded in the two configurations: this arm expands "
          f"those trajectories into {lrow['states']} auto-decoder codes (stride "
          f"{tr['config']['state_stride']}) where the incumbent checkpoint carries {ick['codes']} "
          f"-- {lrow['states'] / ick['codes']:.2f}x more states at the SAME step budget, so each "
          'state gets that factor less optimisation. Budget-per-state and the fixed step budget '
          'are the two candidates this cell can see for the remaining distance; the next '
          'experiment should vary them at fixed data.')
        w('')
    w('**Gates.** ' + '; '.join(
        [f"whitening round trip {tr['gates']['whitening_round_trip']:.3e} (< 1e-10)",
         f"identity (*) against regenerated fields {tr['gates']['identity_star_relative']:.3e} (< 1e-9)"]
        + ([f"`{ev['config']['incumbent_arm']}` reproduces abl01's "
            f"`{ev['gates']['incumbent_reproduces_abl01']['reference']}` on "
            f"{ev['gates']['incumbent_reproduces_abl01']['compared']} cases to "
            f"{ev['gates']['incumbent_reproduces_abl01']['worst_relative_delta']:.3e} "
            f"(declared {ev['gates']['incumbent_reproduces_abl01']['tolerance']:g})"]
           if ev.get('gates', {}).get('incumbent_reproduces_abl01') else [])
        + [f"every recorded error recomputed from the saved fields by NumPy to "
           f"{ea['checks']['recorded_errors_recomputed_from_saved_fields']['detail']:.3e}",
           f"every same-grid discrepancy to "
           f"{ea['checks']['same_grid_recomputed_from_saved_fields']['detail']:.3e}"]) + '.')
    w('Audit checks, both jobs: ' + ', '.join(
        f"`{k}` {'pass' if v['passed'] else '**FAIL**'}"
        for k, v in list(ta['checks'].items()) + list(ea['checks'].items())) + '.')
    w('')
    w('**The trained checkpoints** (held-out representation oracle on '
      f"{tr['data']['holdout']['snapshots']} states of "
      f"{tr['config']['holdout_trajectories']} trajectories from seed "
      f"{tr['config']['holdout_seed']} that no arm ever fitted; span floor "
      f"{100 * tr['data']['holdout_span_floor']['mean']:.4f} % mean):")
    w('')
    w('| arm | traj | $K$ | $R$ | bank | objective | recon (train) mean | held-out mean % | '
      'held-out worst % | mean-code-only mean % | train GPU h |')
    w('|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|')
    for t in ta['arm_table']:
        w('| `' + t['arm'] + '` | ' + ' | '.join([
            str(t['trajectories']), str(t['K']), str(t['R']), t['bank'], t['objective'],
            ('--' if t['recon_train_mean'] is None else f"{t['recon_train_mean']:.4e}"),
            f"{100 * t['holdout_mean']:.4f}", f"{100 * t['holdout_max']:.4f}",
            f"{100 * t['holdout_mean_only']:.4f}", f"{t['train_gpu_hours']:.3f}"]) + ' |')
    w('')
    w(f"**Data-vs-capacity diagnostic: {sel['diagnostic_verdict']}.** At fixed $K$ = "
      f"{tr['K_incumbent']} the held-out oracle worst along "
      f"{[n for n, _ in sel['density_curve']]} is {[round(v, 6) for _, v in sel['density_curve']]}; "
      f"monotone decreasing: {'yes' if sel['monotone'] else 'no'}; relative improvement "
      f"{sel['saturation_rung']} -> {sel['next_rung']} is "
      f"{100 * sel['relative_gain_mid_to_next']:.2f} % and {sel['saturation_rung']} -> "
      f"{sel['top_rung']} is {100 * sel['relative_gain_mid_to_top']:.2f} %, against the declared "
      f"{100 * sel['saturation_fraction']:.0f} % saturation threshold. Selected: density "
      f"{sel['best_density']}, $K$ {sel['best_K']}, objective `{sel['best_objective']}`, "
      f"combination arm `{sel['best_combination_arm']}`. Bank-rank arms actually run: "
      f"{sel.get('bank_rank_arms')}.")
    w('')
    w('**The three layers at 256 intervals**, six development cases, primary metric the same-grid '
      'discrepancy against the same-job converged full-order solve:')
    w('')
    w('| arm | $K$ | $R$ | quad | bank floor % | best-found % | best-found $t{=}0$ % | '
      'solved same-grid % | solved evolved % | solved $t{=}0$ % | worst reference % | '
      'budget exits | completed | median GPU ms | cost vs incumbent |')
    w('|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|')
    for x in rows:
        w('| `' + x['arm'] + '` | ' + ' | '.join([
            str(x['K']), str(x['R']), x['quadrature'], f(x['worst_bank_projection_percent']),
            f(x['worst_best_found_percent']), f(x['worst_best_found_t0_percent']),
            f(x['worst_same_grid_percent']), f(x['worst_same_grid_evolved_percent']),
            f(x['worst_t0_percent']), f(x['worst_reference_percent']), str(x['budget_exits']),
            'yes' if x['all_completed'] else 'NO', f(x['median_gpu_ms'], 3),
            f(x['median_gpu_ms'] / inc['median_gpu_ms'], 3)]) + ' |')
    for x in foms:
        w('| `' + x['arm'] + '` (FOM) | - | - | - | - | - | - | '
          + f(x['worst_same_grid_percent']) + ' | ' + f(x['worst_same_grid_evolved_percent'])
          + ' | ' + f(x['worst_t0_percent']) + ' | ' + f(x['worst_reference_percent'])
          + ' | - | - | ' + f(x['median_gpu_ms'], 3) + ' | - |')
    w('')
    w(f"**Pre-registered success** (best-found worst < {crit['best_found']} %, solved worst "
      f"same-grid < {crit['same_grid']} %, every solve stationary and completed, cost <= "
      f"{crit['cost_factor']}x the incumbent): "
      + (f"**{len(passing)} of {len(ea['verdicts'])} arms PASS** ("
         + ', '.join(f"`{v['arm']}`" for v in passing) + ').'
         if passing else f"**no arm passes**, out of {len(ea['verdicts'])} evaluated."))
    w('')
    w('**Recorded deviations.** The density ladder is a nested prefix of the incumbent\'s own draw '
      'rather than three independent `params_draw(0,n)` draws (recorded deviation D1 in DESIGN.md), '
      'and a 4608 rung was added so one arm is a like-for-like retrain of the incumbent. The joint '
      'bank+head arms cannot use identity (*) and train against a seeded subset of '
      f"{tr['config']['joint_points']} of the {(tr['config']['intervals'] - 1) ** 2} interior "
      'points, at a capped density, continuing from the selected frozen-bank arm; their '
      'orthonormality weight is calibrated at the warm start rather than inherited, and their '
      'held-out oracle uses the mean-code initialisation only, so the mean-code-only column is the '
      'like-for-like one across the two families. Amendments A1-A3 in DESIGN.md are dated and were '
      'all made before any evaluation number existed; the first training submission (`3745663`) was '
      'CANCELLED while pending, consumed no GPU time and produced no output, because two defects in '
      'the joint arms were found after submission. The weak-residual training terms use the exact '
      'dense advection rather than the empirical-quadrature rule, because that rule is fitted at a '
      'fixed head and would drift as theta moves.')
    w('')
    jw = {r['arm']: r['train'] for r in tr['arms'] if r['spec']['bank'] == 'joint'}
    bad = {k: v for k, v in jw.items() if v['data_term_at_warm_start'] > 1e-2}
    if bad:
        w('**RETRACTED AS A RANK TEST (amendment A4).** '
          + ', '.join(f'`{k}`' for k in bad)
          + ' did not inherit the warm start it was supposed to continue from: `widen` keeps the '
            'incumbent feature columns but initialises the new head output columns at random, so '
            'the widened decoder starts from a data term of '
          + ', '.join(f"{v['data_term_at_warm_start']:.3e}" for v in bad.values())
          + ' where its narrow sibling starts from '
          + ', '.join(f"{v['data_term_at_warm_start']:.3e}" for k, v in jw.items() if k not in bad)
          + '. Because lambda_orth is calibrated at the warm start, it was then set against that '
            'large value and dominated the run. The wide-rank arm is reported with its numbers but '
            'it says nothing about bank rank; a fair rank arm needs a warm-start-preserving '
            'widening and a penalty calibrated after it. Not rerun: the job cap reserved the '
            'remaining job for the evaluation.')
        w('')
    ulp = ta['checks'].get('archived_draws_reproduce_locally_to_one_ulp')
    if ulp is not None:
        w('**Three audit gates failed on a byte hash and were replaced by a stronger value check '
          '(amendment A5).** The cluster NumPy and the local NumPy disagree by one unit in the last '
          'place in np.exp, and the viscosity column is the only one that passes through it; a '
          'direct probe shows their default_rng streams are otherwise identical. '
          + ' '.join(f"The {k} draw differs in {v['rows_differing']} rows, column(s) "
                     f"{v['columns_differing']}, at most {v['max_ulp']:.0f} ULP."
                     for k, v in ulp['detail'].items())
          + ' The attempt\'s own draws are now regenerated in the cluster interpreter, accepted '
            'only because each hashes to exactly the value the job recorded, committed as '
            'artifacts/<attempt>/draws.npz, and checked against a local re-derivation to <= 1 ULP, '
            'against the declared ranges, and for cohort disjointness on the actual values. The '
            'evaluation cohort is asserted BITWISE equal to abl01\'s own recorded cases in the job '
            'and again in the audit. No reported number moves: 1 ULP in nu is O(1e-16) relative.')
        w('')
    w(f"**The objective weights are fixed and declared, but their calibration point is weaker than "
      f"DESIGN.md said, and this is the cell's main recorded flaw.** The declared rule is "
      f"beta = rho * L_rec / L_term 'at the incumbent head'. The driver evaluated the incumbent "
      f"head at {tr['weights']['calibration_states']} of its own training codes paired with the "
      f"same number of UNRELATED snapshots' targets: the "
      f"incumbent's codes index its own dense state pick and the driver strided both arrays "
      f"independently instead of joining them on it. The join was recoverable -- the checkpoint "
      f"carries `hfit_pick`, the global state ids its codes belong to, in exactly the "
      f"trajectory-major order this lane uses -- and was not used. Realised values: "
      f"L_rec {tr['weights']['L_rec']:.6g}, L_weak {tr['weights']['L_weak']:.6g}, "
      f"L_smooth {tr['weights']['L_smooth']:.6g}, beta_w = beta_t = "
      f"{tr['weights']['beta_w']:.6g}, gamma {tr['weights']['gamma']:.6g}. Because the weights are "
      f"a RATIO of two terms evaluated at the same arbitrary point, the arms stay well defined at "
      f"fixed reported weights; what is NOT licensed is the claim that the added term contributes a "
      f"tenth of the loss at the incumbent's own fit. The realised share of the loss each added "
      f"term held at the END of training is in the report's training table, so a weight that turned "
      f"out to be uninformative is visible rather than hidden.")
    w('')
    w('**A compile-time note worth carrying forward.** `common.make_projector` jits a closure over '
      'the cached bank, so XLA reported a large captured constant during the first projection '
      'compile. At this bank size that costs compile time only and the job ran at full speed, but '
      'the same pattern at a larger bank or a finer mesh is the captured-constant landmine '
      '`CLAUDE.md` warns about; the bank should be an explicit jit argument if this code is reused '
      'there.')
    w('')
    w(f"Source-generated report: `{a.report}` (SHA256 `{a.report_sha256}`) with its generator "
      f"beside it. Raw archives Git-tracked as bounded chunks under "
      f"`experiments/b-head-train/artifacts/`"
      + (': ' + a.archives if a.archives else '') + '. The trained checkpoints are inside the '
      'training archive. Both exact remote attempt directories were removed after checksum '
      'collection and the namespace is empty. Not pushed, per the coordinator\'s standing '
      'instruction; commits are local only.')
    w('')
    w('**Open.** Other meshes, other checkpoints, more than one training seed per arm, the sealed '
      'final cohort, and a step budget chosen per density rather than held fixed across the ladder. '
      'No earlier numerical result is retracted and no worktree was merged.')
    w('')
    Path(a.out).write_text('\n'.join(o) + '\n')
    print(a.out)


if __name__ == '__main__':
    main()
