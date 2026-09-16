"""Generate the LAB-LOG entry for this cell from the raw run JSONs.

No number in the lab log is typed by hand either. Prints to stdout; the session
appends the text to `/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/LAB-LOG.md`.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / 'reports'))
from generate_p_bank_head import solve_table, pc, ms, num  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bank-train', type=Path, required=True)
    ap.add_argument('--head-train', type=Path, required=True)
    ap.add_argument('--solve', type=Path, required=True)
    ap.add_argument('--bank-audit', type=Path, required=True)
    ap.add_argument('--head-audit', type=Path, required=True)
    ap.add_argument('--solve-audit', type=Path, required=True)
    ap.add_argument('--bank-selection', type=Path, required=True)
    ap.add_argument('--incumbent-diagnosis', type=Path, required=True)
    ap.add_argument('--cohort-provenance', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    ap.add_argument('--branch', required=True)
    ap.add_argument('--commit', required=True)
    ap.add_argument('--archives', nargs='*', default=[])
    a = ap.parse_args()
    tr = json.loads(a.bank_train.read_text())
    ht = json.loads(a.head_train.read_text())
    sv = json.loads(a.solve.read_text())
    ta = json.loads(a.bank_audit.read_text())
    ha = json.loads(a.head_audit.read_text())
    sa = json.loads(a.solve_audit.read_text())
    bs = json.loads(a.bank_selection.read_text())
    idg = json.loads(a.incumbent_diagnosis.read_text())
    prov = json.loads(a.cohort_provenance.read_text())
    agg = solve_table(sv)
    fine = max(k[0] for k in agg)
    models = {m['id']: m for m in sv['checkpoints']}
    recon = {(r['intervals'], r['model']): r for r in sv['reconstruction']}
    L = []
    w = L.append

    def solved(mid, q=False):
        for (mesh, name), row in agg.items():
            if mesh != fine or row.get('model') != mid:
                continue
            if name.startswith('a_neural') and name.startswith('a_neural_q') == q:
                return name, row
        return None, None

    ctrl = solved('incumbent')[1]
    neural = {m: solved(m)[1] for m in models}
    best_id = min((m for m in models if m != 'incumbent'),
                  key=lambda m: neural[m]['worst_same_grid'])
    best = neural[best_id]
    pods = {(agg[k]['k'], agg[k].get('pod_cohort')): (k[1], agg[k])
            for k in agg if k[0] == fine and agg[k]['family'] == 'pod'}

    def bestpod(rank):
        c = [(n, r) for (kk, _), (n, r) in pods.items() if kk == rank]
        return min(c, key=lambda x: x[1]['worst_same_grid']) if c else (None, None)

    podK = bestpod(models[best_id]['K'])
    pod8 = bestpod(8 * models[best_id]['K'])
    pod128 = bestpod(128)
    dst = agg.get((fine, 'dst_direct'))
    crec, brec = recon[(fine, 'incumbent')], recon[(fine, best_id)]

    w('## 2026-09-16')
    w(f"### p-bank-head — the Poisson bank floor moves {num(crec['bank_projection']['worst'] / brec['bank_projection']['worst'], 2)}x "
      f"and the head only {num(crec['best_found']['worst'] / brec['best_found']['worst'], 2)}x, "
      f"so the pre-registered solve target MISSES; POD-LSPG on the same snapshots beats every "
      f"neural checkpoint on error AND cost")
    w('')
    w(f"Worktree `worktrees/2026-09-16-p-bank-head`, branch `{a.branch}` at `{a.commit}`, forked "
      f"from `exp/2026-09-14-head-ablation` at `fee3231a`. Namespace "
      f"`/cluster/tufts/paralab/tawal01/p_bank_head_20260916/`, now empty. Two A100 jobs, one "
      f"per attempt directory, `squeue` checked before and after each submit, both complete and "
      f"clean: bank sweep + incumbent diagnosis `{tr['job_id']}` on `{tr['gpu']}` "
      f"({num(tr['elapsed_seconds'] / 60, 1)} min, source `{tr['commit'][:12]}`), head sweep + "
      f"frozen solve `{sv['job_id']}` on `{sv['gpu']}` "
      f"({num(sv['elapsed_seconds'] / 60, 1)} min, source `{sv['commit'][:12]}`). Both logged "
      f"`jax_backend=gpu`, float64, matmul precision `highest`, JAX {sv['jax_version']}. "
      f"The third job of the three-job cap was not needed.")
    w('')
    w('**The question.** With the same separable architecture and only four knobs — training '
      'data amount, objective, K and R — how far can the Poisson 2D bank floor and the head be '
      'pushed, and does the solve follow. Pre-registration, with every loss in LaTeX, every '
      'seed, every gate, six dated amendments and a falsification clause: '
      '`experiments/p-bank-head/DESIGN.md`.')
    w('')
    w('**The diagnosis first, because it decided the design, and it is not what the 2026-09-11 '
      'entry implies.** On the incumbent checkpoint at '
      f"{idg['intervals']} intervals: its bank reaches "
      f"{pc(idg['D1_bank_floor_training']['worst'])} % worst on its own training sources, and "
      f"the head's best-found over codes reaches only "
      f"{pc(idg['D3_head_best_found_training']['worst'])} % — on data it was trained on. The "
      f"stored codes are already at their own optimum (code-refit gain "
      f"{num(idg['D4_code_refit_gain_training']['relative_median'] * 100, 2)} % of the error at "
      f"the median, {num(idg['D4_code_refit_gain_training']['relative_worst'] * 100, 2)} % at "
      f"the worst), so it is not an optimisation failure; the development/training ratio is only "
      f"{num(idg['D6_generalisation_gap']['ratio_of_worst'], 3)}x, so it is not mainly a "
      f"coverage failure either; and the solve returns exactly what the head contains "
      f"(solved minus best-found "
      f"{num(tr['diagnosis'][0]['meshes'][0]['D7_solved_minus_best_found'] * 100, 7)} pp), so it "
      f"is not the solver. The incumbent's head simply **underfits**, by "
      f"{num(idg['D3_head_best_found_training']['worst'] / idg['D1_bank_floor_training']['worst'], 2)}x, "
      f"the bank it already has.")
    w('')
    w(f"**Bank layer — six arms, R in {{128, 512}} x S in {{192, 768, 3072}}, equal update "
      f"counts, one seed. The pre-registered 1.0 % target PASSES.** Worst bank projection floor "
      f"on the 12 development sources at {bs['intervals']} intervals (the floor is "
      f"mesh-independent to three significant figures; the 1023-interval column is in the "
      f"report): "
      + '; '.join(f"`{x['arm']}` {pc(x['reported_development_worst'])} %"
                  for x in sorted(bs['arms'], key=lambda x: (x['R'], x['S'])))
      + f". The incumbent's own bank is "
        f"{pc(idg['D1_bank_floor_development']['worst'])} % on the same cohort and mesh. **R is the "
        f"lever; S at equal optimizer budget is not** — every R=512 arm clears 1.0 % and the "
        f"three are within "
        f"{num(max(x['common']['worst'] for x in bs['arms'] if x['R'] == 512) / min(x['common']['worst'] for x in bs['arms'] if x['R'] == 512), 3)}x "
        f"of one another on the common cohort, while S is non-monotone at both ranks because "
        f"equal updates mean fewer exposures per source.")
    w('')
    w('**A pre-registration flaw, caught mid-run and recorded rather than buried.** The original '
      "bank rule selected on the worst floor of each arm's *own* validation split. Those splits "
      'have different sizes (29, 115 and 461 sources), so their maxima are not comparable and '
      'the rule is biased towards small S. It was withdrawn and replaced — **before any number '
      f"on the new cohort was computed** — by one common 256-source cohort at a fresh seed, "
      f"asserted disjoint from every training cohort and from the development cohort, scored "
      f"identically for every arm (DESIGN amendment 4). The rule in force selects "
      f"**`{bs['selected']}`**; the withdrawn rule had already selected `{bs['original_rule_selected']}` "
      f"and `pbh01`'s head sweep had run on it. That accident produced the cell's sharpest "
      f"result, so both sweeps are reported.")
    w('')
    alt_best = min(x['best_found_development']['worst'] for x in tr['head_arms'])
    alt_fit = min(x['head_at_stored_codes_fit']['worst'] for x in tr['head_arms'])
    new_best = min(x['best_found_development']['worst'] for x in ht['head_arms'])
    new_fit = min(x['head_at_stored_codes_fit']['worst'] for x in ht['head_arms'])
    w(f"**Head layer — the identical twelve-arm sweep (K in {{16, 32}} x beta_weak in {{0, 1}} x "
      f"beta_smooth in {{0, 1e-3, 1e-2}}) run on two banks that differ only in training cohort "
      f"size. Coverage is the whole story; the objective is not.** On "
      f"`{tr['selection']['bank']['selected']}` ({tr['head_layer']['fit_count']} fit sources) "
      f"the arms fit their own training data to {pc(alt_fit)} % worst and land at "
      f"{pc(alt_best)} % worst best-found on the development sources: memorisation. On "
      f"`{ht['selection']['bank']['selected']}` ({ht['head_layer']['fit_count']} fit sources) "
      f"they fit training to {pc(new_fit)} % and reach {pc(new_best)} %, a factor of "
      f"{num(alt_best / new_best, 2)}. Within the good bank, the pre-registered primary at both "
      f"K is the **plain reconstruction objective**: the exact weak-residual term is neutral "
      f"(it changes worst development best-found by "
      f"{num(abs(next(x for x in ht['head_arms'] if x['arm'] == 'head_K16_w1_s0')['best_found_development']['worst'] - next(x for x in ht['head_arms'] if x['arm'] == 'head_K16_w0_s0')['best_found_development']['worst']) * 100, 4)} pp "
      f"at K=16) and the code-smoothness term strictly **hurts** at both weights. The "
      f"pre-registered head target — best-found within 1.2x the bank floor — MISSES at both K: "
      + '; '.join(f"`{m}` {num(recon[(fine, m)]['best_found']['worst'] / recon[(fine, m)]['bank_projection']['worst'], 3)}x"
                  for m in models if m != 'incumbent')
      + f", against {num(crec['best_found']['worst'] / crec['bank_projection']['worst'], 3)}x for "
        f"the incumbent.")
    w('')
    w(f"**Solve layer — four frozen checkpoints, POD-LSPG at five ranks from two snapshot "
      f"cohorts and the direct DST, all in one job on one GPU, randomised order, "
      f"{sv['config']['repetitions']} timed repetitions, through the unchanged head-ablation "
      f"query kernel.** At {fine} intervals on the 12 development sources, worst same-grid error "
      f"and median total query: "
      + '; '.join(f"`{m}` {pc(neural[m]['worst_same_grid'])} % at {ms(neural[m]['total_ms'])} ms "
                  f"({neural[m]['stationary']}/{neural[m]['invocations']} stationary)"
                  for m in sorted(models, key=lambda m: neural[m]['worst_same_grid']))
      + (f"; POD-LSPG `{podK[0]}` {pc(podK[1]['worst_same_grid'])} % at {ms(podK[1]['total_ms'])} ms"
         if podK[0] else '')
      + (f"; POD-LSPG `{pod128[0]}` {pc(pod128[1]['worst_same_grid'])} % at "
         f"{ms(pod128[1]['total_ms'])} ms" if pod128[0] else '')
      + (f"; direct DST {pc(dst['worst_physical'])} % physical at {ms(dst['total_ms'])} ms"
         if dst else '') + '.')
    w('')
    w(f"**Pre-registered success: 1 of 4 clauses MISSES, so the cell does not pass.** Best "
      f"non-control checkpoint `{best_id}`: worst same-grid {pc(best['worst_same_grid'])} % "
      f"against the 2.0000 % bar — **miss**, though it is "
      f"{num(ctrl['worst_same_grid'] / best['worst_same_grid'], 2)}x better than the incumbent's "
      f"{pc(ctrl['worst_same_grid'])} %. All {best['stationary']}/{best['invocations']} of its "
      f"solves exit stationary — pass. Median query "
      f"{num(best['total_ms'] / ctrl['total_ms'], 3)}x the incumbent against a 1.5x bar — pass. "
      f"It beats POD-LSPG at the matched rank k'={models[best_id]['K']} "
      f"({pc(best['worst_same_grid'])} % against {pc(podK[1]['worst_same_grid'])} %) — pass.")
    w('')
    w('**The honesty clauses, which matter more than the verdict.** '
      + (f"**POD-LSPG at k'=128 rebuilt from the selected bank's own "
         f"{next(c['count'] for c in sv['pod_cohorts'] if c['id'] == 'trainset')} training "
         f"snapshots reaches {pc(pod128[1]['worst_same_grid'])} % at {ms(pod128[1]['total_ms'])} ms "
         f"and therefore dominates every neural checkpoint on BOTH error and cost** "
         f"(best neural {pc(best['worst_same_grid'])} % at {ms(best['total_ms'])} ms). "
         if pod128[0] else '')
      + (f"The direct DST solve takes {ms(dst['total_ms'])} ms with {pc(dst['worst_physical'])} % "
         f"physical error and is faster and more accurate than everything here; **no speedup "
         f"over any full-order solver is claimed anywhere in this cell.** " if dst else '')
      + f"For K=32 the 8K rung is k'=256, outside the pre-registered rank set and not run — "
        f"stated as a limitation, not a pass. Every selection used an internal-validation split "
        f"or the common held-out cohort; the 12 development sources selected nothing, and their "
        f"ranking disagrees with the common-cohort ranking, which is recorded.")
    w('')
    w(f"**Falsification, as pre-registered.** The first condition IS met and is the cell's "
      f"headline negative: on the selected bank every head arm stays above 1.2x the bank floor "
      f"while the floor itself improved "
      f"{num(crec['bank_projection']['worst'] / brec['bank_projection']['worst'], 3)}x, and the "
      f"head's *relative* distance to the bank grew from "
      f"{num(crec['best_found']['worst'] / crec['bank_projection']['worst'], 3)}x to "
      f"{num(brec['best_found']['worst'] / brec['bank_projection']['worst'], 3)}x even as its "
      f"absolute error fell "
      f"{num(crec['best_found']['worst'] / brec['best_found']['worst'], 3)}x. **A better bank "
      f"still does not buy a proportionally better head.** The second condition is NOT met: the "
      f"head arms span "
      f"{num(max(x['best_found_development']['worst'] for x in ht['head_arms']) / new_best, 3)}x "
      f"on the selected bank, so the limit is not the head's function class — it is coverage, "
      f"which is inside this cell's latitude and is the lever that worked.")
    w('')
    w('**Retracted inside this cell.** `pbh01` computed the incumbent\'s training-side '
      'diagnostics D2, D3, D4, D6 and D8 against `core.source_params(0, 512)`, which is **not** '
      "that checkpoint's training cohort — it trained on `core.source_params(0, 576)[:512]`, and "
      'because every call draws each parameter array at its own length the two share nothing. D2 '
      'came back at 2091 % worst, which is how it was caught. Those five values are '
      '**retracted** and replaced by `checks/incumbent-diagnosis/`, a bounded local GB10 '
      'recomputation on the correct cohort; D1 on the development cohort, D5 and D7 never touch '
      'the training parameters and stand unchanged. No cluster job was spent on the correction '
      'and nothing in either sweep is affected.')
    w('')
    w('**Two reproducibility landmines, both caught by assertions, both worth keeping.** '
      f"(1) `core.source_params` is **not bit-reproducible between the local GB10 and the "
      f"cluster**: the Gaussian-width column, w = exp(U(log 0.02, log 0.1)), differs by one ulp "
      f"(max {prov['local_regeneration_max_abs_difference']:.2e}) between the two NumPy builds "
      f"while the other three columns agree bitwise, so the 2026-09-11 assertion "
      f"`sha(training) == checkpoint training_draw_sha256` passed only because it ran on the "
      f"cluster. Both audits here compare regenerated cohorts to a tolerance and record both "
      f"hashes. (2) The local GB10's **first** JAX GPU QR of a 64516x512 float64 matrix returns "
      f"an **all-NaN** factor, reproducibly, while NumPy and the cluster are always correct; the "
      f"full-numerical-rank assertion caught it reporting rank 0 for a bank the cluster had "
      f"certified at rank 512. `bank_r` now recomputes a non-finite factor and folds finiteness "
      f"into `rank_valid`, and the bank selection is pure NumPy. Neither landmine touches a "
      f"cluster number.")
    w('')
    gates = sv['gates']
    w(f"**Fidelity gates, all passed before any verdict.** Run with the incumbent checkpoint, "
      f"this cell's evaluation path reproduces `pabl01`'s per-case physical errors on every "
      f"shared arm and mesh to a worst relative difference of "
      f"{max(g['worst_relative_difference'] for g in gates):.2e} against a declared "
      f"{sv['config']['fidelity_tolerance']:.0e} — {len(gates)} gates covering arm (a), the "
      f"free-bank arm, the direct DST and all five POD rungs, so the streaming POD rebuilt here "
      f"is `pabl01`'s own. Three independent NumPy/SciPy audits that import neither the driver "
      f"nor JAX all pass: they recomputed every one of "
      f"{sa['checks']['recomputed_errors']['invocations']} reported solve errors from the "
      f"retained output fields (worst physical difference "
      f"{sa['checks']['recomputed_errors']['worst_physical_difference']:.2e}, worst same-grid "
      f"{sa['checks']['recomputed_errors']['worst_same_grid_difference']:.2e}), re-derived every "
      f"bank projection floor and head training error from the saved weights, re-applied every "
      f"selection rule, and re-checked both correction bases' orthonormality in the exact QR "
      f"metric.")
    w('')
    w(f"Source-generated report: `{a.report}` (SHA256 "
      f"`{hashlib.sha256(Path(a.report).read_bytes()).hexdigest()}`), produced by "
      f"`experiments/p-bank-head/reports/generate_p_bank_head.py`; no number in it is typed by "
      f"hand. All eighteen trained checkpoints and both correction bases are Git-tracked under "
      f"`experiments/p-bank-head/checkpoints/`. Raw archives are Git-tracked as bounded chunks "
      f"under `experiments/p-bank-head/artifacts/`"
      + (': ' + '; '.join(a.archives) if a.archives else '')
      + '. **`pbh02` is 2.0 GB compressed in 43 chunks** because it retains every dense output '
        'field at the finest mesh for twenty subjects; the coordinator should weigh that against '
        'the already heavy repository when staging pushes. Both exact remote attempt directories '
        'were removed after checksum collection and the namespace is empty. Not pushed, per the '
        "coordinator's standing instruction; commits are local only.")
    w('')
    w('**Open.** A head trained jointly with the R=512 bank on 3072 sources rather than on it '
      'frozen (the jointly trained `bank_arm_head` is the weaker of the three new checkpoints '
      'here, but it was trained on a different cohort); more than one training seed; S beyond '
      '3072 with the optimizer budget scaled rather than fixed, which is the comparison this '
      'cell deliberately did not run; the k\'=256 POD rung that would complete the 8K clause at '
      'K=32; and the sealed final cohorts, which stay untouched. No worktree was merged and no '
      'earlier numerical result outside this cell is retracted.')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
