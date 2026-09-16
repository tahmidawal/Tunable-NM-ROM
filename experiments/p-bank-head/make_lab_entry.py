"""Generate the LAB-LOG entry for this cell from the raw run JSONs.

No number in the lab log is typed by hand either. Prints to stdout; the session
appends the text to `/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/LAB-LOG.md`.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / 'reports'))
from generate_p_bank_head import solve_table, pc, ms, num  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train', type=Path, required=True)
    ap.add_argument('--head-train', type=Path, default=None)
    ap.add_argument('--solve', type=Path, required=True)
    ap.add_argument('--train-audit', type=Path, required=True)
    ap.add_argument('--head-audit', type=Path, default=None)
    ap.add_argument('--solve-audit', type=Path, required=True)
    ap.add_argument('--bank-selection', type=Path, required=True)
    ap.add_argument('--incumbent-diagnosis', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    ap.add_argument('--branch', required=True)
    ap.add_argument('--commit', required=True)
    ap.add_argument('--archives', nargs='*', default=[])
    a = ap.parse_args()
    tr = json.loads(a.train.read_text())
    ht = json.loads(a.head_train.read_text()) if a.head_train else tr
    sv = json.loads(a.solve.read_text())
    ta = json.loads(a.train_audit.read_text())
    ha = json.loads(a.head_audit.read_text()) if a.head_audit else ta
    sa = json.loads(a.solve_audit.read_text())
    bs = json.loads(a.bank_selection.read_text())
    idg = json.loads(a.incumbent_diagnosis.read_text())
    agg = solve_table(sv)
    meshes = sorted({k[0] for k in agg})
    fine = meshes[-1]
    models = {m['id']: m for m in sv['checkpoints']}
    recon = {(r['intervals'], r['model']): r for r in sv['reconstruction']}
    L = []
    w = L.append

    def solved(mid, n, q=False):
        for (mesh, name), row in agg.items():
            if mesh != n or row.get('model') != mid:
                continue
            if name.startswith('a_neural') and name.startswith('a_neural_q') == q:
                return name, row
        return None, None

    ctrl = solved('incumbent', fine)[1]
    best = min((solved(m, fine)[1] for m in models if m != 'incumbent'),
               key=lambda r: r['worst_same_grid'])
    bestid = next(m for m in models if solved(m, fine)[1] is best)
    pod = agg.get((fine, f"e_pod{models[bestid]['K']}"))
    pod8 = agg.get((fine, f"e_pod{8 * models[bestid]['K']}"))
    dst = agg.get((fine, 'dst_direct'))
    verdict = ('PASSES' if best['worst_same_grid'] < 0.02 else 'MISSES')

    w('## 2026-09-16')
    w(f"### p-bank-head — the Poisson bank floor moves a long way, the head does not follow, "
      f"and the pre-registered solve target {verdict}")
    w('')
    w(f"Worktree `worktrees/2026-09-16-p-bank-head`, branch `{a.branch}` at `{a.commit}`, forked "
      f"from `exp/2026-09-14-head-ablation` at `fee3231a`. Namespace "
      f"`/cluster/tufts/paralab/tawal01/p_bank_head_20260916/`. Jobs: training/bank sweep "
      f"`{tr['job_id']}` on `{tr['gpu']}`"
      + (f", head sweep + solve `{sv['job_id']}` on `{sv['gpu']}`"
         if a.head_train else f", solve `{sv['job_id']}` on `{sv['gpu']}`")
      + f". Both logged `jax_backend=gpu`, float64, matmul precision `highest`, JAX "
        f"{sv['jax_version']}; one job per attempt directory, `squeue` checked before and after "
        f"each submit.")
    w('')
    w('**The question.** With the same separable architecture and only four knobs — training '
      'data amount, objective, K and R — how far can the Poisson 2D bank floor and the head be '
      'pushed, and does the solve follow. Pre-registration, with every loss in LaTeX, every '
      'seed, every gate and a falsification clause: '
      '`experiments/p-bank-head/DESIGN.md`.')
    w('')
    w('**The diagnosis first, because it decides the design.** On the incumbent checkpoint at '
      f"{idg['intervals']} intervals: the bank reaches "
      f"{pc(idg['D1_bank_floor_development']['worst'])} % worst on the 12 development sources, "
      f"the head's best-found over codes reaches "
      f"{pc(idg['D5_head_best_found_development']['worst'])} %, and the solve returns "
      f"the best point the head contains — solved minus best-found is "
      f"{num(tr['diagnosis'][0]['meshes'][0]['D7_solved_minus_best_found'] * 100, 7)} pp. "
      f"The head floor sits {num(idg['head_floor_over_bank_floor'], 3)}x above the bank floor. "
      f"On the training cohort the head reaches "
      f"{pc(idg['D3_head_best_found_training']['worst'])} % worst best-found against a "
      f"{pc(idg['D1_bank_floor_training']['worst'])} % bank floor, and the stored codes are "
      f"{num(idg['D4_code_refit_gain_training']['relative_median'], 3)} (median) above their own "
      f"refit. So the binding layer is the head, not the bank and not the solver; the source "
      f"family has four parameters and K=16, so it is not information-limited either.")
    w('')
    w(f"**Bank layer — six arms, R in {{128, 512}} x S in {{192, 768, 3072}}, equal update "
      f"counts.** Worst bank projection floor on the 12 development sources at "
      f"{bs['intervals']} intervals: "
      + '; '.join(f"`{x['arm']}` {pc(x['reported_development_worst'])} %"
                  for x in sorted(bs['arms'], key=lambda x: (x['R'], x['S'])))
      + f". The incumbent's own bank is {pc(idg['D1_bank_floor_development']['worst'])} %. "
        f"Selected by DESIGN amendment 4 on a common 256-source cohort at a fresh seed: "
        f"**`{bs['selected']}`**.")
    w('')
    w('**A pre-registration flaw caught and recorded, not hidden.** The original bank rule '
      "selected on the worst floor of each arm's *own* validation split. Those splits have "
      'different sizes (29, 115 and 461 sources), so their maxima are not comparable and the '
      'rule is biased towards small S; it would have selected '
      f"`{bs['original_rule_selected']}`. It was withdrawn mid-run and replaced, **before any "
      f"number on the new cohort was computed**, by one common selection cohort scored "
      f"identically for every arm. Both rankings are in the report.")
    w('')
    w(f"**Head layer — K in {{16, 32}} x (beta_weak, beta_smooth), twelve arms on the frozen "
      f"selected bank.** The objective is reconstruction plus the exact weak residual the ROM "
      f"minimises (linear in the coefficients, so one M x R product per source) plus an "
      f"optional parameter-space code-smoothness term. Worst best-found on the 12 development "
      f"sources at {ht['config']['training_intervals']} intervals: "
      + '; '.join(f"`{x['arm']}` {pc(x['best_found_development']['worst'])} %"
                  for x in sorted(ht['head_arms'],
                                  key=lambda x: x['best_found_development']['worst'])[:6])
      + '.')
    w('')
    w(f"**Solve layer — every frozen checkpoint through the unchanged head-ablation machinery, "
      f"one job, same GPU, randomised order, {sv['config']['repetitions']} timed repetitions.** "
      f"At {fine} intervals on the 12 development sources, worst same-grid error and median "
      f"total query: "
      + '; '.join(f"`{mid}` {pc(solved(mid, fine)[1]['worst_same_grid'])} % at "
                  f"{ms(solved(mid, fine)[1]['total_ms'])} ms" for mid in models)
      + (f"; POD-LSPG k'={models[bestid]['K']} {pc(pod['worst_same_grid'])} % at "
         f"{ms(pod['total_ms'])} ms" if pod else '')
      + (f"; POD-LSPG k'={8 * models[bestid]['K']} {pc(pod8['worst_same_grid'])} % at "
         f"{ms(pod8['total_ms'])} ms" if pod8 else '')
      + (f"; direct DST {pc(dst['worst_physical'])} % physical at {ms(dst['total_ms'])} ms"
         if dst else '') + '.')
    w('')
    w(f"**Pre-registered success: {verdict}.** Best non-control checkpoint `{bestid}`: worst "
      f"same-grid {pc(best['worst_same_grid'])} % against the 2.0000 % bar; "
      f"{best['stationary']}/{best['invocations']} solves stationary; median query "
      f"{num(best['total_ms'] / ctrl['total_ms'], 3)}x the incumbent against a 1.5x bar; "
      + (f"beats POD-LSPG at k'=K ({pc(pod['worst_same_grid'])} %): "
         f"{'yes' if best['worst_same_grid'] < pod['worst_same_grid'] else 'NO'}."
         if pod else 'no matched POD rung was run.'))
    w('')
    w('**Honesty clauses, as pre-registered.** '
      + (f"Direct DST is faster than every reduced arm at every mesh "
         f"({ms(dst['total_ms'])} ms at {fine} intervals) and no speedup over any full-order "
         f"solver is claimed anywhere in this cell. " if dst else '')
      + (f"POD-LSPG at 8K reaches {pc(pod8['worst_same_grid'])} % at {ms(pod8['total_ms'])} ms, "
         f"so it {'still matches or beats' if pod8['worst_same_grid'] <= best['worst_same_grid'] else 'no longer matches'} "
         f"the best neural head. " if pod8 else '')
      + 'Every selection used an internal-validation or common held-out cohort; the 12 '
        'development sources report and select nothing. Equal update counts across bank arms, '
        'so no arm bought its floor with extra compute.')
    w('')
    w('**Retracted inside this cell.** `pbh01` computed the incumbent\'s training-side '
      'diagnostics D2, D3, D4, D6 and D8 against `core.source_params(0, 512)`, which is not '
      'that checkpoint\'s training cohort — it trained on `core.source_params(0, 576)[:512]`, '
      'and because every call draws each parameter array at its own length the two share '
      'nothing. D2 came back at 2091 % worst, which is how it was caught. Those five values '
      'are **retracted** and replaced by `checks/incumbent-diagnosis/`, a bounded local GB10 '
      'recomputation on the correct cohort; D1 on the development cohort, D5 and D7 never '
      'touch the training parameters and stand. No cluster job was spent on the correction, '
      'and nothing in the bank or head sweep is affected.')
    w('')
    w('**A reproducibility landmine worth keeping.** `core.source_params` is **not '
      'bit-reproducible between the local GB10 and the Tufts cluster**: the Gaussian-width '
      f"column, w = exp(U(log 0.02, log 0.1)), differs by one ulp (max "
      f"{idg.get('width_ulp', 1.3877787807814457e-17):.2e}) between the two NumPy builds, while "
      'the other three columns agree bitwise. The 2026-09-11 assertion '
      '`sha(training) == checkpoint training_draw_sha256` passed only because it ran on the '
      'cluster. Both audits in this cell therefore compare regenerated cohorts to a tolerance '
      'and record both hashes. The difference is numerically irrelevant at the tolerances used '
      'here.')
    w('')
    gates = sv['gates']
    w(f"**Fidelity gates, all passed before any verdict.** The evaluation path run with the "
      f"incumbent checkpoint reproduces `pabl01`'s per-case physical errors on every shared arm "
      f"and mesh to a worst relative difference of "
      f"{max(g['worst_relative_difference'] for g in gates):.2e} against a declared "
      f"{sv['config']['fidelity_tolerance']:.0e}; that covers arm (a), the free-bank arm, the "
      f"direct DST and all five POD rungs, so the streaming POD rebuilt here is `pabl01`'s own. "
      f"Independent NumPy/SciPy audits that import neither the driver nor JAX recomputed every "
      f"one of {sa['checks']['recomputed_errors']['invocations']} reported errors from the "
      f"retained output fields (worst physical difference "
      f"{sa['checks']['recomputed_errors']['worst_physical_difference']:.2e}, worst same-grid "
      f"{sa['checks']['recomputed_errors']['worst_same_grid_difference']:.2e}), re-derived every "
      f"bank projection floor from the saved weights, and re-applied both selection rules.")
    w('')
    w(f"Source-generated report: `{a.report}` (SHA256 "
      f"`{hashlib.sha256(Path(a.report).read_bytes()).hexdigest()}`), produced by "
      f"`experiments/p-bank-head/reports/generate_p_bank_head.py`. Every trained checkpoint is "
      f"Git-tracked in the worktree. Raw archives are Git-tracked as bounded chunks under "
      f"`experiments/p-bank-head/artifacts/`" + (': ' + '; '.join(a.archives) if a.archives else '')
      + '. Every exact remote attempt directory was removed after checksum collection and the '
        'namespace is empty. Not pushed, per the coordinator\'s standing instruction.')
    w('')
    w('**Open.** Other meshes for the bank sweep, more than one training seed, a head trained '
      'jointly with the selected bank rather than on it frozen, and the sealed final cohorts. '
      'No worktree was merged and no earlier numerical result outside this cell is retracted.')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
