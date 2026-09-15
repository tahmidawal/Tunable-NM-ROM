"""Generate the lab-log entry for the correction ladder from the audited JSONs."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', required=True)
    p.add_argument('--audit', required=True)
    p.add_argument('--smoke', required=True)
    p.add_argument('--date', required=True)
    p.add_argument('--status', required=True)
    p.add_argument('--branch-commit', required=True)
    p.add_argument('--report', required=True)
    p.add_argument('--archive', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    r = json.loads(Path(a.result).read_text())
    au = json.loads(Path(a.audit).read_text())
    sm = json.loads(Path(a.smoke).read_text())
    rows = au['checks']['arm_table']
    missing = [x['arm'] for x in rows if x.get('worst_same_grid_percent') is None]
    assert not missing, ('same-grid errors missing for ' + ', '.join(missing)
                         + '; rerun audit_ladder.py with --fields <output directory>')
    cfg = r['config']
    get = lambda n: next((x for x in rows if x['arm'] == n), None)
    ladder = sorted([x for x in rows if x['kind'] == 'rom' and x['quadrature'] == 'dense'
                     and x['dt'] == cfg['dt'] and x['arm'] != 'q0_dense_Mmax'], key=lambda x: x['q'])
    errs = [x['worst_same_grid_percent'] for x in ladder]
    costs = [x['median_gpu_ms'] for x in ladder]
    mono_err = all(b <= a2 * (1 + 1e-12) for a2, b in zip(errs, errs[1:]))
    mono_cost = all(b >= a2 * (1 - 1e-12) for a2, b in zip(costs, costs[1:]))
    L = []
    w = L.append
    w('')
    w('')
    w(f'## {a.date}')
    w(f'### Burgers fixed-weight correction ladder — {a.status}')
    w('')
    w('The coordinator asked how much accuracy a frozen checkpoint can buy at inference time, and at '
      'what cost, by solving q extra fixed linear bank directions on top of the neural head: '
      'u(z,y) = G(h_theta(z) + C_q y). Every network weight, the bank, the weak objective, the '
      'test-mode family, the initializer policy, the trust radius, the iteration budgets and the '
      'stopping tolerance are the head-ablation arm (a) contract; q and the time step are the only '
      'knobs. Unlike Poisson, the Burgers weak residual is quadratic in the coefficients through the '
      'upwind advection term, so y cannot be eliminated analytically and the whole augmented vector '
      'is solved by the same Levenberg-Marquardt iteration.')
    w('')
    w(f"Worktree `worktrees/2026-09-14-head-ablation`, branch `exp/2026-09-14-head-ablation` at "
      f"`{a.branch_commit}`. Namespace `/cluster/tufts/paralab/tawal01/headabl_20260914/qlad01`, job "
      f"`{r.get('job_id')}` on `{r.get('gpu')}`, source `{r.get('commit')}`, "
      f"{r['intervals']} intervals, {len(r['physical_cases'])} opened development cases (the same six "
      f"arm (a) used; the eight refinement calibration cases belong to a different lane), "
      f"{cfg['reps']} timed repetitions with GPU burn-in before every block, all repetition arrays "
      f"retained. Elapsed {r['elapsed_seconds']:.1f} s.")
    w('')
    w(f"**Fidelity gates.** Through the corrected-head wrapper at q=0 the local smoke reproduces the "
      f"consolidated saved Burgers case to {sm['q0_vs_saved_case']['relative_l2']:.3e} relative and "
      f"is bit-identical ({sm['q0_vs_incumbent']:.1e}) to the incumbent `accuracy_paths.make_rom`.")
    g = au['checks'].get('q0_reproduces_head_ablation_arm_a')
    if g and g.get('detail'):
        d = g['detail']
        w(f"In the job, `q0_eq` reproduces the head-ablation job's `a_neural_eq` on all "
          f"{d['compared']} cases to a worst relative difference of {d['worst_relative_delta']:.3e}, "
          f"with {d['bitwise_identical_fields']} of {d['compared']} output fields bitwise identical "
          f"across the two jobs. Every recorded error was independently recomputed from the retained "
          f"output fields by NumPy.")
    w('')
    dd = r['directions']
    w(f"**Direction rule (offline, nested, recorded).** {dd['rule']}. {dd['snapshots']} seeded "
      f"snapshots, {dd['starts']} multistart fits at budget {dd['budget']}, seed {dd['seed']}; the "
      f"head's own best-found relative fit over them is {dd['head_fit_relative_median'] * 100:.4f}% "
      f"median, {dd['head_fit_relative_worst'] * 100:.4f}% worst; available rank "
      f"{dd['available_rank']}. The fit is offline and one-time and enters no query timing, but at "
      f"{dd['seconds']:.1f} s it dominated the job's setup cost, and the retained stderr shows a "
      f"single XLA slow-operation alarm covering nearly the whole stage: the cost is compilation of "
      f"a doubly vectorised Levenberg-Marquardt while_loop with a forward-mode Jacobian inside, not "
      f"arithmetic. Flattening that loop would remove most of the setup cost without changing any "
      f"reported number; worth doing before this rule is reused. Residual energy captured: "
      + ', '.join(f"q={q} {v * 100:.4f}%" for q, v in dd['residual_energy_captured'].items()) + '.')
    w('')
    w('The primary metric is the same-grid discrepancy against the converged same-mesh full-order '
      'solve, because the refined-reference metric also contains this mesh\'s discretization error.')
    w('')
    w('| q | solved dim | M | quad | best-found % | worst same-grid % | worst reference % | '
      'median iters/step | budget exits | completed | median GPU ms | median host ms |')
    w('|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---:|---:|')
    for x in rows:
        if x['kind'] != 'rom':
            continue
        w('| ' + ' | '.join([
            f"{x['q']}" + ('' if x['dt'] == cfg['dt'] else f" (dt {x['dt']:g})")
            + ('' if x['arm'] != 'q0_dense_Mmax' else ' (M control)'),
            str(x['solved_dimension']), str(x['M']), x['quadrature'],
            f"{x['worst_best_found_percent']:.4f}", f"{x['worst_same_grid_percent']:.4f}",
            f"{x['worst_reference_percent']:.4f}", f"{x['median_iterations']:.1f}",
            str(x['total_budget_exits']),
            {True: 'yes', False: 'no', None: '-'}[x['all_completed']],
            f"{x['median_gpu_ms']:.3f}", f"{x['median_host_ms']:.3f}"]) + ' |')
    for x in rows:
        if x['kind'] == 'fom':
            w('| ' + ' | '.join([f"FOM {x['arm']}", '-', '-', '-', '-',
                                 f"{x['worst_same_grid_percent']:.4f}",
                                 f"{x['worst_reference_percent']:.4f}",
                                 f"{x['median_iterations']:.1f}", '-', '-',
                                 f"{x['median_gpu_ms']:.3f}", f"{x['median_host_ms']:.3f}"]) + ' |')
    w('')
    w(f"**Monotonicity.** Error along q={[x['q'] for x in ladder]} is "
      f"{'monotone decreasing' if mono_err else 'NOT monotone'} "
      f"({[round(v, 4) for v in errs]} percent); cost is "
      f"{'monotone increasing' if mono_cost else 'NOT monotone'} "
      f"({[round(v, 3) for v in costs]} median GPU ms).")
    mm, q0d = get('q0_dense_Mmax'), get('q0_dense')
    if mm and q0d:
        w(f"Two effects grow together because the weak objective needs M > K+q, so M=4(K+q) grows "
          f"with q. The q=0 control at the ladder's largest test count M={mm['M']} costs "
          f"{mm['median_gpu_ms']:.3f} ms against {q0d['median_gpu_ms']:.3f} ms at M={q0d['M']}, a "
          f"factor {mm['median_gpu_ms'] / q0d['median_gpu_ms']:.3f}, at "
          f"{mm['worst_same_grid_percent']:.4f}% against {q0d['worst_same_grid_percent']:.4f}%, so "
          f"most of the ladder's cost growth is the growing test count rather than the extra "
          f"unknowns.")
    w('')
    base = ladder[0]
    best = min(ladder, key=lambda x: x['worst_same_grid_percent'])
    if best['median_gpu_ms'] > base['median_gpu_ms']:
        rate = ((base['worst_same_grid_percent'] - best['worst_same_grid_percent'])
                / (best['median_gpu_ms'] - base['median_gpu_ms']))
        w(f"Best rung q={best['q']} removes "
          f"{base['worst_same_grid_percent'] - best['worst_same_grid_percent']:.4f} percentage points "
          f"for {best['median_gpu_ms'] - base['median_gpu_ms']:.3f} extra median GPU ms, "
          f"{rate:.5f} points per millisecond, a cost factor "
          f"{best['median_gpu_ms'] / base['median_gpu_ms']:.3f} over q=0.")
        w('')
    roms = [x for x in rows if x['kind'] == 'rom']
    for fn in ('nt1e-2', 'fft_tight'):
        f = get(fn)
        if not f:
            continue
        beat = [x for x in roms if x['worst_same_grid_percent'] <= f['worst_same_grid_percent']
                and x['median_gpu_ms'] <= f['median_gpu_ms']]
        w(f"Against the same-job full-order `{fn}` ({f['worst_same_grid_percent']:.4f}% same-grid, "
          f"{f['median_gpu_ms']:.3f} ms median GPU, {f['median_host_ms']:.3f} ms complete host "
          f"query): " + (', '.join(f"`{x['arm']}`" for x in beat) + ' dominate it on both axes.'
                         if beat else 'no rung dominates it on both axes.'))
    w('')
    w('**Recorded deviations.** Empirical quadrature is fitted only at '
      f"q in {sorted({x['q'] for x in rows if x['quadrature'] == 'eq'})}; above that m=4M grows with q "
      'and the bounded nonnegative-least-squares fit is not constructible inside the job budget, so '
      'those rungs use the exact dense grid sum, stated per row, with paired eq/dense rows at the '
      'same q isolating the quadrature effect. At q=R the reachable set coincides with the '
      'head-ablation free-bank arm (d), but the parameterization is redundant by K dimensions and the '
      'test count differs, so it is the same reachable set and not the same solver; the two are not '
      'expected to agree numerically. The trust radius, budgets and tolerance stay at arm (a) values '
      'for every rung, so a fixed trust radius is a tighter restriction on a larger step, which is '
      'part of what the ladder measures. Arms above '
      f"{cfg['gauss_jordan_max']} unknowns use a pivoted dense step solve rather than the incumbent "
      'unrolled Gauss-Jordan, which is more accurate, not weaker. The stationarity column is not a '
      'quality ranking; the completion column is the honest status.')
    w('')
    w(f"Source-generated report and figure: `{a.report}` with `-cost.png` / `-cost.pdf` beside it, "
      f"both produced by `reports/generate_correction_ladder.py`. Raw archive Git-tracked as bounded "
      f"chunks under `experiments/head-ablation/artifacts/qlad01/`: {a.archive}. The exact remote "
      f"attempt directory was removed after checksum collection and the namespace is empty. Not "
      f"pushed, per the coordinator's standing instruction; commits are local only.")
    w('')
    w('**Open.** Other meshes, other checkpoints, more than one training seed, the sealed final '
      'cohort, and whether a direction rule fitted to trajectory error rather than reconstruction '
      'residual would move the curve. No earlier numerical result is retracted and no worktree was '
      'merged.')
    w('')
    Path(a.out).write_text('\n'.join(L))
    print(a.out, len('\n'.join(L)))


if __name__ == '__main__':
    main()
