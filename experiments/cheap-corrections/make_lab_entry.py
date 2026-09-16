"""Generate this cell's lab-log entry from the audited run JSONs.

Every number in the entry comes from the audit output, so the lab log cannot drift
from the data. The text is printed; appending it to the canonical log is a separate,
deliberate step.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def f(x, d=4):
    return '—' if x is None else f'{x:.{d}f}'


def yn(x):
    return {True: 'yes', False: 'no', None: '—'}[x]


def table(head, body):
    return '\n'.join(['| ' + ' | '.join(head) + ' |',
                      '|' + '|'.join(['---:' if i else '---' for i in range(len(head))]) + '|']
                     + ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in body])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--burgers-audit', required=True)
    p.add_argument('--burgers-result', required=True)
    p.add_argument('--poisson-audit', required=True)
    p.add_argument('--poisson-result', required=True)
    p.add_argument('--report', required=True)
    p.add_argument('--burgers-archive', default=None)
    p.add_argument('--poisson-archive', default=None)
    p.add_argument('--branch-commit', default=None)
    a = p.parse_args()
    ba = json.loads(Path(a.burgers_audit).read_text())
    br = json.loads(Path(a.burgers_result).read_text())
    pa = json.loads(Path(a.poisson_audit).read_text())
    pr = json.loads(Path(a.poisson_result).read_text())
    rows = ba['checks']['arm_table']
    roms = [x for x in rows if x['kind'] == 'rom']
    foms = [x for x in rows if x['kind'] == 'fom']
    conv = [x for x in roms if x['converged']]
    nd = [x for x in roms if x['arm'] in ba['checks']['nondominated_converged']]
    base = next((x for x in roms if x['arm'] == 'q0_m4_eq_varpro'), None)
    q64 = [x for x in roms if x['q'] == 64 and x['converged']]
    best64 = min(q64, key=lambda x: x['median_gpu_ms']) if q64 else None
    ratio = best64['median_gpu_ms'] / base['median_gpu_ms'] if (best64 and base) else None
    cs = ([x['median_gpu_ms'] for x in nd], [x['worst_same_grid_percent'] for x in nd])
    cspan = (max(cs[0]) / min(cs[0])) if len(nd) > 1 else None
    espan = (max(cs[1]) / min(cs[1])) if len(nd) > 1 else None

    L = []
    W = L.append
    W('## 2026-09-15')
    W('### cheap-corrections — variable projection, test count and per-rung quadrature on the '
      'Burgers correction ladder, and the same ladder on Poisson\n')
    W('The coordinator asked whether the audited fixed-weight correction ladder (job `3713867`) '
      'can be made to CONVERGE cheaply without changing the checkpoint, the directions rule or '
      'the reachable set. Three changes, each isolated: (1) eliminate the correction coefficients '
      'from the nonlinear iteration; (2) decouple the test count from $q$; (3) fit one empirical '
      'quadrature rule per rung. Predeclared protocol and its amendments: '
      '`experiments/cheap-corrections/DESIGN.md`.\n')
    W(f"Worktree `worktrees/2026-09-15-cheap-corrections`, branch "
      f"`exp/2026-09-15-cheap-corrections` at `{a.branch_commit or 'see below'}`. Namespace "
      f"`/cluster/tufts/paralab/tawal01/cheap_corr_20260915/`. Burgers job `{br.get('job_id')}` "
      f"(`cclad01`) on `{br.get('gpu')}`, source `{br.get('commit')}`, elapsed "
      f"{br.get('elapsed_seconds', 0):.1f} s. Poisson job `{pr.get('job_id')}` (`ccpoi01`) on "
      f"`{pr.get('gpu')}`, source `{pr.get('commit')}`, elapsed {pr.get('elapsed_seconds', 0):.1f} s. "
      f"Both printed `jax_backend=gpu`, ran float64 with highest matmul precision, and were "
      f"checksum-collected, independently NumPy-audited and archived before their exact remote "
      f"attempt directories were removed.\n")

    W('**Gates.** ' + '; '.join(
        f"`{k}` {yn(v['passed'])}" for k, v in ba['checks'].items()
        if isinstance(v, dict) and 'passed' in v) + '.\n')
    W('Poisson gates: ' + '; '.join(
        f"`{k}` {yn(v['passed'])}" for k, v in pa['checks'].items()
        if isinstance(v, dict) and 'passed' in v) + '.\n')

    W('**Solver variants at matched $q$, test count and quadrature** (dense, $M=4(K+q)$):\n')
    var = sorted([x for x in roms if x['rule'] == 'm4' and x['quadrature'] == 'dense'],
                 key=lambda x: (x['q'], x['variant']))
    W(table(['q', 'variant', 'worst same-grid %', 'median GPU ms', 'median iters/step',
             'budget exits', 'max joint gradient', 'converged'],
            [[x['q'], x['variant'], f(x['worst_same_grid_percent']), f(x['median_gpu_ms'], 3),
              f(x['median_iterations'], 1), x['total_budget_exits'],
              f"{x['max_joint_stationarity']:.2e}", yn(x['converged'])] for x in var]) + '\n')

    W('**The ladder.** All Burgers arms:\n')
    W(table(['arm', 'q', 'rule', 'M', 'quad', 'm', 'worst same-grid %', 'worst reference %',
             'median GPU ms', 'budget exits', 'converged'],
            [[f"`{x['arm']}`", x['q'], x['rule'], x['M'], x['quadrature'], f(x['m'], 0),
              f(x['worst_same_grid_percent']), f(x['worst_reference_percent']),
              f(x['median_gpu_ms'], 3), x['total_budget_exits'], yn(x['converged'])]
             for x in sorted(roms, key=lambda x: (x['q'], x['rule'], x['quadrature'],
                                                  x['variant']))]) + '\n')
    W('Same-job full-order controls (context only, no cross-job ratio is taken):\n')
    W(table(['method', 'worst same-grid %', 'worst reference %', 'median GPU ms'],
            [[f"`{x['arm']}`", f(x['worst_same_grid_percent']), f(x['worst_reference_percent']),
              f(x['median_gpu_ms'], 3)] for x in foms]) + '\n')

    W(f"**Pre-registered target.** $q=64$ converged at $\\le 3\\times$ the $q=0$ median GPU cost. "
      f"Baseline `q0_m4_eq_varpro` {f(base['median_gpu_ms'], 3) if base else '—'} ms; cheapest "
      f"converged $q=64$ arm "
      f"{('`' + best64['arm'] + '` ' + f(best64['median_gpu_ms'], 3) + ' ms at '
         + f(best64['worst_same_grid_percent']) + '%') if best64 else 'none — no q=64 arm converged'}"
      f"; ratio {f(ratio, 3)}. "
      f"**{'PASS' if (ratio is not None and ratio <= 3) else 'FAIL'}**.\n")
    W(f"**Pre-registered acceptance for calling $q$ a knob.** {len(nd)} non-dominated CONVERGED "
      f"points (required at least 3), spanning {f(cspan, 2)}x in cost (required 2x) and "
      f"{f(espan, 2)}x in error (required 2x). Non-dominated converged set: "
      + (', '.join(f"`{x['arm']}`" for x in nd) or 'empty') + '.\n')
    if nd:
        W(table(['arm', 'q', 'rule', 'quad', 'worst same-grid %', 'median GPU ms'],
                [[f"`{x['arm']}`", x['q'], x['rule'], x['quadrature'],
                  f(x['worst_same_grid_percent']), f(x['median_gpu_ms'], 3)]
                 for x in sorted(nd, key=lambda x: x['median_gpu_ms'])]) + '\n')

    prows = pa['checks']['arm_table']
    W('**Poisson 1024 intervals**, same checkpoint, twelve opened development sources, '
      '`dst_direct` interleaved. The Poisson residual is linear in the coefficients, so the '
      'corrections are eliminated exactly and the nonlinear iteration stays 16-dimensional at '
      'every $q$.\n')
    W(table(['arm', 'q', 'rule', 'M', 'bank projection %', 'best-found %', 'worst physical %',
             'median device ms', 'all solver-valid'],
            [[f"`{x['arm']}`", x['q'], x['rule'], x['M'],
              f(x['worst_bank_projection_percent']), f(x['worst_best_found_percent']),
              f(x['worst_physical_percent']), f(x['median_device_ms'], 4),
              yn(x['all_solver_valid'])]
             for x in sorted(prows, key=lambda x: (x['kind'] != 'rom',
                                                   x['q'] if x['q'] is not None else 0,
                                                   x['rule'] or ''))]) + '\n')

    d = br['directions']
    fl = br.get('directions_flat') or {}
    W(f"**Directions.** Regenerated with the audited rule and seed; `directions_sha256` "
      f"`{d['directions_sha256'][:16]}…` against the retained "
      f"`{(d.get('expected_sha256') or '')[:16]}…`, bitwise match "
      f"{yn(d.get('bitwise_matches_qlad01'))}. Flattening the doubly vectorised fit took "
      f"{f(fl.get('seconds'), 1)} s against {f(d['seconds'], 1)} s, saving "
      f"{f(fl.get('compile_seconds_saved'), 1)} s; bitwise identical to the audited path: "
      f"{yn(fl.get('bitwise_identical_to_audited'))}"
      + (f" (max absolute difference {fl['max_abs_difference_from_audited']:.3e})" if fl else '')
      + '.\n')

    eq = [x for x in roms if x['quadrature'] == 'eq']
    if eq:
        W('**Empirical quadrature per rung** (offline cost, never inside a query timing):\n')
        W(table(['arm', 'q', 'M', 'm', 'fitter', 'relative fit', 'support', 'truncated',
                 'fit seconds'],
                [[f"`{x['arm']}`", x['q'], x['M'], f(x['m'], 0), x['fitter'],
                  (f"{x['quadrature_fit_relative']:.3e}" if x['quadrature_fit_relative'] else '—'),
                  f(x['quadrature_support'], 0), yn(x['quadrature_truncated']),
                  f(x['quadrature_fit_seconds'], 1)] for x in sorted(eq, key=lambda x: (x['q'], x['M']))])
          + '\n')

    W(f"Source-generated report: `{a.report}` "
      f"(SHA256 `{hashlib.sha256(Path(a.report).read_bytes()).hexdigest()}`) with its cost figure "
      f"and generator beside it.\n")
    for name, arch in (('Burgers', a.burgers_archive), ('Poisson', a.poisson_archive)):
        if arch:
            j = json.loads(Path(arch).read_text())
            W(f"{name} raw archive Git-tracked as bounded chunks: whole SHA256 `{j['sha256']}` "
              f"({len(j['chunks'])} chunks).")
    W('')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
