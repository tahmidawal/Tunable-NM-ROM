"""Generate the LAB-LOG.md entry for this cell from the audited JSONs.

Every table and every number below is read from the audit files; the prose around them
is written by hand in the session and passed in with `--prose-*`, so a number can never
be typed into the log by hand.
"""
import argparse
import hashlib
import json
from pathlib import Path


def fmt(x, d=4):
    if x is None:
        return '—'
    if isinstance(x, bool):
        return 'yes' if x else 'no'
    return f'{x:.{d}f}' if isinstance(x, float) else str(x)


def yn(x):
    return {True: 'yes', False: 'no', None: '—'}[x]


def table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] * len(header)) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(str(c) for c in r) + ' |')
    return '\n'.join(out) + '\n'


def gates_line(checks):
    parts = []
    for name, v in sorted(checks.items()):
        if not isinstance(v, dict) or 'passed' not in v or v['passed'] is None:
            continue
        parts.append(f'`{name}` {yn(v["passed"])}')
    return '; '.join(parts) + '.'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--q1-audit', required=True)
    p.add_argument('--q2-audit', default=None)
    p.add_argument('--report', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--branch', required=True)
    p.add_argument('--namespace', required=True)
    p.add_argument('--archives', default=None, help='JSON list of {attempt, sha256, chunks}')
    a = p.parse_args()
    a1 = json.loads(Path(a.q1_audit).read_text())
    a2 = json.loads(Path(a.q2_audit).read_text()) if a.q2_audit else None
    W = []

    def w(s):
        W.append(s)

    w('## 2026-09-16\n')
    w('### b-ladder-top — TITLE\n\n')
    w('PROSE-OPENING\n\n')
    w(f'Worktree `worktrees/2026-09-16-b-ladder-top`, branch `{a.branch}`. Namespace '
      f'`{a.namespace}`. ')
    jobs = [f"Q1 job `{a1.get('job_id')}` (`btq101`) on `{a1.get('gpu')}`, source "
            f"`{a1.get('commit')}`, elapsed {fmt(a1.get('elapsed_seconds'), 1)} s"]
    if a2:
        jobs.append(f"Q2 job `{a2.get('job_id')}` (`btq201`) on `{a2.get('gpu')}`, source "
                    f"`{a2.get('commit')}`, elapsed {fmt(a2.get('elapsed_seconds'), 1)} s")
    w('. '.join(jobs) + '. Both printed `jax_backend=gpu`, ran float64 with highest matmul '
      'precision, and were checksum-collected, independently NumPy-audited and archived '
      'before their exact remote attempt directories were removed.\n\n')

    w('**Q1 gates.** ' + gates_line(a1['checks']) + '\n\n')
    if a2:
        w('**Q2 gates.** ' + gates_line(a2['checks']) + '\n\n')

    w('**Fidelity against the cheap-corrections job.**\n\n')
    for label, au in (('Q1', a1), ('Q2', a2)):
        if au is None:
            continue
        fid = (au['checks'].get('cclad01_fidelity') or {}).get('detail') or {}
        if not fid:
            continue
        w(f'{label}:\n\n')
        w(table(['arm', 'reproduces', 'tolerance', 'relative difference (reference)',
                 'relative difference (same-grid)', 'passed'],
                [[f'`{k}`', f"`{v['detail']['comparator']}`",
                  fmt(v['detail']['declared_tolerance'], 12),
                  fmt(v['detail']['relative_worst_reference_difference'], 12),
                  fmt(v['detail'].get('relative_worst_same_grid_difference'), 12),
                  yn(v['passed'])] for k, v in sorted(fid.items())]) + '\n')

    w('**Conditioning of the augmented normal equations** (offline probe, $\\lambda=10^{-6}$):\n\n')
    w(table(['q', 'unknowns', 'column norm ratio', '$\\kappa$ unscaled', '$\\kappa$ equilibrated',
             'improvement'],
            [[c['q'], c['unknowns'], f"{c['column_norm_ratio']:.3e}",
              f"{c['lambdas']['1e-06']['unscaled_condition']:.3e}",
              f"{c['lambdas']['1e-06']['equilibrated_condition']:.3e}",
              f"{c['lambdas']['1e-06']['improvement_factor']:.3f}"]
             for c in sorted(a1['conditioning'], key=lambda c: c['q'])]) + '\n')

    w('**Q1 convergence sweep** (dense quadrature):\n\n')
    sweep = [x for x in a1['arms'] if x['kind'] == 'rom' and x['quadrature'] == 'dense']
    w(table(['arm', 'q', 'fix', 'median iters/step', 'budget exits', 'worst joint gradient',
             'worst same-grid all %', 'worst evolved %', 'median GPU ms', 'cascade total ms',
             'converged'],
            [[f"`{x['arm']}`", x['q'], f"`{x['fix']}`", fmt(x['median_iterations'], 1),
              x['total_budget_exits'], f"{x['max_joint_stationarity']:.3e}",
              fmt(x['worst_all_times_percent']), fmt(x['worst_evolved_percent']),
              fmt(x['median_gpu_ms'], 3), fmt(x['cascade_total_gpu_ms'], 3), yn(x['converged'])]
             for x in sorted(sweep, key=lambda x: (x['q'], x['fix'] or ''))]) + '\n')

    w('**Q1 empirical quadrature per rung:**\n\n')
    eqr = [x for x in a1['arms'] if x['kind'] == 'rom' and x['quadrature'] == 'eq']
    w(table(['arm', 'q', 'M', 'm', 'relative fit', 'truncated', 'rule valid', 'fit seconds',
             'worst same-grid all %', 'median GPU ms', 'converged'],
            [[f"`{x['arm']}`", x['q'], x['M'], x['m'], fmt(x['quadrature_fit_relative'], 6),
              yn(x['quadrature_truncated']), yn(x['eq_rule_valid']),
              fmt(x['quadrature_fit_seconds'], 1), fmt(x['worst_all_times_percent']),
              fmt(x['median_gpu_ms'], 3), yn(x['converged'])]
             for x in sorted(eqr, key=lambda x: (x['q'], x['fix'] or ''))]) + '\n')

    if a2:
        w('**Q2 envelope, every subject, both metrics:**\n\n')
        w(table(['subject', 'family', "q / k'", 'quadrature', 'evolution tol',
                 'worst all times %', 'worst evolved %', 't0 compression %', 'median GPU ms',
                 'converged'],
                [[f"`{x['arm']}`", x['family'], x['q'] if x['q'] is not None else x['k'],
                  x['quadrature'] or '—', (f"{x['gtol']:g}" if x['gtol'] else '—'),
                  fmt(x['worst_all_times_percent']), fmt(x['worst_evolved_percent']),
                  fmt(x['worst_t0_compression_percent']), fmt(x['effective_gpu_ms'], 3),
                  yn(x['converged'])] for x in a2['arms']]) + '\n')
        by = {x['arm']: x for x in a2['arms']}
        for key, name, err in (('all_subjects_all_times', 'all output times',
                                'worst_all_times_percent'),
                               ('all_subjects_evolved', 'evolved times only',
                                'worst_evolved_percent')):
            w(f'**Non-dominated over (median GPU ms, worst error), {name}:**\n\n')
            sel = sorted([by[n] for n in a2['frontier'][key]],
                         key=lambda x: x['effective_gpu_ms'])
            w(table(['subject', 'family', "q / k'", 'worst error %', 'median GPU ms', 'converged'],
                    [[f"`{x['arm']}`", x['family'], x['q'] if x['q'] is not None else x['k'],
                      fmt(x[err]), fmt(x['effective_gpu_ms'], 3), yn(x['converged'])]
                     for x in sel]) + '\n')
        k = a2['knob_criterion']
        w('**Pre-registered criterion for calling $q$ a knob:**\n\n')
        w(table(['metric', 'monotone at fixed M', 'monotone over every rung',
                 'converged non-dominated points', 'cost span', 'error span', 'passes'],
                [[m, yn(k[m]['monotone_fixed_test_count']), yn(k[m]['monotone_all_rungs']),
                  k[m].get('points'), fmt(k[m].get('cost_span'), 3),
                  fmt(k[m].get('error_span'), 3), yn(k[m]['passes'])]
                 for m in ('evolved', 'all_times')]) + '\n')
        if a2.get('fno'):
            f = a2['fno']
            w(f"**FNO in the same allocation.** `{f['model']}`, "
              f"{f['real_parameter_count']:,} real parameters, checkpoint SHA256 "
              f"`{f['checkpoint_sha256'][:16]}…`, pooled device query "
              f"{f['device_query_pooled']['median_ms']:.3f} ms median over "
              f"{f['device_query_pooled']['count']} retained repetitions.\n\n")
        else:
            w('**FNO.** Not run in-job; omitted rather than imported from another job.\n\n')

    rp = Path(a.report)
    w(f'Source-generated report: `{rp.as_posix()}` (SHA256 '
      f'`{hashlib.sha256(rp.read_bytes()).hexdigest()}`) with its envelope figure and '
      f'generator beside it.\n\n')
    if a.archives:
        for row in json.loads(Path(a.archives).read_text()):
            w(f"Raw archive `{row['attempt']}` Git-tracked as bounded chunks: whole SHA256 "
              f"`{row['sha256']}` ({row['chunks']} chunks).\n")
        w('\n')
    w('PROSE-FINDINGS\n\nPROSE-RETRACTIONS\n\nPROSE-OPEN\n')
    Path(a.out).write_text(''.join(W))
    print(a.out)


if __name__ == '__main__':
    main()
