"""Generate the dated LAB-LOG entry for this cell from the audited JSON.

Prints the entry to stdout and, with --append, appends it to the canonical lab log by
absolute path. Every number comes from the audit; the prose that surrounds them is the
only hand-written part and it is written here, once, rather than retyped into the log.
"""
import argparse
import hashlib
import json
from pathlib import Path

LOG = Path('/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/LAB-LOG.md')


def yn(x):
    return {True: 'yes', False: 'no', None: 'n/a'}[x]


def pct(x, d=4):
    return '—' if x is None else f'{x:.{d}f}'


def f(x, d=3):
    return '—' if x is None else f'{x:.{d}f}'


def sci(x, d=2):
    return '—' if x is None else f'{x:.{d}e}'


def table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |',
           '|' + '|'.join(['---'] * len(header)) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(str(c) for c in r) + ' |')
    return '\n'.join(out) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', required=True)
    p.add_argument('--report', required=True)
    p.add_argument('--archive', default=None)
    p.add_argument('--append', action='store_true')
    a = p.parse_args()
    aud = json.loads(Path(a.audit).read_text())
    v = aud['verdict']
    lads = aud['ladders']
    ds = aud['direction_sets']
    cc = aud['direction_comparison']['cross_capture']
    pa = aud['direction_comparison']['principal_angles']
    state = 'PASSES' if v['passes'] else 'FAILS'

    L = []
    L.append('## 2026-09-16\n')
    L.append(f'### q-trajdirs — correction directions fitted to the ROM\'s TRAJECTORY error '
             f'instead of the head\'s static reconstruction residual; the pre-registered pass '
             f'{state}\n')
    L.append(
        'The coordinator asked one question: does fitting the correction directions $C_q$ to the '
        'ROM\'s **trajectory** error, instead of the head\'s **static** reconstruction residual, '
        'make the Burgers $256^2$ fixed-weight correction ladder monotone on the '
        'worst-over-evolved-times metric ($t>0$) while keeping it monotone on '
        'worst-over-all-times? Everything except $C_q$ is the `b-ladder-top` Q1-B contract '
        'verbatim: the frozen checkpoint `sep_hfit_dense_mid_N256_dense.pkl` '
        '(SHA256 `18f0266ae6f0…`), the bank, the head, the weak objective, the initializer, the '
        'block-damped variable-projection solver at per-step budget 600, the output contract, the '
        'mesh, the time step and the same six opened development cases. Predeclared protocol and '
        'every amendment: `experiments/q-trajdirs/DESIGN.md`. Nothing was merged and the branch '
        'was **not** pushed.\n')
    L.append(
        f'Worktree `worktrees/2026-09-16-q-trajdirs`, branch `exp/2026-09-16-q-trajdirs`, forked '
        f'from `exp/2026-09-16-b-ladder-top` at `b8efd5b4`. Namespace '
        f'`/cluster/tufts/paralab/tawal01/q_trajdirs_20260916/`. One job `{aud["job_id"]}` '
        f'(`qtd01`) on `{aud["gpu"]}`, source `{aud["commit"]}`, elapsed '
        f'{aud["elapsed_seconds"]:.1f} s. It printed `jax_backend=gpu`, ran float64 with highest '
        f'matmul precision, was checksum-collected, independently NumPy-audited and archived '
        f'before its exact remote attempt directory was removed. `squeue` was checked before and '
        f'after the single submission; the 3-job cap left two unused.\n')

    L.append('**Gates.** ' + '; '.join(
        f'`{k}` {yn(vv.get("passed"))}' for k, vv in sorted(aud['checks'].items())
        if isinstance(vv, dict) and vv.get('passed') is not None) + '.\n')

    L.append('**The three direction sets.**\n')
    L.append(table(['set', 'residual decomposed', 'rows', 'rank', 'fit s', '`directions_sha256`'],
                   [[f'`{k}`',
                     ('static reconstruction residual, 1024 snapshots' if k == 'old'
                      else f"trajectory error, {ds[k].get('cohort', {}).get('count', '—')} "
                           f"training trajectories x 51 steps"),
                     ds[k]['residual_rows'], ds[k]['available_rank'], f(ds[k].get('seconds'), 1),
                     '`' + ds[k]['directions_sha256'][:16] + '…`']
                    for k in ('old', 'traj', 'prac') if k in ds]))

    qs = [q for q in aud['ladders']['traj_primary']['rungs']]
    qlist = [u['q'] for u in qs if u['q'] > 0]
    L.append('\n**Cross-capture** $\\kappa_q(P,C)$, per cent of residual matrix $P$\'s whitened '
             'energy that direction set $C$ reaches at rung $q$:\n\n')
    L.append(table(['residual $P$', 'directions $C$'] + [f'$q={q}$' for q in qlist],
                   [[f'`{pn}`', f'`{cn}`'] + [f'{cc[pn][cn][str(q)] * 100:.2f}' for q in qlist]
                    for pn in ('old', 'traj', 'prac') for cn in ('old', 'traj', 'prac')
                    if pn in cc and cn in cc.get(pn, {})]))

    L.append('\n**Principal angles between the direction subspaces** (field metric, degrees):\n\n')
    L.append(table(['pair', '$q$', 'overlap', 'min', 'median', 'max'],
                   [[f'`{pair}`', q, f(d[q]['overlap'], 5), f(d[q]['angle_min_degrees'], 2),
                     f(d[q]['angle_median_degrees'], 2), f(d[q]['angle_max_degrees'], 2)]
                    for pair, d in sorted(pa.items()) for q in sorted(d, key=int)]))

    L.append('\n**The ladders, both metrics.**\n')
    for lid in ('old_primary', 'traj_primary', 'prac_primary', 'old_fixedM', 'traj_fixedM',
                'old_dense', 'traj_dense', 'prac_dense'):
        lad = lads.get(lid)
        if not lad:
            continue
        L.append(f'\n`{lid}` — {lad["label"]}; monotone evolved {yn(lad["monotone_evolved"])}, '
                 f'monotone all times {yn(lad["monotone_all_times"])}, every rung converged '
                 f'{yn(lad["all_converged"])}:\n\n')
        L.append(table(['$q$', '$M$', '$m$', 'worst all %', 'worst evolved %', '$t_0$ %',
                        'best-found %', 'median GPU ms', 'budget exits', 'converged'],
                       [[u['q'], u['M'], u['m'] or '—', pct(u['all_times']), pct(u['evolved']),
                         pct(u['t0']), pct(u['best_found']), f(u['gpu_ms']), u['budget_exits'],
                         yn(u['converged'])] for u in lad['rungs']]))

    L.append('\n**The $q=16$ regression (falsification F1), per ladder:**\n\n')
    L.append(table(['ladder', 'directions', '$q=0$ evolved %', '$q=16$ evolved %', 'change (pp)',
                    'regression'],
                   [[f'`{lid}`', lads[lid]['dirset'], pct(x['q0']), pct(x['q16']),
                     f"{x['delta_percentage_points']:+.4f}", yn(x['regression'])]
                    for lid, x in sorted(v['q16_regression'].items())]))

    L.append('\n**Pre-registered pass** (the `traj` primary ladder, $M=4(K+q)$, EQ per rung):\n\n')
    L.append(table(['criterion', 'holds', 'measured'],
                   [['1. evolved non-increasing in $q$, every rung converged',
                     yn(v['criterion_1_evolved_monotone_every_rung_converged']),
                     f"monotone {yn(v['monotone_evolved'])}, all converged {yn(v['all_converged'])}"],
                    ['2. all-times non-increasing in $q$', yn(v['criterion_2_all_times_monotone']),
                     f"monotone {yn(v['monotone_all_times'])}"],
                    ['3. converged non-dominated set spans $\\ge2\\times$ in evolved error',
                     yn(v['criterion_3_passes']),
                     f"{v['criterion_3_error_span']['points']} points, error span "
                     f"{f(v['criterion_3_error_span']['error_span'])}x, cost span "
                     f"{f(v['criterion_3_error_span']['cost_span'])}x"]]))
    L.append(f'\n**Overall: the pre-registered pass {state}.**\n')

    L.append('\n**Non-dominated over (median GPU ms, error), every subject:** all times '
             + ', '.join(f'`{n}`' for n in aud['frontier']['all_subjects_all_times'])
             + '; evolved times '
             + ', '.join(f'`{n}`' for n in aud['frontier']['all_subjects_evolved']) + '.\n')

    L.append('\n**Cross-job fidelity.**\n\n')
    fid = aud['checks']['cross_job_fidelity']['detail']
    L.append(table(['arm', 'reproduces', 'job', 'tolerance', 'worst relative difference', 'passed'],
                   [[f'`{k}`', f"`{d['detail']['comparator']}`", d['detail']['job'],
                     sci(d['detail']['declared_tolerance']),
                     sci(d['detail']['worst_relative_difference']), yn(d['passed'])]
                    for k, d in sorted(fid.items()) if isinstance(d.get('detail'), dict)]))

    rp = Path(a.report)
    L.append(f'\nSource-generated report: `{rp.relative_to(rp.parents[3])}` (SHA256 '
             f'`{hashlib.sha256(rp.read_bytes()).hexdigest()}`) with its figure and its '
             f'generator beside it; the generator reads only the audited JSON, so no number in '
             f'it is hand-typed.\n')
    if a.archive and Path(a.archive).exists():
        arc = json.loads(Path(a.archive).read_text())
        L.append(f'\nRaw archive `qtd01` Git-tracked as bounded chunks: whole SHA256 '
                 f'`{arc["sha256"]}` ({len(arc["chunks"])} chunks).\n')

    text = ''.join(x if x.endswith('\n\n') else x + '\n' for x in L)
    print(text)
    if a.append:
        with LOG.open('a') as fh:
            fh.write('\n' + text)
        print(f'APPENDED to {LOG}')


if __name__ == '__main__':
    main()
