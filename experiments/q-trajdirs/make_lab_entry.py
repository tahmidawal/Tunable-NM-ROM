"""Generate the dated LAB-LOG entry for this cell from the audited JSONs.

Prints the entry to stdout and, with --append, appends it to the canonical lab log by
absolute path. Every number comes from an audit; the prose around them is written here,
once, rather than retyped into the log.
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


def gates(checks):
    return '; '.join(f'`{k}` {yn(v.get("passed"))}' for k, v in sorted(checks.items())
                     if isinstance(v, dict) and v.get('passed') is not None) + '.'


def ladders(lads, order=None):
    out = []
    for lid in (order or sorted(lads)):
        lad = lads.get(lid)
        if not lad:
            continue
        out.append(f'\n`{lid}` — {lad["label"]}; monotone evolved '
                   f'{yn(lad["monotone_evolved"])}, monotone all times '
                   f'{yn(lad["monotone_all_times"])}, every rung converged '
                   f'{yn(lad["all_converged"])}:\n\n')
        out.append(table(['$q$', '$M$', '$m$', 'all %', 'evolved %', '$t_0$ %', 'best-found %',
                          'median GPU ms', 'budget exits', 'converged'],
                         [[u['q'], u['M'], u['m'] or '—', pct(u['all_times']), pct(u['evolved']),
                           pct(u['t0']), pct(u['best_found']), f(u['gpu_ms']), u['budget_exits'],
                           yn(u['converged'])] for u in lad['rungs']]))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', required=True, help='the dense-ladder job audit (qtd02)')
    p.add_argument('--archives', required=True)
    p.add_argument('--control-audit', default=None)
    p.add_argument('--report', required=True)
    p.add_argument('--archive-json', action='append', default=[])
    p.add_argument('--append', action='store_true')
    a = p.parse_args()
    aud = json.loads(Path(a.audit).read_text())
    arc = json.loads(Path(a.archives).read_text())
    ctl = json.loads(Path(a.control_audit).read_text()) if a.control_audit else None
    v = aud['verdict']
    state = 'PASSES' if v['passes'] else 'FAILS'

    L = []
    L.append('## 2026-09-16\n')
    L.append(f'### q-trajdirs — redirected mid-flight: the trajectory-direction hypothesis was '
             f'killed by the q-diag lane before this lane\'s own job landed, and the DENSE '
             f'correction ladder {state} the redirected knob criterion\n')
    L.append(
        'The lane opened on one question — does fitting the correction directions $C_q$ to the '
        "ROM's **trajectory** error, instead of the head's **static** reconstruction residual, "
        'make the Burgers $256^2$ ladder monotone on the worst-over-evolved-times metric? It was '
        'pre-registered in full, implemented, smoked and submitted. **While that job was '
        'running, the `q-diag` lane answered the underlying question from data that already '
        'existed: the $q=16$ evolved-metric regression is the EMPIRICAL QUADRATURE, not the '
        'directions.** The coordinator redirected the lane; both halves are recorded below and '
        'the redirect itself is section 11 of `experiments/q-trajdirs/DESIGN.md`, appended '
        'without editing anything above it. Nothing was merged and the branch was **not** '
        'pushed.\n')
    L.append(
        f'Worktree `worktrees/2026-09-16-q-trajdirs`, branch `exp/2026-09-16-q-trajdirs`, forked '
        f'from `exp/2026-09-16-b-ladder-top` at `b8efd5b4`. Namespace '
        f'`/cluster/tufts/paralab/tawal01/q_trajdirs_20260916/`. Two A100 jobs, one attempt '
        f'directory each, `squeue` checked before and after every submission, the three-job cap '
        f'leaving one unused: the redirected primary `{aud["job_id"]}` (`qtd02`) on '
        f'`{aud["gpu"]}`, source `{aud["commit"]}`, elapsed {aud["elapsed_seconds"]:.1f} s'
        + (f'; and the control `{ctl["job_id"]}` (`qtd01`) on `{ctl["gpu"]}`, source '
           f'`{ctl["commit"]}`, elapsed {ctl["elapsed_seconds"]:.1f} s' if ctl else '')
        + '. Both printed `jax_backend=gpu`, ran float64 with highest matmul precision, and were '
          'checksum-collected, independently NumPy-audited and archived before their exact '
          'remote attempt directories were removed.\n')

    # --------------------------------------------------------------- the verdict
    L.append('**The redirected verdict.** On the `' + v['ladder'] + '` ladder — incumbent '
             'directions, DENSE quadrature, $M=4(K+q)$, per-step budget 600, one allocation:\n\n')
    L.append(table(['criterion', 'holds', 'measured'],
                   [['1. worst evolved error non-increasing in $q$, every rung converged',
                     yn(v['criterion_1_evolved_monotone_every_rung_converged']),
                     f"monotone {yn(v['monotone_evolved'])}, all converged "
                     f"{yn(v['all_converged'])}"],
                    ['2. converged non-dominated set spans $\\ge2\\times$ in error AND '
                     '$\\ge2\\times$ in cost, evolved metric', yn(v['criterion_3_passes']),
                     f"{v['criterion_3_error_span']['points']} points, error span "
                     f"{f(v['criterion_3_error_span']['error_span'])}x, cost span "
                     f"{f(v['criterion_3_error_span']['cost_span'])}x"],
                    ['(reported) worst all-times error non-increasing in $q$',
                     yn(v['criterion_2_all_times_monotone']), '—']]))
    L.append(f'\n**Overall: the redirected criterion {state}.** Converged non-dominated arms on '
             'the evolved metric: '
             + ', '.join(f'`{n}`' for n in v['converged_nondominated_evolved']) + '.\n')

    # ------------------------------------------------------------------- part 1
    L.append('\n**Part 1, no GPU — the dense ladder in data that already existed.** The four '
             'comparator archives (`cclad01`, `btq101`, `btq102`, `btq201`) were restored from '
             'their Git-tracked chunks and every same-grid error recomputed in NumPy from the '
             'saved fields, reproducing each job\'s archived reference error to '
             + sci(max(c['value'] for k, c in arc['checks'].items() if 'recomputes' in k))
             + ' relative:\n\n')
    L.append(table(['ladder', 'rungs', 'monotone evolved', 'monotone all times',
                    'every rung converged', 'non-dominated points', 'evolved error span',
                    'cost span', 'passes redirected criterion'],
                   [[f'`{lid}`', ', '.join(str(u['q']) for u in d['rungs']),
                     yn(d['monotone_evolved']), yn(d['monotone_all_times']),
                     yn(d['all_converged']), d['span_evolved']['points'],
                     f(d['span_evolved']['error_span']), f(d['span_evolved']['cost_span']),
                     yn(d['passes_redirected_criterion'])]
                    for lid, d in sorted(arc['ladders'].items())]))
    nm = sum(1 for d in arc['ladders'].values()
             for k in ('monotone_evolved', 'monotone_all_times') if d[k])
    L.append(f'\n**{nm} of {2 * len(arc["ladders"])}** dense ladder/metric combinations in the '
             'archives are monotone — the `q-diag` census reproduced independently from the '
             'fields rather than from its tables.\n')

    # ------------------------------------------------------------------- part 2
    L.append('\n**Part 2 — the redirected job, every dense rung at budget 600 in one '
             'allocation.**\n')
    L += ladders(aud['ladders'], ['dense_m4', 'dense_fixedM'])
    L.append('\nSame-job full-order controls:\n\n')
    by = {x['arm']: x for x in aud['arms']}
    L.append(table(['control', 'worst all times %', 'worst evolved %', 'median GPU ms'],
                   [[f'`{x["arm"]}`', pct(x['worst_all_times_percent']),
                     pct(x['worst_evolved_percent']), f(x['median_gpu_ms'])]
                    for x in aud['arms'] if x['family'] == 'fom']))
    L.append('\nNon-dominated over (median GPU ms, error), every subject: all times '
             + ', '.join(f'`{n}`' for n in aud['frontier']['all_subjects_all_times'])
             + '; evolved times '
             + ', '.join(f'`{n}`' for n in aud['frontier']['all_subjects_evolved']) + '.\n')

    L.append('\n**Gates, `qtd02`.** ' + gates(aud['checks']) + '\n')
    fid = aud['checks']['cross_job_fidelity']['detail']
    L.append('\n**Cross-job fidelity, `qtd02`.**\n\n')
    L.append(table(['arm', 'reproduces', 'job', 'tolerance', 'worst relative difference',
                    'passed'],
                   [[f'`{k}`', f"`{d['detail']['comparator']}`", d['detail']['job'],
                     sci(d['detail']['declared_tolerance']),
                     sci(d['detail']['worst_relative_difference']), yn(d['passed'])]
                    for k, d in sorted(fid.items()) if isinstance(d.get('detail'), dict)]))

    # ------------------------------------------------------------------ control
    if ctl:
        cv = ctl['verdict']
        cds = ctl['direction_sets']
        cc = ctl['direction_comparison']['cross_capture']
        pa = ctl['direction_comparison']['principal_angles']
        qs = sorted({u['q'] for d in ctl['ladders'].values() for u in d['rungs'] if u['q'] > 0})
        L.append('\n**The control, `qtd01` — the trajectory-fitted directions, run because it '
                 'was already running.** Its own pre-registered pass (section 6 of `DESIGN.md`) '
                 f'**{"PASSES" if cv["passes"] else "FAILS"}**; it is reported as a test of the '
                 '`q-diag` verdict, not as this cell\'s subject. Three direction sets differing '
                 'only in which residual matrix is decomposed:\n\n')
        L.append(table(['set', 'residual decomposed', 'rows', 'rank', 'fit s',
                        '`directions_sha256`'],
                       [[f'`{k}`',
                         ('static reconstruction residual, 1024 snapshots' if k == 'old'
                          else f"trajectory error, "
                               f"{cds[k].get('cohort', {}).get('count', '—')} trajectories "
                               f"x 51 internal steps"),
                         cds[k]['residual_rows'], cds[k]['available_rank'],
                         f(cds[k].get('seconds'), 1),
                         '`' + cds[k]['directions_sha256'][:16] + '…`']
                        for k in ('old', 'traj', 'prac') if k in cds]))
        L.append('\nCross-capture $\\kappa_q(P,C)$, per cent of residual matrix $P$\'s whitened '
                 'energy that direction set $C$ reaches at rung $q$:\n\n')
        L.append(table(['residual $P$', 'directions $C$'] + [f'$q={q}$' for q in qs],
                       [[f'`{pn}`', f'`{cn}`'] + [f'{cc[pn][cn][str(q)] * 100:.2f}' for q in qs]
                        for pn in ('old', 'traj', 'prac') if pn in cc
                        for cn in ('old', 'traj', 'prac') if cn in cc.get(pn, {})]))
        if pa:
            L.append('\nPrincipal angles between the direction subspaces (field metric, '
                     'degrees):\n\n')
            L.append(table(['pair', '$q$', 'overlap', 'min', 'median', 'max'],
                           [[f'`{pair}`', q, f(d[q]['overlap'], 5),
                             f(d[q]['angle_min_degrees'], 2), f(d[q]['angle_median_degrees'], 2),
                             f(d[q]['angle_max_degrees'], 2)]
                            for pair, d in sorted(pa.items()) for q in sorted(d, key=int)]))
        L.append('\nIts ladders:\n')
        L += ladders(ctl['ladders'], ['old_primary', 'traj_primary', 'prac_primary',
                                      'old_dense', 'traj_dense', 'prac_dense',
                                      'old_fixedM', 'traj_fixedM'])
        L.append('\nThe $q=16$ evolved-metric regression, per ladder of the control — the direct '
                 'test of `q-diag`\'s verdict that the quadrature, not the direction rule, '
                 'carries it:\n\n')
        L.append(table(['ladder', 'directions', 'quadrature', '$q=0$ evolved %',
                        '$q=16$ evolved %', 'change (pp)', 'regression'],
                       [[f'`{lid}`', ctl['ladders'][lid]['dirset'],
                         ctl['ladders'][lid]['quadrature'], pct(x['q0']), pct(x['q16']),
                         f"{x['delta_percentage_points']:+.4f}", yn(x['regression'])]
                        for lid, x in sorted(cv['q16_regression'].items())]))
        L.append('\n**Gates, `qtd01`.** ' + gates(ctl['checks']) + '\n')

    rp = Path(a.report)
    L.append(f'\nSource-generated report: `experiments/q-trajdirs/reports/{rp.name}` (SHA256 '
             f'`{hashlib.sha256(rp.read_bytes()).hexdigest()}`) with its figure and its generator '
             f'beside it; the generator reads only the audited JSONs, so no number in it is '
             f'hand-typed. The archive-only Part 1 is '
             f'`experiments/q-trajdirs/checks/dense-from-archives.json`, produced by '
             f'`dense_from_archives.py` with no GPU.\n')
    for path in a.archive_json:
        p2 = Path(path)
        if p2.exists():
            d = json.loads(p2.read_text())
            L.append(f'\nRaw archive `{p2.parent.name}` Git-tracked as bounded chunks: whole '
                     f'SHA256 `{d["sha256"]}` ({len(d["chunks"])} chunks).\n')

    text = ''.join(x if x.endswith('\n\n') else x + '\n' for x in L)
    print(text)
    if a.append:
        with LOG.open('a') as fh:
            fh.write('\n' + text)
        print(f'APPENDED to {LOG}')


if __name__ == '__main__':
    main()
