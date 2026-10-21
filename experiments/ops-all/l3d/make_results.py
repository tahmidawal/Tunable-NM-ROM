"""results.json + REPORT.md for group l3d, generated only from pulled files:
runs/<panel job>/output/summary.json (per cell) and runs/<train job>/output/<cell>/<arm>/result.json.
Cells without a panel are listed with their state. No number is typed by hand.
Recomputes from the raw timing samples / per-case errors: medians, worst/median errors, the FOM rule, speedups,
and checks them against the panel's own summary (audit)."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE / 'runs'
CELLS = [('poisson', n) for n in (32, 64, 128, 256)] + [('heat', n) for n in (32, 64, 128, 256)]
TRAIN = {('poisson', 32): 'tr_p3264', ('poisson', 64): 'tr_p3264', ('heat', 32): 'tr_h3264', ('heat', 64): 'tr_h3264',
         ('poisson', 128): 'tr_p128', ('heat', 128): 'tr_h128', ('poisson', 256): 'tr_256merged', ('heat', 256): 'tr_256merged'}
FAM = {'fno': 'FNO', 'unet': 'U-Net', 'transolver': 'Transolver', 'deeponet': 'DeepONet'}
STATE = json.loads((RUNS / 'STATE.json').read_text()) if (RUNS / 'STATE.json').exists() else {}


def job_meta(job):
    p = RUNS / job / 'JOBID'
    return p.read_text().strip() if p.exists() else None


def training(problem, n):
    job = TRAIN[(problem, n)]
    d = RUNS / job / 'output' / f'{problem}{n}'
    out = {}
    if not d.exists():
        return job, out
    for arm in sorted(p for p in d.iterdir() if p.is_dir()):
        r = arm / 'result.json'
        out[arm.name] = json.loads(r.read_text()) if r.exists() else None
        pr = arm / 'whole_gpu_probe.json'
        if out[arm.name] is not None and pr.exists() and not out[arm.name].get('complete'):
            q = json.loads(pr.read_text())
            peak = q['probe_peak_bytes'].get('1')
            out[arm.name] = dict(out[arm.name], failure=(
                f"not trained: no micro-batch fit its {out[arm.name]['memory_fraction']:.2f} share of a shared H200 "
                f"(job {out[arm.name]['job_id']}); whole-H200 probe (job {q['job_id']}) fits micro-batch 1 at "
                f"{peak / 2**30:.0f} GiB peak, so it needs a dedicated H200, not granted (GPU cap 2, DESIGN A5)"))
    return job, out


def audit(s):
    """Recompute rows from raw data; return max relative mismatch."""
    worst = 0.0
    rows = s['rows']
    for m, a in s['accuracy'].items():
        if a['failure'] or not a['errors']:
            continue
        e = np.asarray(a['errors']).max(1)
        worst = max(worst, abs(100 * e.max() - rows[m]['worst_pct']) / rows[m]['worst_pct'])
        ms = [x['ms'] for x in s['timing_samples'].get(m, [])]
        if ms:
            worst = max(worst, abs(np.median(ms) - rows[m]['median_ms']) / rows[m]['median_ms'])
    acc = next(m for m in rows if m.startswith('nmrom_accurate'))
    pool = [m for m in rows if rows[m]['kind'] == 'fom' and rows[m].get('checks_all_passed') and rows[m].get('finite')
            and 'median_ms' in rows[m] and rows[m]['worst_pct'] <= rows[acc]['worst_pct']]
    fom = min(pool, key=lambda m: rows[m]['median_ms']) if pool else None
    return worst, fom == s['fom_rule']['chosen']


def main():
    cells = []
    for problem, n in CELLS:
        key = f'{problem}{n}'
        tjob, tr = training(problem, n)
        cell = dict(cell=key, problem=problem, mesh=n, train_job=tjob, train_job_id=job_meta(tjob),
                    panel_job=f'pn_{key}', panel_job_id=job_meta(f'pn_{key}'), state=STATE.get(key))
        sp = RUNS / f'pn_{key}' / 'output' / 'summary.json'
        s = json.loads(sp.read_text()) if sp.exists() else None
        if s and s.get('complete'):
            rows = s['rows']
            acc = next(m for m in rows if m.startswith('nmrom_accurate'))
            fast = next(m for m in rows if m.startswith('nmrom_fast'))
            fom = s['fom_rule']['chosen']
            mism, fom_ok = audit(s)
            cell.update(gpu=s['gpu'], cohort=s['cohort']['cohort'], cases=s['cohort']['count'], status=s['status'],
                        gates=s['gates'], reproduction=s['reproduction'],
                        audit=dict(max_relative_mismatch=mism, fom_rule_recomputed_equal=fom_ok),
                        fom=dict(setting=fom, ms=rows[fom]['median_ms'] if fom else None, worst_pct=rows[fom]['worst_pct'] if fom else None),
                        nmrom_accurate=dict(setting=acc, ms=rows[acc]['median_ms'], worst_pct=rows[acc]['worst_pct'],
                                            speedup=rows[acc].get('speedup_vs_fom')),
                        nmrom_fast=dict(setting=fast, ms=rows[fast]['median_ms'], worst_pct=rows[fast]['worst_pct'],
                                        speedup=rows[fast].get('speedup_vs_fom')),
                        fom_grid={m: dict(ms=r.get('median_ms'), worst_pct=r.get('worst_pct'), converged=r.get('checks_all_passed'))
                                  for m, r in rows.items() if r['kind'] == 'fom'})
        ops = {}
        for arm, res in tr.items():
            fam = arm.split('-')[0]
            fam = dict(fno='fno', unet='unet', tsol='transolver', don='deeponet')[fam]
            rec = dict(arm=arm)
            if res is None:
                rec['failure'] = 'training did not finish (no result.json)'
            elif not res.get('complete'):
                rec['failure'] = res.get('failure', 'incomplete')
                rec['probe_peak_bytes'] = res.get('probe_peak_bytes')
                rec['memory_fraction'] = res.get('memory_fraction')
            else:
                rec.update(parameters=res['real_parameter_count'], epochs=res['epochs_completed'], steps=res['optimisation_steps'],
                           best_epoch=res['best_epoch'], stop_reason=res['stop_reason'], micro_batch=res['micro_batch'],
                           memory_fraction=res['memory_fraction'],
                           validation_mean_pct=100 * res['validation_best_checkpoint']['mean_case'])
            if s and s.get('complete'):
                name = f'op_{fam}_{arm}'
                r = s['rows'].get(name)
                if r and 'median_ms' in r:
                    rec.update(worst_pct=r['worst_pct'], median_pct=r['median_pct'], ms=r['median_ms'],
                               speedup_vs_fom=r.get('speedup_vs_fom'), drift_ok=r.get('drift_ok'))
                elif r:
                    rec.setdefault('failure', r.get('failure') or 'no timing')
                elif name in s.get('operators', {}):
                    rec.setdefault('failure', s['operators'][name].get('status'))
            ops[fam] = rec
        cell['operators'] = ops
        cells.append(cell)
    # cost_points.json (coordinator schema, shared plotting script)
    pts = []
    for c in cells:
        rec = dict(pde='Poisson 3D' if c['problem'] == 'poisson' else 'Heat 3D', mesh=c['mesh'],
                   complete=bool(c.get('status') == 'final'), panel_job=c.get('panel_job_id'), gpu=(c.get('gpu') or '').split(',')[0] or None,
                   cohort=None, error_metric=('worst over cases of the same-grid relative L2 error' if c['problem'] == 'poisson'
                                              else 'worst over cases of the max over all six output times (t=0 included) of the same-grid relative L2 error'),
                   nmrom=[], fom=[], operators={})
        if 'fom' in c:
            rec['cohort'] = 'development' if c['cohort'] == 'development' else 'held-out'
            for k in ('nmrom_accurate', 'nmrom_fast'):
                rec['nmrom'].append(dict(setting=c[k]['setting'], worst_percent=c[k]['worst_pct'], gpu_ms=c[k]['ms']))
            for m, g in c['fom_grid'].items():
                rec['fom'].append(dict(setting=m, worst_percent=g['worst_pct'], gpu_ms=g['ms'], unstable=not bool(g['converged']),
                                       is_reference=(m == c['fom']['setting'])))
        for fam, label in FAM.items():
            o = c['operators'].get(fam)
            if o and 'ms' in o:
                rec['operators'][label] = dict(ok=True, worst_percent=o['worst_pct'], gpu_ms=o['ms'], failure=None)
            else:
                rec['operators'][label] = dict(ok=False, worst_percent=None, gpu_ms=None,
                                               failure=(o or {}).get('failure', 'not run') if o else 'not run')
        pts.append(rec)
    (HERE / 'cost_points.json').write_text(json.dumps(dict(group='l3d', cells=pts), indent=1) + '\n')
    (HERE / 'results.json').write_text(json.dumps(dict(group='l3d', cells=cells), indent=1, default=float) + '\n')

    def f(x, d=3):
        return '—' if x is None else (f'{x:.{d}g}' if isinstance(x, float) else str(x))
    lines = ['# ops-all / l3d — neural operators for the Poisson 3D and Heat 3D rows of Table 1',
             '', 'Generated by `make_results.py` from pulled panel summaries and training records (no hand-typed numbers). '
             'Status per cell as printed; see caveats in the Notes section and DESIGN.md.', '',
             '| cell | panel job | GPU | cohort | FOM setting | FOM ms | FOM err % | NM acc ms | NM acc err % | NM fast ms | NM fast err % | status |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for c in cells:
        if 'fom' not in c:
            lines.append(f"| {c['cell']} | {c['panel_job_id'] or '—'} | — | — | — | — | — | — | — | — | — | {c['state'] or 'not run'} |")
            continue
        lines.append(f"| {c['cell']} | {c['panel_job_id']} | {c['gpu'].split(',')[0]} | {c['cohort']} ({c['cases']}) | {c['fom']['setting']} | "
                     f"{f(c['fom']['ms'])} | {f(c['fom']['worst_pct'])} | {f(c['nmrom_accurate']['ms'])} | {f(c['nmrom_accurate']['worst_pct'])} | "
                     f"{f(c['nmrom_fast']['ms'])} | {f(c['nmrom_fast']['worst_pct'])} | {c['status']} |")
    lines += ['', '## Operators (worst % / median % / median GPU ms / speedup vs the cell FOM)', '',
              '| cell | ' + ' | '.join(FAM.values()) + ' |', '|---|---|---|---|---|']
    for c in cells:
        cols = []
        for fam in FAM:
            o = c['operators'].get(fam)
            if o is None:
                cols.append('not run')
            elif 'ms' in o:
                cols.append(f"{f(o['worst_pct'])} / {f(o['median_pct'])} / {f(o['ms'])} ms / {f(o.get('speedup_vs_fom'))}× ({o['arm']}, {o['epochs']} ep)")
            else:
                cols.append(f"FAILED ({o['arm']}): {o.get('failure', 'pending')}"[:160])
        lines.append(f"| {c['cell']} | " + ' | '.join(cols) + ' |')
    lines += ['', '## Gates and audit', '']
    for c in cells:
        if 'gates' in c:
            rep = c['reproduction']
            g = max((v['max_relative_gap'] for v in rep['arms'].values() if v['gated']), default=None)
            lines.append(f"- {c['cell']}: gates {c['gates']}; reproduction vs job {rep['source_job']} max gated gap {f(g)}; "
                         f"local recomputation mismatch {f(c['audit']['max_relative_mismatch'])}, FOM rule recomputed equal: {c['audit']['fom_rule_recomputed_equal']}")
    # Table-1 source-job times for the same settings (the timing comparison caveat is generated, not typed)
    lines += ['', '## Same settings, this job vs the Table-1 source job (median GPU ms)', '',
              '| cell | NM acc here / source | NM fast here / source | Table-1 FOM here / source |', '|---|---|---|---|']
    for c in cells:
        if 'fom' not in c:
            continue
        e = json.loads((HERE / 'expected' / f"{c['problem']}_{c['mesh']}.json").read_text())['median_ms']
        a, fa = c['nmrom_accurate']['setting'], c['nmrom_fast']['setting']
        srcname = lambda m: ('lin_' + m.split('_')[2] + '_cn') if c['problem'] == 'heat' else (m.split('_')[2] + '_linear')
        t1 = 'fom_cg_rtol0.01' if c['problem'] == 'poisson' else 'fom_cncg_dt0.025_rtol1e-4'
        t1s = 'cg_0.01' if c['problem'] == 'poisson' else t1
        lines.append(f"| {c['cell']} | {f(c['nmrom_accurate']['ms'])} / {f(e[srcname(a)])} | {f(c['nmrom_fast']['ms'])} / {f(e[srcname(fa)])} | "
                     f"{f(c['fom_grid'][t1]['ms'])} / {f(e[t1s])} |")
    notes = (HERE / 'NOTES.md').read_text() if (HERE / 'NOTES.md').exists() else ''
    (HERE / 'REPORT.md').write_text('\n'.join(lines) + '\n\n' + notes)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
