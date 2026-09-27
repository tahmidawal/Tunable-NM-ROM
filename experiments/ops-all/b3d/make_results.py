"""Generate results.json and REPORT.md from the pulled panel/training JSONs (runs/<job>/...). No hand-typed numbers."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = json.loads((HERE / 'configs' / 'table1_heldout_expected.json').read_text())
TABLE1_FOM = {'65': 'fom_dt0.005_nt0.001_lt0.1', '129': 'fom_dt0.01_nt0.01_lt0.1', '257': 'fom_dt0.01_nt0.01_lt0.1'}
MESHNAME = {'65': '64^3', '129': '128^3', '257': '256^3'}
FAMILY = {'fno': 'FNO', 'unet': 'U-Net', 'transolver': 'Transolver', 'deeponet': 'DeepONet'}
CELLS = {'65': 'pn65', '129': 'c129', '257': 'p257b'}
TRAIN_ONLY = {'65': 'c65', '257': 't257'}


def find(job, *parts):
    p = HERE / 'runs' / job / 'output'
    for x in parts:
        p = p / x
    return p if p.exists() else None


def jobid(job):
    logs = sorted((HERE / 'runs' / job / 'logs').glob('*.out')) if (HERE / 'runs' / job / 'logs').exists() else []
    return logs[-1].stem if logs else None


def panel_failure(job):
    """Last error line of a job's stderr + PANEL_EXIT from stdout (pulled logs only)."""
    d = HERE / 'runs' / job / 'logs'
    if not d.exists():
        return None
    out = ' '.join(f.read_text() for f in d.glob('*.out'))
    if 'PANEL_EXIT=' not in out or 'PANEL_EXIT=0' in out:
        return None
    errs = [l for f in d.glob('*.err') for l in f.read_text().splitlines() if 'Error' in l]
    return (errs[-1].strip() if errs else 'panel exited non-zero')[:400]


def training_failure(mesh, arm):
    for job in (CELLS[mesh], TRAIN_ONLY.get(mesh)):
        if job and (p := find(job, 'train', arm, 'result.json')):
            return json.loads(p.read_text())
    return None


def cell(mesh):
    out = dict(mesh=MESHNAME[mesh], nodes=int(mesh), cohort='sealed held-out 923901 x 32 (Table 1 cohort, opened before)',
               table1_job=EXP[mesh]['job'], table1_gpu=EXP[mesh]['gpu'])
    pr = find(CELLS[mesh], 'panel', 'result.json')
    if pr is None or not json.loads(pr.read_text()).get('complete'):
        pf = panel_failure(CELLS[mesh])
        out['status'] = f'FAILED (panel job {jobid(CELLS[mesh])}): {pf}' if pf else 'panel not collected'
        out['panel_failure'] = pf
        out['training_only'] = {}
        for tj in (TRAIN_ONLY.get(mesh), CELLS[mesh]):
            if not (tj and find(tj, 'train')):
                continue
            for rj in sorted(find(tj, 'train').glob('*/result.json')):
                t = json.loads(rj.read_text())
                v = t.get('validation_best_checkpoint') or {}
                out['training_only'][t['arm'] + '@' + tj] = dict(job=jobid(tj), gpu=json.loads((find(tj, 'train', 'data.json') or rj).read_text()).get('environment', {}).get('gpu'), trained=t.get('trained'), failure=t.get('failure'),
                                                      epochs=t.get('epochs_completed'), stop_reason=t.get('stop_reason'),
                                                      micro_batch=t.get('micro_batch'),
                                                      validation_mean_case_max_pct=100 * v['mean_case_max'] if v else None,
                                                      validation_worst_case_max_pct=100 * v['worst_case_max'] if v else None)
        return out
    r = json.loads(pr.read_text())
    out.update(status='complete' if r.get('complete') else 'incomplete', job_id=r.get('job_id'), gpu=r.get('gpu'),
               training_jobs=sorted({j for j in ((jobid(CELLS[mesh]) if find(CELLS[mesh], 'train') else None), jobid(TRAIN_ONLY.get(mesh, 'none'))) if j}))
    if not r.get('complete'):
        return out
    tim = r['timing']['summary']
    arms = list(r['arms'])
    acc, fast = arms[0], arms[1]
    accw = r['arms'][acc]['worst_evolved']
    elig = [k for k, v in r['fom'].items() if v['all_finite'] and v['worst_evolved'] <= accw]
    fom = min(elig, key=lambda k: tim[k]['median_ms']) if elig else None
    fms = tim[fom]['median_ms'] if fom else None
    out['gates'] = {k: r['gates'][k] for k in ('timing_drift_worst', 'timing_drift_pass', 'timing_neighbour_ratio',
                                                'timing_neighbour_ratio_fom', 'timing_neighbour_pass',
                                                'deterministic_outputs', 'deterministic_pass', 'reference_residual')}
    out['gates']['nmrom_reproduction_max_rel'] = max(v['rel'] for v in r['gates']['nmrom_reproduction'].values())
    out['fom'] = dict(rule='fastest FOM setting in this job with worst <= NM-ROM accurate worst', setting=fom,
                      ms=fms, worst_pct=100 * r['fom'][fom]['worst_evolved'] if fom else None,
                      median_pct=100 * r['fom'][fom]['median_worst_evolved'] if fom else None,
                      table1_setting=TABLE1_FOM[mesh], table1_setting_ms=tim[TABLE1_FOM[mesh]]['median_ms'],
                      table1_setting_worst_pct=100 * r['fom'][TABLE1_FOM[mesh]]['worst_evolved'])
    for tag, a in (('nmrom_accurate', acc), ('nmrom_fast', fast)):
        out[tag] = dict(arm=a, worst_pct=100 * r['arms'][a]['worst_evolved'],
                        median_pct=100 * r['arms'][a]['median_worst_evolved'], ms=tim[a]['median_ms'],
                        speedup=fms / tim[a]['median_ms'] if fms else None,
                        speedup_vs_table1_fom_setting=tim[TABLE1_FOM[mesh]]['median_ms'] / tim[a]['median_ms'],
                        table1_worst_pct=100 * EXP[mesh]['nmrom_worst'][a])
    ops = {}
    for arm, rec in r.get('operators', {}).items():
        tr = rec.get('training', {})
        fam = FAMILY[tr.get('family') or tr.get('config', {}).get('family', '?')] if (tr.get('family') or tr.get('config')) else arm
        o = dict(arm=arm, status=rec.get('status'), training_dir=rec.get('training_dir'),
                 parameters=tr.get('real_parameter_count'), epochs=tr.get('epochs_completed'),
                 optimisation_steps=tr.get('optimisation_steps'), stop_reason=tr.get('stop_reason'),
                 micro_batch=tr.get('micro_batch'), training_seconds=tr.get('training_seconds'),
                 validation_mean_case_max_pct=100 * tr['validation']['mean_case_max'] if tr.get('validation') else None)
        if rec.get('status') == 'evaluated':
            o.update(worst_pct=100 * rec['worst_evolved'], median_pct=100 * rec['median_worst_evolved'],
                     ms=rec['median_ms'], speedup=fms / rec['median_ms'] if fms else None,
                     speedup_vs_table1_fom_setting=tim[TABLE1_FOM[mesh]]['median_ms'] / rec['median_ms'])
        else:
            o['failure'] = rec.get('failure') or tr.get('failure') or 'not trained'
        ops[fam] = o
    out['operators'] = ops
    return out


def cost_point(mesh):
    """Coordinator schema (cost_points.json); values copied from the pulled panel JSON only."""
    c = dict(pde='Burgers 3D', mesh=int(mesh) - 1, complete=False, panel_job=None, gpu=None, cohort='held-out',
             error_metric='worst over 32 cases of max over the five evolved times of same-grid relative L2 '
                          '(normalised by the initial field)', nmrom=[], fom=[],
             operators={f: dict(ok=False, worst_percent=None, gpu_ms=None, failure='panel not run yet')
                        for f in ('FNO', 'U-Net', 'Transolver', 'DeepONet')})
    pr = find(CELLS[mesh], 'panel', 'result.json')
    if pr is None or not json.loads(pr.read_text()).get('complete'):
        pf = panel_failure(CELLS[mesh])
        if pf:
            c.update(panel_job=jobid(CELLS[mesh]), gpu='NVIDIA H200', panel_failure=pf)
            for f in c['operators']:
                c['operators'][f]['failure'] = 'cell panel failed: ' + pf
        return c
    r = json.loads(pr.read_text())
    c.update(panel_job=str(r.get('job_id')), gpu=r.get('gpu'))
    if not r.get('complete'):
        return c
    g = r['gates']
    c['complete'] = bool(g['timing_drift_pass'] and g['timing_neighbour_pass'] and g['deterministic_pass'])
    tim = r['timing']['summary']
    arms = list(r['arms'])
    accw = r['arms'][arms[0]]['worst_evolved']
    elig = [k for k, v in r['fom'].items() if v['all_finite'] and v['worst_evolved'] <= accw]
    ref = min(elig, key=lambda k: tim[k]['median_ms']) if elig else None
    c['nmrom'] = [dict(setting=('accurate ' if i == 0 else 'fast ') + a, worst_percent=100 * r['arms'][a]['worst_evolved'],
                       gpu_ms=tim[a]['median_ms']) for i, a in enumerate(arms)]
    c['fom'] = [dict(setting=k, worst_percent=100 * v['worst_evolved'], gpu_ms=tim[k]['median_ms'],
                     unstable=not v['all_finite'], is_reference=False, table1_comparator=(k == ref)) for k, v in r['fom'].items()]
    for arm, rec in r.get('operators', {}).items():
        tr = rec.get('training', {})
        fam = FAMILY[tr.get('family') or tr.get('config', {}).get('family')]
        if rec.get('status') == 'evaluated':
            c['operators'][fam] = dict(ok=True, worst_percent=100 * rec['worst_evolved'], gpu_ms=rec['median_ms'], failure=None)
        else:
            c['operators'][fam] = dict(ok=False, worst_percent=None, gpu_ms=None,
                                       failure=rec.get('failure') or tr.get('failure') or rec.get('status'))
    c['reference_note'] = ('truth = same-grid Newton-BiCGStab dt 0.005 ntol 1e-10 ltol 1e-11, computed in the panel but '
                           'not timed, so it is not in the fom list (no entry has is_reference=true)')
    return c


def fmt(x, d=2):
    return '—' if x is None else (f'{x:.{d}f}' if abs(x) < 1000 else f'{x:.0f}')


def main():
    cells = {MESHNAME[m]: cell(m) for m in ('65', '129', '257')}
    (HERE / 'cost_points.json').write_text(json.dumps(dict(group='b3d', cells=[cost_point(m) for m in ('65', '129', '257')]),
                                                      indent=1) + '\n')
    (HERE / 'results.json').write_text(json.dumps(dict(group='C b3d', cells=cells), indent=1) + '\n')
    L = ['# Burgers 3D — neural operators vs NM-ROM and Newton–BiCGStab (Table 1 held-out cohort)', '',
         'Group C (`b3d`) of `OPS-ALL-PROTOCOL.md`. Generated by `make_results.py` from the pulled job JSONs; '
         'numbers final for every cell marked complete. Design: `DESIGN.md`.', '',
         '| mesh | job / GPU | FOM (rule) ms / worst % | NM-ROM acc. worst % / ms / speedup | NM-ROM fast worst % / ms / speedup | '
         'FNO | U-Net | Transolver | DeepONet |', '|---|---|---|---|---|---|---|---|---|']
    for name, c in cells.items():
        if c.get('status') != 'complete':
            L.append(f'| {name} | {c.get("job_id", "—")} | {c["status"]} | | | | | | |')
            for arm, t in c.get('training_only', {}).items():
                L.append(f'|  | training {t["job"]} ({t["gpu"]}) | {arm}: ' + (f'trained, {t["epochs"]} epochs, stop {t["stop_reason"]}, '
                         f'validation mean/worst case-max {fmt(t["validation_mean_case_max_pct"])} / {fmt(t["validation_worst_case_max_pct"])} %'
                         if t['trained'] else f'NOT TRAINED: {t["failure"]}') + ' | | | | | | |')
            continue
        f, a, s = c['fom'], c['nmrom_accurate'], c['nmrom_fast']
        row = [name, f'{c["job_id"]} / {c["gpu"]}', f'{fmt(f["ms"])} / {fmt(f["worst_pct"])} ({f["setting"]})',
               f'{fmt(a["worst_pct"])} / {fmt(a["ms"])} / {fmt(a["speedup"])}×',
               f'{fmt(s["worst_pct"])} / {fmt(s["ms"])} / {fmt(s["speedup"])}×']
        for fam in ('FNO', 'U-Net', 'Transolver', 'DeepONet'):
            o = c['operators'].get(fam)
            if o is None:
                row.append('—')
            elif 'worst_pct' in o:
                row.append(f'{o["arm"]}: {fmt(o["worst_pct"])} / {fmt(o["median_pct"])} / {fmt(o["ms"])} ms ({fmt(o["speedup"])}×)')
            else:
                row.append(f'{o["arm"]}: not trained ({o["failure"][:80]})')
        L.append('| ' + ' | '.join(row) + ' |')
    L += ['', 'Operator cells: worst % / median % over the 32 cases / median GPU ms (speedup vs FOM rule, same job).', '',
          '| mesh | Table 1 FOM setting (this job) ms / worst % | speedup vs it: NM-ROM acc. / fast / FNO / U-Net / Transolver / DeepONet |',
          '|---|---|---|']
    for name, c in cells.items():
        if c.get('status') == 'complete':
            f = c['fom']
            sp = [c['nmrom_accurate'], c['nmrom_fast']] + [c['operators'].get(k, {}) for k in ('FNO', 'U-Net', 'Transolver', 'DeepONet')]
            L.append(f'| {name} | {f["table1_setting"]}: {fmt(f["table1_setting_ms"])} / {fmt(f["table1_setting_worst_pct"])} | '
                     + ' / '.join(fmt(x.get('speedup_vs_table1_fom_setting')) + '×' for x in sp) + ' |')
    L += ['']
    L += ['## Gates and training details', '']
    for name, c in cells.items():
        if c.get('status') != 'complete':
            continue
        g = c['gates']
        L.append(f'* **{name}** (job {c["job_id"]}, {c["gpu"]}; training jobs {", ".join(c["training_jobs"])}): timing drift '
                 f'{g["timing_drift_worst"]:.3f} (pass {g["timing_drift_pass"]}), neighbour {g["timing_neighbour_ratio"]:.3f}/'
                 f'{g["timing_neighbour_ratio_fom"]:.3f} (pass {g["timing_neighbour_pass"]}), determinism {g["deterministic_outputs"]:.1e}; '
                 f'NM-ROM worst errors vs Table 1 job {c["table1_job"]}: max rel. diff {g["nmrom_reproduction_max_rel"]:.1e}. '
                 f'Table 1 FOM setting {c["fom"]["table1_setting"]}: {fmt(c["fom"]["table1_setting_ms"])} ms, '
                 f'{fmt(c["fom"]["table1_setting_worst_pct"])} %.')
        for fam, o in c['operators'].items():
            L.append(f'  * {fam} {o["arm"]}: {o["parameters"]} params, micro-batch {o["micro_batch"]}, {o["epochs"]} epochs / '
                     f'{o["optimisation_steps"]} steps, stop {o["stop_reason"]}, validation mean case-max '
                     f'{fmt(o["validation_mean_case_max_pct"])} %' + (f'; FAILURE: {o["failure"]}' if 'failure' in o else ''))
    L += ['', '## Caveats', '', (HERE / 'CAVEATS.md').read_text() if (HERE / 'CAVEATS.md').exists() else '', '',
          '## Glossary', '',
          '* **worst %** — largest over the 32 held-out cases of the per-case maximum, over the five evolved output times, '
          'of the relative L2 error against the same-grid tight Newton–BiCGStab reference (normalised by the initial field).',
          '* **median %** — median over cases of that per-case maximum.',
          '* **ms** — median GPU time of one query (initial field on the GPU → six output fields on the GPU), same job.',
          '* **speedup** — FOM (rule) ms / method ms, same job.',
          '* **FOM (rule)** — the fastest full-order Newton–BiCGStab setting in this job at least as accurate (worst) as '
          'the NM-ROM accurate setting (the Table 1 rule).',
          '* **accurate / fast** — the two frozen NM-ROM settings of Table 1 (span width R′ and time step).',
          '* **micro-batch** — samples per backward pass (8-sample batches accumulated when 8 do not fit).']
    (HERE / 'REPORT.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L[:12]))


if __name__ == '__main__':
    main()
