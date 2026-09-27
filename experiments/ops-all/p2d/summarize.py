"""Build results.json and REPORT.md from the pulled panel and training JSONs (never hand-typed numbers).

    python summarize.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / 'runs'
sys.path.insert(0, str(HERE / 'cluster'))
import scheduler as S  # noqa: E402

REF = json.loads((HERE / 'configs' / 'table1_reference.json').read_text())
FAMS = ['fno', 'unet', 'transolver', 'deeponet']
LABEL = dict(fno='FNO', unet='U-Net', transolver='Transolver', deeponet='DeepONet')


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def log_tail(job):
    outs = sorted((RUNS / job / 'pull' / 'logs').glob('*.out')) if (RUNS / job / 'pull' / 'logs').exists() else []
    errs = sorted((RUNS / job / 'pull' / 'logs').glob('*.err')) if (RUNS / job / 'pull' / 'logs').exists() else []
    txt = ''.join(p.read_text(errors='replace')[-1500:] for p in outs + errs)
    return txt


def training_record(it, m):
    stem = m.replace('.json', '')
    job = it['job']
    base = RUNS / job / 'pull' / 'output'
    res = load(base / 'result.json') if job in S.LEGACY_SINGLE else load(base / stem / 'result.json')
    prov = load(base / 'provenance.json') if job in S.LEGACY_SINGLE else load(base / stem / 'provenance.json')
    jid = (RUNS / job / 'JOBID.txt').read_text().strip() if (RUNS / job / 'JOBID.txt').exists() else None
    rec = dict(training_job=job, training_job_id=jid, config=m, co_scheduled=[x.replace('.json', '') for x in it['members']],
               training_gpu=(prov or {}).get('environment', {}).get('gpu'))
    if res is None:
        rec['training_status'] = 'no result (job running, not submitted, or died before writing result.json)'
        if (RUNS / job / 'PULLED').exists():
            tail = log_tail(job)
            rec['training_status'] = 'no result.json; log tail: ' + tail[-600:]
        return rec
    rec.update(training_status='complete' if res.get('complete') else f"failed: {res.get('failure')}",
               failure_message=res.get('failure_message'), epochs=res.get('epochs_completed'),
               steps=res.get('optimisation_steps'), stop_reason=res.get('stop_reason'), micro_batch=res.get('micro_batch'),
               best_epoch=res.get('best_epoch'), training_seconds=res.get('training_seconds'),
               real_parameter_count=res.get('real_parameter_count'), train_count=(prov or {}).get('train_count'),
               validation_worst=(res.get('validation_best_checkpoint') or {}).get('worst'),
               validation_mean=(res.get('validation_best_checkpoint') or {}).get('mean'),
               peak_allocated_bytes=res.get('peak_allocated_bytes'), config_used=res.get('config'))
    return rec


def fastest_cg_at_most(subj, err):
    ok = [(v['gpu_ms'], k) for k, v in subj.items() if v['family'] == 'cg' and v['cg_converged'] and v['worst_error'] <= err]
    return min(ok)[1] if ok else None


def cell(problem, mesh, pre):
    ref = REF[problem][str(mesh)]
    acc, fast, named = ref['accurate'], ref['fast'], ref['named_fom']
    pj = S.PANEL_NAME.get(pre, f'{pre}p')
    out = dict(problem=problem, mesh=mesh, cell=pre, table1_source_job=ref['source_job'], nmrom_accurate=acc, nmrom_fast=fast,
               named_fom=named, cohort=None)
    items = [it for it in S.train_items() if it['cell'] == pre]
    trainrec = {}
    for it in items:
        for m in it['members']:
            trainrec[m.replace('.json', '').split('_')[0]] = training_record(it, m)
    res = load(RUNS / pj / 'pull' / 'output' / 'result.json')
    out['panel_job'] = pj
    out['panel_job_id'] = (RUNS / pj / 'JOBID.txt').read_text().strip() if (RUNS / pj / 'JOBID.txt').exists() else None
    if res is None or not res.get('complete'):
        out['status'] = 'panel not complete' if out['panel_job_id'] else 'panel not submitted'
        if (RUNS / pj / 'PULLED').exists():
            out['panel_log_tail'] = log_tail(pj)[-800:]
        out['operators'] = {f: trainrec.get(f, {}) for f in FAMS}
        return out
    subj = res['subjects']
    out.update(status='complete', gpu=res['gpu'], gpu_uuid=res['gpu_uuid'], node=res.get('node'),
               cohort=dict(role=res['cohort'].get('role'), cases=len(res['cohort']['parameters']), note=res['cohort'].get('note')),
               gates=res['gates'], reproduction=res.get('reproduction'), drift=res.get('drift', {}).get('ratios'),
               elapsed_seconds=res.get('elapsed_seconds'))
    rule = fastest_cg_at_most(subj, subj[acc]['worst_error'])
    out['fom_named'] = dict(name=named, ms=subj[named]['gpu_ms'], worst_pct=100 * subj[named]['worst_error'],
                            median_pct=100 * subj[named]['median_error'], iterations_note='unpreconditioned CG')
    out['fom_rule'] = dict(rule='fastest tested CG with worst error <= NM-ROM accurate worst error (same job)', name=rule,
                           ms=subj[rule]['gpu_ms'] if rule else None, worst_pct=100 * subj[rule]['worst_error'] if rule else None)
    fom_ms = subj[named]['gpu_ms']
    for key, name in (('nmrom_accurate_result', acc), ('nmrom_fast_result', fast)):
        v = subj[name]
        out[key] = dict(arm=name, ms=v['gpu_ms'], worst_pct=100 * v['worst_error'], median_pct=100 * v['median_error'],
                        speedup_vs_named_fom=fom_ms / v['gpu_ms'], deterministic=v['deterministic'],
                        table1_worst_pct=100 * ref[name]['worst'], table1_gpu_ms=ref[name]['gpu_ms'])
    out['cg_grid'] = {k: dict(ms=v['gpu_ms'], worst_pct=100 * v['worst_error'], median_pct=100 * v['median_error'],
                              converged=v['cg_converged']) for k, v in subj.items() if v['family'] == 'cg'}
    ops = {}
    panel_ops = {o['family']: o for o in res.get('operators', [])}
    for f in FAMS:
        rec = dict(trainrec.get(f, {}))
        po = panel_ops.get(f, {})
        rec['panel_status'] = po.get('status')
        if po.get('reason'):
            rec['panel_reason'] = po['reason']
        name = f'op_{f}'
        if name in subj:
            v = subj[name]
            own = fastest_cg_at_most(subj, v['worst_error'])
            rec.update(worst_pct=100 * v['worst_error'], median_pct=100 * v['median_error'], ms=v['gpu_ms'],
                       ms_A1=v['gpu_ms_A1'], ms_A2=v['gpu_ms_A2'], speedup_vs_named_fom=fom_ms / v['gpu_ms'],
                       own_matched_cg=own, speedup_vs_own_matched_cg=(subj[own]['gpu_ms'] / v['gpu_ms']) if own else None,
                       max_relative_to_first_rep=v['max_relative_to_first'], all_finite=v['all_finite'],
                       checkpoint_sha256=po.get('checkpoint_sha256'), panel_param_count=po.get('real_parameter_count'))
        ops[f] = rec
    out['operators'] = ops
    return out


def fmt(x, nd=2):
    if x is None:
        return '—'
    if isinstance(x, float):
        return f'{x:.{nd}f}' if abs(x) >= 0.01 or x == 0 else f'{x:.2e}'
    return str(x)


def op_cell(o):
    if 'worst_pct' in o:
        tag = ' ᶜ' if len(o.get('co_scheduled', [])) > 1 else ''
        return f"{fmt(o['worst_pct'])} / {fmt(o['median_pct'])} / {fmt(o['ms'], 3)}{tag}"
    st = o.get('panel_status') or o.get('training_status') or 'pending'
    if o.get('training_status', '').startswith('failed'):
        st = o['training_status']
    return st.replace('|', '/')[:60]


def write_cost_points(cells):
    """Coordinator schema (shared plotting script), values copied from the pulled panel records."""
    out = []
    rerun_done = any(c['cell'] == 's1024r' and c.get('status') == 'complete' for c in cells)
    for c in cells:
        if c['cell'] == 's1024' and rerun_done:
            continue   # superseded in cost_points by the one-network-per-GPU rerun s1024r (DESIGN A6); kept in results.json
        res = load(RUNS / c['panel_job'] / 'pull' / 'output' / 'result.json') if c.get('panel_job') else None
        done = bool(res and res.get('complete'))
        e = dict(pde='Poisson 2D' if c['problem'] == 'square' else 'Poisson L-shape 2D', mesh=c['mesh'],
                 complete=bool(done and all(v is not False for v in res['gates'].values())),
                 panel_job=c.get('panel_job_id'), gpu=res['gpu'] if done else None, cohort='development',
                 error_metric='worst same-grid relative L2 over the cohort (steady problem)', nmrom=[], fom=[], operators={})
        if done:
            subj = res['subjects']
            for k, v in subj.items():
                if v['family'] == 'nm-rom':
                    e['nmrom'].append(dict(setting=k, worst_percent=100 * v['worst_error'], gpu_ms=v['gpu_ms']))
                elif v['family'] == 'cg':
                    e['fom'].append(dict(setting=k, worst_percent=100 * v['worst_error'], gpu_ms=v['gpu_ms'],
                                         unstable=not v['cg_converged'], is_reference=(k == c['named_fom'])))
        for f in FAMS:
            o = c['operators'].get(f, {})
            ok = 'worst_pct' in o
            fail = None
            if not ok:
                fail = o.get('panel_reason') or o.get('panel_status') or o.get('training_status') or 'pending'
                if (o.get('training_status') or '').startswith('failed'):
                    fail = f"training {o['training_status']} ({o.get('failure_message') or ''})".strip()
                elif o.get('panel_status') == 'query_out_of_memory':
                    fail = 'query out of memory in the panel (beside the NM-ROM bank on an 80 GB card)'
            e['operators'][LABEL[f]] = dict(ok=ok, worst_percent=o.get('worst_pct'), gpu_ms=o.get('ms'), failure=fail)
        out.append(e)
    (HERE / 'cost_points.json').write_text(json.dumps(dict(group='p2d', cells=out), indent=1) + '\n')


def main():
    cells = [cell(p, m, pre) for p, m, pre, _, _ in sorted(S.CELLS, key=lambda c: (c[0] != 'square', c[1]))]
    (HERE / 'results.json').write_text(json.dumps(dict(
        generated_by='summarize.py', protocol='../../../OPS-ALL-PROTOCOL.md', design='DESIGN.md',
        units=dict(ms='median GPU-query milliseconds over A1 u A2 (operators, NM-ROM) or B (CG)', pct='relative L2 x 100'),
        cells=cells), indent=1) + '\n')
    L = ['# ops-all p2d — neural operators on the Poisson 2D and L-shape Table 1 cells', '',
         'Generated by `summarize.py` from the pulled panel/training JSONs; do not edit by hand. Status of every number: '
         'provisional group result (not yet audited by the coordinator). See the caveats below the table.', '',
         '| problem | mesh | panel job (GPU) | FOM named: ms / err % | NM-ROM accurate: ms / err % (× FOM) | NM-ROM fast: ms / err % (× FOM) | '
         'FNO worst % / median % / ms | U-Net | Transolver | DeepONet |', '|---|---|---|---|---|---|---|---|---|---|']
    for c in cells:
        if c.get('status') != 'complete':
            ops = ' | '.join(op_cell(c['operators'].get(f, {})) for f in FAMS)
            L.append(f"| {c['problem']} | {c['mesh']}² ({c['cell']}) | {c['panel_job']} ({c['status']}) | — | — | — | {ops} |")
            continue
        fn, a, f_ = c['fom_named'], c['nmrom_accurate_result'], c['nmrom_fast_result']
        ops = ' | '.join(op_cell(c['operators'][f]) for f in FAMS)
        L.append(f"| {c['problem']} | {c['mesh']}² ({c['cell']}) | {c['panel_job_id']} ({c['gpu']}) | {fn['name']}: {fmt(fn['ms'])} / {fmt(fn['worst_pct'])} | "
                 f"{fmt(a['ms'], 3)} / {fmt(a['worst_pct'])} ({fmt(a['speedup_vs_named_fom'], 1)}×) | "
                 f"{fmt(f_['ms'], 3)} / {fmt(f_['worst_pct'])} ({fmt(f_['speedup_vs_named_fom'], 1)}×) | {ops} |")
    L += ['', 'ᶜ = the network was trained co-scheduled with other networks on one GPU (DESIGN A1). Operator cells: worst % / '
          'median % / median GPU ms over the Table 1 cohort, same job as the FOM and NM-ROM.', '',
          '## Per-cell detail', '']
    for c in cells:
        L.append(f"### {c['problem']} {c['mesh']}² ({c['cell']}{', one network per GPU rerun' if c['cell'] == 's1024r' else ''}) — {c.get('status')}")
        if c.get('status') == 'complete':
            L.append(f"Panel job {c['panel_job_id']} on {c['gpu']} ({c.get('node')}); cohort {c['cohort']['role']} "
                     f"({c['cohort']['cases']} cases). Gates: {c['gates']}. FOM rule (fastest CG ≤ accurate error): "
                     f"{c['fom_rule']['name']} {fmt(c['fom_rule']['ms'])} ms / {fmt(c['fom_rule']['worst_pct'])} %. "
                     f"Table 1 source job {c['table1_source_job']}: accurate {fmt(c['nmrom_accurate_result']['table1_worst_pct'])} % "
                     f"at {fmt(c['nmrom_accurate_result']['table1_gpu_ms'], 3)} ms there, fast {fmt(c['nmrom_fast_result']['table1_worst_pct'])} % "
                     f"at {fmt(c['nmrom_fast_result']['table1_gpu_ms'], 3)} ms there.")
            L.append('')
            L.append('| operator | config | train job (GPU) | co-scheduled | epochs | stop | worst % | median % | ms | × named FOM | own matched CG | × own |')
            L.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
            for f in FAMS:
                o = c['operators'][f]
                L.append(f"| {LABEL[f]} | {o.get('config')} | {o.get('training_job_id')} ({o.get('training_gpu')}) | "
                         f"{','.join(o.get('co_scheduled', []))} | {o.get('epochs')} | {o.get('stop_reason')} | {fmt(o.get('worst_pct'))} | "
                         f"{fmt(o.get('median_pct'))} | {fmt(o.get('ms'), 3)} | {fmt(o.get('speedup_vs_named_fom'), 2)} | "
                         f"{o.get('own_matched_cg') or '—'} | {fmt(o.get('speedup_vs_own_matched_cg'), 2)} |"
                         + ('' if 'worst_pct' in o else f" status: {op_cell(o)} |"))
        else:
            for f in FAMS:
                o = c['operators'].get(f, {})
                L.append(f"- {LABEL[f]}: job {o.get('training_job')} {o.get('training_job_id')}: {op_cell(o)}")
        L.append('')
    write_cost_points(cells)
    caveats = (HERE / 'CAVEATS.md').read_text() if (HERE / 'CAVEATS.md').exists() else ''
    L += ['## Caveats', '', caveats, '', '## Glossary', '',
          '- **FOM named**: the full-order solver setting printed in paper Table 1 for that row (unpreconditioned CG at the named relative tolerance), re-timed in this job.',
          '- **NM-ROM accurate / fast**: the two frozen Table 1 settings (square: span of the first 512 / 128 ordered bank functions; L-shape: 128 / 64), same code as the Table 1 job.',
          '- **worst % / median %**: worst and median over the cohort cases of the same-grid relative L2 error, in percent.',
          '- **ms**: median GPU-query time (device work between the synchronised source upload and the field download), milliseconds.',
          '- **× named FOM**: named-FOM time divided by the subject time, same job. **own matched CG**: the fastest tested CG setting at least as accurate as the subject.',
          '- **co-scheduled**: networks trained at the same time as separate processes on one GPU (each with the full 3000 s wall budget).',
          '- **stop**: why training ended (`wall_budget` = the 3000 s budget).', '- **cohort**: the Table 1 evaluation cases (development sources), never used for any choice.']
    (HERE / 'REPORT.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L[:20]))


if __name__ == '__main__':
    main()
