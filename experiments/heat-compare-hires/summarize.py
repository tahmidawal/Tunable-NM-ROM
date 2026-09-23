"""summary.json + TABLE.md for one panel run, generated only from results.json and audit.json.

usage: summarize.py <results.json> <audit.json> <out_dir> [<hires-heat h2d-final04 summary.json>]
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np

rpath, apath, odir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
raw = rpath.read_bytes(); res = json.loads(raw); audit = json.loads(apath.read_text())
ref04 = json.loads(Path(sys.argv[4]).read_text()) if len(sys.argv) > 4 else None
n = res['mesh']


def failures(row):
    fam = row['family']
    if fam in ('nmrom', 'qm'):
        bad = 0
        for s0, s1 in row['stats']:
            bad += int(np.asarray(s0).reshape(-1, 5)[-1, 2] != 1) + int((np.asarray(s1).reshape(-1, 5)[:, 2] != 1).sum())
        return bad
    if fam in ('fom', 'control') and row['stats'] and row['stats'][0]:
        return int(sum((np.asarray(s[0]).reshape(-1, 3)[:, 2] != 1).sum() for s in row['stats']))
    return 0


LABEL = {'nmrom': 'NM-ROM (ours)', 'linear_bank': 'linear bank (q=R top rung)', 'pod': 'POD', 'qm': 'quadratic manifold',
         'operator': 'neural operator', 'fom': 'CN-CG full-order', 'control': 'control'}
rows = []
for name, r in res['arms'].items():
    same = np.asarray(r['same']); ms = np.asarray(r['device_ms'])
    rows.append(dict(method=name, family=r['family'], family_label=LABEL[r['family']], unknowns=r.get('unknowns'),
                     worst_all_times=float(same.max()), median_case_max=float(np.median(same.max(1))),
                     worst_evolved=float(same[:, 1:].max()), physical_worst=float(np.max(r['physical'])),
                     device_ms_median=float(np.median(ms)), device_ms_case_medians=np.median(ms, 1).tolist(),
                     repetitions=int(ms.shape[1]), cases=int(ms.shape[0]), failures=failures(r),
                     setup_seconds=r.get('setup_seconds'), meta={k: r[k] for k in ('q', 'opt', 'gamma', 'method', 'dt', 'rtol', 'named', 'checkpoint_sha256',
                                                                           'operator_family', 'best_epoch', 'parameters', 'parameter_dtype') if k in r}))
foms = [x for x in rows if x['family'] == 'fom' and x['failures'] == 0]
named = next(x for x in rows if x['family'] == 'fom' and x['method'].endswith('_NAMED'))
assert named['failures'] == 0, 'named FOM has failed solves'
most_accurate = min(foms, key=lambda f: f['worst_all_times'])
for x in rows:
    x['speedup_vs_named'] = named['device_ms_median'] / x['device_ms_median']
    ok = [f for f in foms if f['worst_all_times'] <= x['worst_all_times']]
    if x['family'] in ('fom', 'control'):
        x['fom_chosen'] = None; x['speedup'] = None; x['fom_rule'] = 'n/a (candidate FOM or labelled control)'
        continue
    if ok:
        best = min(ok, key=lambda f: f['device_ms_median']); x['fom_rule'] = 'fastest tested CN-CG with worst error <= arm worst error'
    else:
        best = most_accurate; x['fom_rule'] = 'NO tested CN-CG is as accurate; unmatched-accuracy reference ratio vs the most accurate tested CN-CG (no bound implied)'
    x['fom_chosen'] = best['method']; x['fom_chosen_ms'] = best['device_ms_median']; x['fom_chosen_worst'] = best['worst_all_times']
    x['speedup'] = best['device_ms_median'] / x['device_ms_median']; x['no_fom_as_accurate'] = not ok

gates = dict(audit_passed=audit['passed'], audit_failure_count=audit['failure_count'], order_effect=audit['order_effect'],
             panel_order_effect=res['gates'].get('order_effect'), factor_model=res.get('factor_model'))
# recomputation agreement: every statistic above equals the audit's independent recomputation
agree = max(abs(x[k] - audit['recomputed'][x['method']][k2]) / max(abs(audit['recomputed'][x['method']][k2]), 1e-300)
            for x in rows for k, k2 in (('worst_all_times', 'worst_all_times'), ('median_case_max', 'median_case_max'), ('device_ms_median', 'device_ms_median')))
gates['summary_vs_audit_recompute_max_relative'] = agree
if ref04 is not None:
    ref = next(m for m in ref04['meshes'] if m['intervals'] == n and m['cohort'] == 'sealed_opened_once')
    rr = {r['method']: r['error_all_times_worst'] for r in ref['rows']}
    rep = {x['method']: dict(this=x['worst_all_times'], h2d_final04=rr[x['method']], relative=abs(x['worst_all_times'] / rr[x['method']] - 1))
           for x in rows if x['method'] in rr}
    gates['nmrom_reproduces_h2d_final04'] = dict(arms=rep, max_relative=max(v['relative'] for v in rep.values()), tolerance=1e-6,
                                                  passed=max(v['relative'] for v in rep.values()) <= 1e-6)
cfg = res['config']
expected = [a['name'] for a in cfg['nmrom']['arms']] + [f'linear_bank_{i}_BASELINE' for i in cfg['nmrom']['linear_bank_inits']] \
    + (['linear_bank_moments_cn_BASELINE'] if cfg['nmrom'].get('linear_bank_cn') else []) \
    + [f'pod{r}_{m}' for r in cfg['pod']['ranks'] for m in cfg['pod']['methods']] + [f'qm{r}_{a["name"]}' for r in cfg['qm']['ranks'] for a in cfg['qm']['arms']] \
    + ['dst_exact_CONTROL'] + [f'coarse{c}_{cfg["coarse_cg"]}_CONTROL' for c in cfg['coarse_intervals']] + list(cfg['operators']) + list(cfg['fom_order'])
missing = [a for a in expected if a not in res['arms']]
gates['expected_arms_missing'] = missing
gates['all_passed'] = bool(res['complete'] and audit['passed'] and not missing and gates['summary_vs_audit_recompute_max_relative'] < 1e-12
                           and gates.get('nmrom_reproduces_h2d_final04', {}).get('passed', False))
summary = dict(schema='heat-compare-summary-v1', status='final' if gates['all_passed'] else 'PROVISIONAL (a gate failed; diagnostic only)', mesh=n, unknowns=res['unknowns'], results_sha256=hashlib.sha256(raw).hexdigest(),
               audit_sha256=hashlib.sha256(apath.read_bytes()).hexdigest(), source_commit=res['source_commit'], metadata=res['metadata'],
               cohort=res['config'].get('cohort_name'), cohorts=res['config']['cohorts'], named_fom=named['method'], gates=gates, rows=rows,
               qm=res.get('qm') and {r: dict(gamma=v['gamma'], columns=v['columns'], holdout=v['selection']['grid']) for r, v in res['qm'].items()},
               pod_singular_values=res.get('pod', {}).get('singular_values'), operators_missing=res.get('operators_missing', []))
odir.mkdir(parents=True, exist_ok=True)
(odir / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
order = ['nmrom', 'linear_bank', 'pod', 'qm', 'operator', 'control', 'fom']
lines = [f'# {n}^2 ({res["unknowns"]} unknowns) — generated by summarize.py from results.json (sha256 {summary["results_sha256"][:16]}…)', '',
         f'**Status: {summary["status"]}.** GPU `{res["metadata"]["gpu"]}`, job {res["metadata"]["job_id"]}, source {res["source_commit"]}. Audit passed: {audit["passed"]}. '
         f'Order-effect gate: {audit["order_effect"]["passed"]} (max sentinel deviation {100*audit["order_effect"]["max_relative_deviation"]:.1f} %; '
         f'positive control fails as required: {audit["order_effect"]["positive_control_fails_as_required"]}).', '',
         '| method | family | unknowns | worst % | median % | GPU ms | FOM chosen | speedup | x vs named | failures |', '|---|---|---:|---:|---:|---:|---|---:|---:|---:|']
for fam in order:
    for x in sorted([x for x in rows if x['family'] == fam], key=lambda x: x['device_ms_median'] if fam == 'fom' else 0):
        sp = '—' if x['speedup'] is None else (f"{x['speedup']:.2f}" + ('†' if x.get('no_fom_as_accurate') else ''))
        lines.append(f"| {x['method']} | {x['family_label']} | {x['unknowns'] if x['unknowns'] is not None else '—'} | {100*x['worst_all_times']:.4f} | "
                     f"{100*x['median_case_max']:.4f} | {x['device_ms_median']:.3f} | {x['fom_chosen'] or '—'} | {sp} | {x['speedup_vs_named']:.2f} | {x['failures']} |")
lines += ['', '† no tested CN-CG setting is as accurate as this arm; the ratio is against the most accurate tested setting (unmatched accuracy; not a bound).']
(odir / 'TABLE.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines)); print(json.dumps({k: v for k, v in gates.items() if k != 'factor_model'}, indent=1))
