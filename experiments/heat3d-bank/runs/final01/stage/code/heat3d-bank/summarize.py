"""summary.json + SUMMARY.md for one run directory, generated only from results.json / audit.json."""
import hashlib, json, sys
from pathlib import Path
import numpy as np

run = Path(sys.argv[1]); out = Path(sys.argv[2]) if len(sys.argv) > 2 else run / 'pull' / 'out'; raw = (out / 'results.json').read_bytes(); res = json.loads(raw)
audit = json.loads((run / 'audit.json').read_text()) if (run / 'audit.json').exists() else None
summary = dict(schema='heat3d-bank-summary-v1', results_sha256=hashlib.sha256(raw).hexdigest(), source_commit=res['source_commit'], source_sha256=res['source_sha256'],
               model_sha256=res['model_sha256'], reporting_sha256={f: hashlib.sha256((Path(__file__).resolve().parent / f).read_bytes()).hexdigest() for f in ('audit.py', 'summarize.py')}, metadata=res['metadata'], audit_passed=None if audit is None else audit['passed'], meshes=[])
lines = []
names = res['config'].get('cohort_names') or [['all', 10**9]]; cohorts = []; lo = 0
for cname, cnt in names:
    cohorts.append((cname, lo, lo + cnt)); lo += cnt
for mesh, (cname, lo, hi) in [(m, c) for m in res['meshes'] for c in cohorts]:
    rows = []
    for r in mesh['rows']:
        same, phys = np.asarray(r['same'])[lo:hi], np.asarray(r['physical'])[lo:hi]; ms = np.asarray(r['device_ms']).reshape(-1, res['config']['repetitions'])[lo:hi]; r = dict(r, stats=r['stats'][lo:hi])   # timed cases are a prefix (heat3d-bank; stats slice restored per Codex finding 4)
        row = dict(method=r['method'], cases=len(same), timed_cases=len(ms), repetitions=ms.shape[1], error_all_times_worst=float(same.max()), error_evolved_worst=float(same[:, 1:].max()),
                   error_all_times_median=float(np.median(same.max(1))), physical_all_times_worst=float(phys.max()), physical_evolved_worst=float(phys[:, 1:].max()),
                   device_ms_median=float(np.median(ms)) if ms.size else float('nan'), device_ms_median_of_case_medians=float(np.median(np.median(ms, axis=1))) if ms.size else float('nan'),
                   device_ms_p10_p90=[float(np.percentile(ms, 10)), float(np.percentile(ms, 90))] if ms.size else None, failures=0)
        if r['method'].startswith('nmrom'):
            i = np.stack([np.asarray(x[0]).reshape(-1, 5)[-1] for x in r['stats']]); st = np.concatenate([np.asarray(x[1]).reshape(-1, 5) for x in r['stats']])
            row.update(init_attempts_max=float(i[:, 0].max()), init_attempts_mean=float(i[:, 0].mean()), step_attempts_mean=float(st[:, 0].mean()), step_attempts_max=float(st[:, 0].max()),
                       failures=int((st[:, 2] != 1).sum() + (i[:, 2] != 1).sum()))
        elif r['stats'] and r['stats'][0]:
            cg = np.concatenate([np.asarray(x[0]).reshape(-1, 3) for x in r['stats']]); row.update(cg_iterations_per_query=float(cg[:, 0].sum() / len(same)), failures=int((cg[:, 2] != 1).sum()))
        rows.append(row)
    named = next((r for r in rows if r['method'].endswith('_NAMED')), None) or dict(device_ms_median=float('nan')); foms = [r for r in rows if r['method'].startswith('fom_') and not r['failures']]
    coarse = [r for r in rows if r['method'].startswith('coarse') and not r['failures']]
    for r in rows:
        r['speedup_vs_named_fom'] = named['device_ms_median'] / r['device_ms_median']
        if r['method'].startswith('nmrom') or 'BASELINE' in r['method']:
            ok = [f for f in foms if f['error_all_times_worst'] <= r['error_all_times_worst']]; ok_e = [f for f in foms if f['error_evolved_worst'] <= r['error_evolved_worst']]
            okc = [f for f in coarse if f['physical_evolved_worst'] <= r['physical_evolved_worst']]
            for key, pool in (('fastest_fom_error_le_rom_all_times', ok), ('fastest_fom_error_le_rom_evolved', ok_e), ('fastest_coarse_fom_physical_evolved_le_rom', okc)):
                if pool:
                    best = min(pool, key=lambda f: f['device_ms_median']); r[key] = dict(method=best['method'], device_ms=best['device_ms_median'], speedup=best['device_ms_median'] / r['device_ms_median'])
                else: r[key] = None
            r['bar_error_le_1pct'] = r['error_all_times_worst'] <= .01; r['bar_speedup_ge_5'] = r['speedup_vs_named_fom'] >= 5
    summary['meshes'].append(dict(cohort=cname, intervals=mesh['intervals'], unknowns=mesh['unknowns'], bank_condition=mesh['bank_condition'], profile_ms=mesh['profile'], rows=rows,
                                  reference_refinement_over_budget=sum(not c.get('reference_refinement_ok', True) for c in mesh['cases'][lo:hi]), reference_refinement_max=max((c.get('reference_refinement', 0.) for c in mesh['cases'][lo:hi]), default=None)))
    lines += [f"\n### [{cname}] {mesh['intervals']} intervals per axis ({mesh['unknowns']} unknowns), {rows[0]['cases']} cases ({rows[0]['timed_cases']} timed x {rows[0]['repetitions']} repetitions)\n",
              '| method | err all-times worst % | err evolved worst % | GPU ms | x vs named CN-CG | fastest CG with err<=ROM (all-times): x | coarse FOM matched: x | failures |', '|---|---:|---:|---:|---:|---|---|---:|']
    for r in rows:
        f = r.get('fastest_fom_error_le_rom_all_times'); c = r.get('fastest_coarse_fom_physical_evolved_le_rom')
        lines.append(f"| {r['method']} | {100*r['error_all_times_worst']:.4f} | {100*r['error_evolved_worst']:.4f} | {r['device_ms_median']:.3f} | {r['speedup_vs_named_fom']:.3f} | "
                     + (f"{f['method']}: {f['speedup']:.3f}" if f else '-') + ' | ' + (f"{c['method']}: {c['speedup']:.3f}" if c else '-') + f" | {r['failures']} |")
(run / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
(run / 'SUMMARY.md').write_text(f"# {run.name} — generated summary\n\nGenerated by `summarize.py` from `results.json` (sha256 {summary['results_sha256'][:16]}…), source {res['source_commit']}, GPU `{res['metadata']['gpu']}`, job {res['metadata']['job_id']}. Audit passed: {summary['audit_passed']}. Errors are same-grid current-relative L2, worst over cases; times are median GPU ms over all retained repetitions. `x` = comparator time / method time in this allocation.\n" + '\n'.join(lines) + '\n')
print('\n'.join(lines))
