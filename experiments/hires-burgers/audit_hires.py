"""Independent NumPy audit of one hires-burgers attempt (no JAX import).

Recomputes, from the SAVED fields only:
  * the same-grid error of every (arm, case) on the restricted grid (saved for every arm),
  * the full-grid same-grid error for the audit case(s) and compares it with the job's number,
and builds the per-arm table (worst evolved / all-times error, median GPU and host ms over the
retained repetitions, exits, stalls), the speedups against every named FOM in the same
allocation, the 'fastest FOM with error <= the ROM's', the rule certificates and the parity
gates. Writes `<out>` (the small summary the report generator reads).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def med(x):
    return float(np.median(np.asarray(x, float)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive', help='runs/<attempt>/archive')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    arc = Path(a.archive)
    o = arc / 'output'
    r = json.loads((o / 'result.json').read_text())
    cfg = r['config']
    L = r['intervals']
    sub = max(1, L // cfg.get('restrict_to', 256))
    tight = cfg['same_grid_reference']
    gates = {}

    def gate(name, ok, **kw):
        gates[name] = dict(passed=bool(ok), **kw)

    log = ''.join(x.read_text(errors='replace') for x in sorted((arc / 'logs').glob('*.out')))
    gate('log_says_backend_gpu', 'jax_backend=gpu' in log)
    gate('complete', r.get('complete', False))
    gate('x64_and_highest', r['x64'] and r['matmul_precision'] == 'highest')
    gate('cohort_matches_b_panel', r['gates']['evaluation_cohort_matches_b_panel']['passed'] in (True, None))
    gate('phi_free_operator_parity', r['gates']['phi_free_operator_parity']['passed'])
    if 'repetition_output_identical' in r['gates']:
        gate('repetition_output_identical', r['gates']['repetition_output_identical']['passed'])

    # ---- restricted-grid recomputation for every quick row -------------------------------
    cases = sorted({x['case'] for x in r['quick']})
    base = {c: np.load(o / f'restricted_{tight}_case{c}.npz')['fields'] for c in cases}
    worst_gap, rows = 0., []
    for x in r['quick']:
        f = np.load(o / f"restricted_{x['name']}_case{x['case']}.npz")['fields']
        b = base[x['case']]
        n0 = np.linalg.norm(b[0])
        sg = np.linalg.norm((f - b).reshape(len(f), -1), axis=1) / n0
        rows.append(dict(name=x['name'], case=x['case'], restricted_evolved=float(sg[1:].max()),
                         restricted_all=float(sg.max()), job_full_evolved=x['same_grid_evolved'],
                         job_full_all=x['same_grid_all']))
        # restricted and full-grid norms are different samplings of the same smooth error field
        if x['same_grid_evolved'] > 1e-6:
            worst_gap = max(worst_gap, abs(sg[1:].max() / x['same_grid_evolved'] - 1))
    gate('restricted_recomputation_tracks_full_grid', worst_gap < 0.05, worst_relative_gap=worst_gap,
         note='restricted-grid error recomputed in NumPy vs the full-grid number of the job, relative gap')

    # ---- full-grid recomputation on the audit case(s) ------------------------------------
    full = []
    for c in cfg['audit_cases']:
        tp = o / f'full_{tight}_case{c}.npy'
        if not tp.exists():
            continue
        t = np.load(tp, mmap_mode='r')
        n0 = float(np.linalg.norm(t[0]))
        for fp in sorted(o.glob(f'full_*_case{c}.npy')):
            name = fp.name[len('full_'):-len(f'_case{c}.npy')]
            f = np.load(fp, mmap_mode='r')
            sg = [float(np.linalg.norm(np.asarray(f[k]) - np.asarray(t[k]))) / n0 for k in range(len(f))]
            job = next(x for x in r['quick'] if x['name'] == name and x['case'] == c)
            full.append(dict(name=name, case=c, recomputed_evolved=max(sg[1:]), job_evolved=job['same_grid_evolved'],
                             abs_diff=abs(max(sg[1:]) - job['same_grid_evolved']),
                             sha256=hashlib.sha256(fp.read_bytes()).hexdigest()))
    gate('full_grid_errors_recomputed', bool(full) and all(x['abs_diff'] <= 1e-12 for x in full),
         worst_abs_diff=max([x['abs_diff'] for x in full] or [None]), arms=len(full))

    # ---- per-arm table ---------------------------------------------------------------------
    inv = r['invocations']
    names = list(dict.fromkeys(x['name'] for x in inv))
    table = {}
    for n in names:
        xs = [x for x in inv if x['name'] == n]
        per_case = {}
        for x in xs:
            per_case.setdefault(x['case'], x)
        t = dict(name=n, family=xs[0]['family'], reps=len(xs) // max(len(per_case), 1), cases=len(per_case),
                 median_gpu_ms=1e3 * med([x['gpu_seconds'] for x in xs]),
                 median_host_ms=1e3 * med([x['host_seconds'] for x in xs]),
                 per_case_median_gpu_ms={str(c): 1e3 * med([x['gpu_seconds'] for x in xs if x['case'] == c]) for c in per_case},
                 worst_evolved_percent=100 * max(x['same_grid_evolved'] for x in per_case.values()),
                 worst_all_times_percent=100 * max(x['same_grid_all'] for x in per_case.values()),
                 median_evolved_percent=100 * med([x['same_grid_evolved'] for x in per_case.values()]),
                 t0_compression_percent=100 * max(x['t0_compression'] for x in per_case.values()))
        if xs[0]['family'] == 'rom':
            t.update(q=xs[0]['q'], M=xs[0]['M'], m=xs[0]['m'], rule=xs[0]['rule'], gtol=xs[0]['gtol'],
                     median_iterations_per_step=med([x['median_iterations'] for x in per_case.values()]),
                     total_iterations_median=med([x['total_iterations'] for x in per_case.values()]),
                     stalled_exits=int(sum(x['stalled_exits'] for x in per_case.values())),
                     budget_exits=int(sum(x['budget_exits'] for x in per_case.values())),
                     tiny_step_exits=int(sum(x['tiny_step_exits'] for x in per_case.values())),
                     rejected_exits=int(sum(x['rejected_exits'] for x in per_case.values())),
                     steps=int(sum(len(x['stop_reasons']) for x in per_case.values())),
                     damping_retries=int(sum(x.get('damping_retries_total', 0) for x in per_case.values())),
                     worst_joint_stationarity=max([x['worst_joint_stationarity'] for x in per_case.values()
                                                   if x['worst_joint_stationarity'] is not None], default=None))
            ru = next((y for y in r['rules'] if y['q'] == t['q'] and y['M'] == t['M'] and y['rule'] == t['rule']), None)
            if ru:
                dep = (ru.get('deployed') or {}).get(n)
                t.update(rho_max_heldout=ru['heldout_population']['rho_max'],
                         rho_max_deployed=(dep or {}).get('rho_max'), control=ru.get('control', False),
                         certified_primary=(bool(ru['heldout_population']['certified_primary'] and dep['certified_primary'])
                                            if dep else None),
                         certificate_basis=('held-out dense-query population AND this arm\'s own visited states' if dep
                                            else 'no deployed-state certificate for this arm (audited base twin: inherits nothing)'))
        else:
            t.update(mesh=xs[0]['mesh'], dt=xs[0]['dt'], ntol=xs[0]['ntol'], ltol=xs[0]['ltol'], impl=xs[0].get('impl'),
                     stalled_steps=int(sum(x['stalled_steps'] for x in per_case.values())),
                     steps=int(sum(x['steps'] for x in per_case.values())),
                     nonlinear_converged=all(x['nonlinear_converged'] for x in per_case.values()),
                     newton_iterations_median=med([x['newton_iterations_total'] for x in per_case.values()]))
        table[n] = t
    phys = {}
    for x in r.get('physical', []):
        phys.setdefault(x['name'], []).append(x['reference_evolved'])
    for n, v in phys.items():
        if n in table:
            table[n]['worst_reference_evolved_percent'] = 100 * max(v)
            table[n]['reference_cases'] = len(v)
    truth_phys = 100 * max(phys['__truth__']) if '__truth__' in phys else None

    # ---- speedups, same allocation only ------------------------------------------------------
    foms = {n: t for n, t in table.items() if t['family'] == 'fom' and t.get('mesh') == L}
    for n, t in table.items():
        if t['family'] != 'rom':
            continue
        t['speedup_gpu'] = {f: foms[f]['median_gpu_ms'] / t['median_gpu_ms'] for f in foms}
        t['speedup_host'] = {f: foms[f]['median_host_ms'] / t['median_host_ms'] for f in foms}
        ok = [f for f in foms if foms[f]['worst_evolved_percent'] <= t['worst_evolved_percent'] and foms[f]['nonlinear_converged']]
        best = min(ok, key=lambda f: foms[f]['median_gpu_ms']) if ok else None
        t['fastest_fom_at_least_as_accurate'] = best
        t['speedup_vs_fastest_fom_at_least_as_accurate'] = (foms[best]['median_gpu_ms'] / t['median_gpu_ms']) if best else None
        t['meets_1_percent'] = bool(t['worst_evolved_percent'] <= 1.0)
        t['meets_half_percent'] = bool(t['worst_evolved_percent'] <= 0.5)

    # ---- verdict (DESIGN section 7: cheapest certified arm with evolved error <= 1 %) ----------
    for k in ('control_rule_fails_certificate', 'five_retained_repetitions_everywhere'):
        if k in r['gates']:
            gate(k, r['gates'][k]['passed'] is not False, detail=r['gates'][k])
    par = [x for x in r['parity'] if x.get('covered')]
    gate('every_covered_fast_arm_passes_parity', all(x['passed'] for x in par), covered=len(par),
         uncovered=[x['fast'] for x in r['parity'] if not x.get('covered')],
         failed=[x['fast'] for x in par if not x['passed']])
    tight_names = [n for n, t in foms.items() if t['ntol'] == foms[tight]['ntol'] and t['ltol'] == foms[tight]['ltol'] and t['dt'] == foms[tight]['dt']]
    best_tight = min(tight_names, key=lambda n: foms[n]['median_gpu_ms'])
    passing = [n for n, t in foms.items() if t['nonlinear_converged'] and t['worst_evolved_percent'] <= cfg.get('fom_passing_percent', 0.1)]
    best_passing = min(passing, key=lambda n: foms[n]['median_gpu_ms']) if passing else None
    verdict = {}
    for label, limit in (('accurate_1_percent', 1.0), ('stretch_half_percent', 0.5)):
        elig = [t for t in table.values() if t['family'] == 'rom' and t.get('certified_primary') and not t.get('control')
                and t['worst_evolved_percent'] <= limit]
        if not elig:
            verdict[label] = dict(met=False, reason='no certified ROM arm at or below the error limit', limit_percent=limit)
            continue
        t = min(elig, key=lambda x: x['median_gpu_ms'])
        sp = dict(vs_tight=foms[best_tight]['median_gpu_ms'] / t['median_gpu_ms'],
                  vs_relaxed_passing=(foms[best_passing]['median_gpu_ms'] / t['median_gpu_ms']) if best_passing else None,
                  vs_fastest_tested_fom_at_least_as_accurate=t['speedup_vs_fastest_fom_at_least_as_accurate'])
        verdict[label] = dict(arm=t['name'], worst_evolved_percent=t['worst_evolved_percent'],
                              worst_all_times_percent=t['worst_all_times_percent'], median_gpu_ms=t['median_gpu_ms'],
                              stalled_exits=t['stalled_exits'], steps=t['steps'], tight_comparator=best_tight,
                              relaxed_passing_comparator=best_passing,
                              fastest_tested_fom_at_least_as_accurate=t['fastest_fom_at_least_as_accurate'], speedups=sp,
                              met_vs_tight=bool(sp['vs_tight'] >= 5), met_vs_relaxed_passing=bool((sp['vs_relaxed_passing'] or 0) >= 5),
                              met_vs_fastest_tested_at_least_as_accurate=bool((sp['vs_fastest_tested_fom_at_least_as_accurate'] or 0) >= 5),
                              limit_percent=limit, selection='cheapest certified non-control ROM arm with worst evolved error <= limit (development cases)')
    fastq0 = [t for t in table.values() if t['family'] == 'rom' and t['q'] == 0]
    if fastq0:
        t = min(fastq0, key=lambda x: x['median_gpu_ms'])
        verdict['fast_setting'] = dict(arm=t['name'], worst_evolved_percent=t['worst_evolved_percent'], median_gpu_ms=t['median_gpu_ms'],
                                       vs_tight=foms[best_tight]['median_gpu_ms'] / t['median_gpu_ms'],
                                       vs_relaxed_passing=(foms[best_passing]['median_gpu_ms'] / t['median_gpu_ms']) if best_passing else None,
                                       vs_fastest_tested_fom_at_least_as_accurate=t['speedup_vs_fastest_fom_at_least_as_accurate'],
                                       certified_primary=t.get('certified_primary'))

    summary = dict(cohort=r.get('cohort_name'), cohort_cases=len(r['physical_cases']), verdict=verdict, attempt=cfg['attempt'], job_id=r['job_id'], commit=r['commit'], gpu=r['gpu'], nvidia_smi=r['nvidia_smi'],
                   intervals=L, elapsed_seconds=r.get('elapsed_seconds'), gates=gates, table=table,
                   truth_vs_refined_reference_evolved_percent=truth_phys, rules=r['rules'], parity=r['parity'],
                   profile=r['profile'], dense_truth=r['dense_truth'], dropped=r['dropped'],
                   populations=r['populations'], full_grid_recheck=full, restricted_recheck=rows,
                   result_sha256=hashlib.sha256((o / 'result.json').read_bytes()).hexdigest(),
                   failed_gates=[k for k, v in gates.items() if not v['passed']])
    Path(a.out).write_text(json.dumps(summary, indent=1) + '\n')
    print('failed gates:', summary['failed_gates'])
    for n, t in table.items():
        print(f"{n:42s} ev {t['worst_evolved_percent']:8.4f}%  all {t['worst_all_times_percent']:8.4f}%  gpu {t['median_gpu_ms']:9.2f} ms"
              + (f"  S_tight {t['speedup_gpu'].get(tight, float('nan')):6.2f}  cert {t.get('certified_primary')}  stalled {t['stalled_exits']}"
                 if t['family'] == 'rom' else f"  stalled_steps {t['stalled_steps']}"))


if __name__ == '__main__':
    main()
