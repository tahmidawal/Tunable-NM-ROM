"""Independent audit of an evaluation attempt. NumPy only: no JAX, no driver.

Every reported error is recomputed from the retained output fields, every
same-grid discrepancy is recomputed against the retained converged full-order
fields, the invocation grid is required to be complete, the timing identity is
checked, and the per-arm table the report prints is built here rather than by
the driver that produced the numbers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def median(x):
    return float(np.median(np.asarray(x, dtype=float)))


def params_draw(seed, count):
    """The family, re-implemented here so the audit does not import the driver."""
    r = np.random.default_rng(seed)
    return np.stack([r.uniform(.15, .85, count), r.uniform(.15, .85, count),
                     r.uniform(.05, .20, count), r.uniform(.5, 2., count),
                     np.exp(r.uniform(np.log(.01), np.log(.1), count))], axis=1)


def sha_array(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('result')
    p.add_argument('--fields', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--train', default=None, help='the training attempt result.json')
    a = p.parse_args()
    r = json.loads(Path(a.result).read_text())
    F = Path(a.fields)
    cfg = r['config']
    checks, notes = {}, []

    def gate(name, ok, detail=None):
        checks[name] = dict(passed=bool(ok), detail=detail)

    gate('complete', r.get('complete') is True)
    gate('backend_gpu', r.get('backend') == 'gpu', r.get('backend'))
    gate('x64', r.get('x64') is True)
    gate('precision_highest', r.get('matmul_precision') == 'highest')
    gate('bank_and_weights_frozen', r.get('spatial_bank_frozen') and r.get('network_weights_frozen'))
    gate('final_cohort_unopened', r.get('final_cohort_unopened') is True)
    gate('checkpoints_unchanged',
         all(r['checkpoint_sha256_after'][c['arm']] == c['sha256'] for c in r['checkpoints']))

    # the cohort is the one abl01 used
    phys = np.asarray(r['physical_cases'])
    want = np.concatenate((params_draw(cfg['eval_seed'], cfg['eval_cases']),
                           params_draw(cfg['eval_fresh_seed'], cfg['eval_fresh_cases'])))
    gate('cohort_reproduced', np.allclose(phys, want), float(np.max(np.abs(phys - want))))

    # references
    refs = {}
    for row in r['reference']:
        d = np.load(F / row['artifact'])
        f = d['fields']
        assert sha_array(f) == row['field_sha256'], row['artifact']
        refs[row['case']] = f
    gate('reference_hashes', True, len(refs))
    base = {}
    for c in refs:
        b = np.load(F / f'samegrid_case{c}.npz')['fields']
        assert sha_array(b) == r['same_grid_reference']['field_sha256'][str(c)]
        base[c] = b
    gate('same_grid_reference_hashes', True, len(base))

    # every invocation recomputed from its saved field
    worst_err = worst_sg = 0.
    cache = {}
    for x in r['invocations']:
        if x['artifact'] not in cache:
            cache[x['artifact']] = np.load(F / x['artifact'])['fields']
        f = cache[x['artifact']]
        assert sha_array(f) == x['field_sha256'], x['artifact']
        t = refs[x['case']]
        n0 = float(np.linalg.norm(t[0]))
        e = np.linalg.norm((f - t).reshape(len(t), -1), axis=1)
        worst_err = max(worst_err, abs(float(np.max(e / n0)) - x['error']['fixed_initial_max'])
                        / max(x['error']['fixed_initial_max'], 1e-300))
        g = np.linalg.norm((f - base[x['case']]).reshape(len(f), -1), axis=1) / n0
        for key, val in (('max', float(np.max(g))), ('evolved_max', float(np.max(g[1:]))),
                         ('t0', float(g[0]))):
            ref = x['same_grid'][key]
            worst_sg = max(worst_sg, abs(val - ref) / max(ref, 1e-300))
        assert x['host_seconds'] >= x['gpu_seconds'] > 0
    gate('recorded_errors_recomputed_from_saved_fields', worst_err < 1e-9, worst_err)
    gate('same_grid_recomputed_from_saved_fields', worst_sg < 1e-9, worst_sg)

    # complete invocation grid
    names = sorted({x['name'] for x in r['invocations']})
    need = len(refs) * cfg['reps']
    missing = [n for n in names if sum(1 for x in r['invocations'] if x['name'] == n) != need]
    gate('every_subject_case_has_all_reps', not missing, missing)
    inconsistent = []
    for n in names:
        h = {(x['case'],): {y['field_sha256'] for y in r['invocations']
                            if y['name'] == n and y['case'] == x['case']}
             for x in r['invocations'] if x['name'] == n}
        for k, v in h.items():
            if len(v) != 1:
                inconsistent.append((n, k))
    gate('repetition_output_identical', not inconsistent, inconsistent)

    # in-job fidelity gate
    g = r.get('gates', {}).get('incumbent_reproduces_abl01')
    gate('incumbent_reproduces_abl01', bool(g and g['passed']), g)

    # ---------------------------------------------------------- arm table ----
    recon = {x['checkpoint']: x for x in r['reconstruction']}
    table = {}
    for x in r['invocations']:
        t = table.setdefault(x['name'], dict(
            arm=x['name'], kind=x['kind'], checkpoint=x.get('checkpoint'), K=x.get('K'),
            R=x.get('R'), M=x.get('M'), m=x.get('m'), quadrature=x.get('quadrature'),
            gpu_ms=[], host_ms=[], ref={}, sg={}, sge={}, sg0={}, iters=[], stat=[],
            stationary=[], completed=[], budget=[]))
        t['gpu_ms'].append(x['gpu_seconds'] * 1e3)
        t['host_ms'].append(x['host_seconds'] * 1e3)
        t['ref'][x['case']] = x['error']['fixed_initial_max']
        t['sg'][x['case']] = x['same_grid']['max']
        t['sge'][x['case']] = x['same_grid']['evolved_max']
        t['sg0'][x['case']] = x['same_grid']['t0']
        t['iters'] += x['iterations']
        if x['kind'] == 'rom':
            t['stat'].append(max(max(x['step_stationarity']), x['ic_stationarity']))
            t['stationary'].append(x['stationary'])
            t['completed'].append(x['completed'])
            t['budget'].append(x['budget_exits'])
    rows = []
    for name, t in table.items():
        rc = recon.get(t['checkpoint'])
        vals = lambda d: np.array([d[c] for c in sorted(d)])
        rows.append(dict(
            arm=name, kind=t['kind'], checkpoint=t['checkpoint'], K=t['K'], R=t['R'],
            M=t['M'], m=t['m'], quadrature=t['quadrature'],
            worst_reference_percent=float(np.max(vals(t['ref'])) * 100),
            median_reference_percent=float(np.median(vals(t['ref'])) * 100),
            worst_same_grid_percent=float(np.max(vals(t['sg'])) * 100),
            median_same_grid_percent=float(np.median(vals(t['sg'])) * 100),
            worst_same_grid_evolved_percent=float(np.max(vals(t['sge'])) * 100),
            worst_t0_percent=float(np.max(vals(t['sg0'])) * 100),
            worst_bank_projection_percent=(float(rc['worst_bank_projection'] * 100) if rc else None),
            worst_best_found_percent=(float(rc['worst_best_found'] * 100) if rc else None),
            worst_best_found_t0_percent=(float(rc['worst_best_found_t0'] * 100) if rc else None),
            worst_best_found_evolved_percent=(float(rc['worst_best_found_evolved'] * 100)
                                              if rc else None),
            median_gpu_ms=median(t['gpu_ms']), median_host_ms=median(t['host_ms']),
            repetitions=len(t['gpu_ms']), median_iterations=median(t['iters']),
            max_iterations=float(np.max(t['iters'])),
            max_stationarity=(float(np.max(t['stat'])) if t['stat'] else None),
            all_stationary=(bool(np.all(t['stationary'])) if t['stationary'] else None),
            all_completed=(bool(np.all(t['completed'])) if t['completed'] else None),
            budget_exits=(int(np.sum(t['budget'])) if t['budget'] else None)))
    rows.sort(key=lambda x: (x['kind'] != 'rom', x['arm']))

    # ------------------------------------------- pre-registered verdicts -----
    inc = next((x for x in rows if x['arm'] == cfg['incumbent_arm']), None)
    crit = cfg.get('success', dict(best_found=1.0, same_grid=1.3, cost_factor=1.5))
    verdicts = []
    for x in rows:
        if x['kind'] != 'rom' or x['worst_best_found_percent'] is None:
            continue
        cost = x['median_gpu_ms'] / inc['median_gpu_ms'] if inc else None
        ok = (x['worst_best_found_percent'] < crit['best_found']
              and x['worst_same_grid_percent'] < crit['same_grid']
              and bool(x['all_completed']) and bool(x['all_stationary'])
              and cost is not None and cost <= crit['cost_factor'])
        verdicts.append(dict(arm=x['arm'], checkpoint=x['checkpoint'],
                             best_found_percent=x['worst_best_found_percent'],
                             same_grid_percent=x['worst_same_grid_percent'],
                             stationary=x['all_stationary'], completed=x['all_completed'],
                             cost_factor=cost, success=bool(ok)))
    passed = [v for v in verdicts if v['success']]
    gate('success_criteria_evaluated', True,
         dict(criteria=crit, n_arms=len(verdicts), n_passing=len(passed),
              passing=[v['arm'] for v in passed]))

    train = json.loads(Path(a.train).read_text()) if a.train else None
    if train:
        declared = {x['arm']: x['sha256'] for x in train['emitted']}
        found = {c['arm']: c['sha256'] for c in r['checkpoints'] if c['arm'] != 'incumbent'}
        gate('evaluated_checkpoints_are_the_trained_ones',
             all(declared.get(k) == v for k, v in found.items()),
             {k: (declared.get(k), v) for k, v in found.items() if declared.get(k) != v})

    out = dict(result=str(a.result), fields=str(F), checks=checks,
               arm_table=rows, verdicts=verdicts,
               all_passed=bool(all(v['passed'] for v in checks.values())), notes=notes)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=float) + '\n')
    for k, v in checks.items():
        print(f"{'PASS' if v['passed'] else 'FAIL'}  {k}  {v['detail']}")
    print('AUDIT', 'OK' if out['all_passed'] else 'FAILED', a.out)


if __name__ == '__main__':
    main()
