"""Independent NumPy audit of the Burgers correction ladder. No JAX, no GPU.

Recomputes every reported error from the retained output fields, measures each arm
against the converged same-mesh full-order solve, and checks q = 0 against the
head-ablation job's arm (a) on the same cases.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ABLATION = ROOT / 'experiments/head-ablation/artifacts/abl01/result.json'


def median(x):
    return float(np.median(np.asarray(x, dtype=float)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('result')
    p.add_argument('--out', required=True)
    p.add_argument('--fields', default=None)
    a = p.parse_args()
    r = json.loads(Path(a.result).read_text())
    checks, fail = {}, []

    def gate(name, ok, detail=None):
        checks[name] = dict(passed=bool(ok), detail=detail)
        if not ok:
            fail.append(name)

    gate('complete', r.get('complete') is True)
    gate('backend_gpu', r['backend'] == 'gpu', r['backend'])
    gate('x64', r['x64'] is True)
    gate('precision_highest', r['matmul_precision'] == 'highest')
    gate('bank_frozen', r['spatial_bank_frozen'] and r['network_weights_frozen'])
    gate('checkpoint_unchanged', r['checkpoint_sha256'] == r['checkpoint_sha256_after'])
    gate('final_cohort_unopened', r['final_cohort_unopened'] is True)
    gate('reference_residuals', all(x['max_relative_residual'] < 2e-11 for x in r['reference']),
         max(x['max_relative_residual'] for x in r['reference']))

    inv = r['invocations']
    reps = r['config']['reps']
    counts = {}
    for x in inv:
        counts.setdefault((x['name'], x['case']), 0)
        counts[(x['name'], x['case'])] += 1
    gate('every_subject_case_has_all_reps', set(counts.values()) == {reps}, sorted(set(counts.values())))
    hashes = {}
    for x in inv:
        hashes.setdefault((x['name'], x['case']), set()).add(x['field_sha256'])
    gate('repetition_output_identical', all(len(v) == 1 for v in hashes.values()),
         [k for k, v in hashes.items() if len(v) > 1][:5])
    gate('every_invocation_paired', all('error' in x and 'gpu_seconds' in x and x['finite'] for x in inv))

    roms = [x for x in inv if x['kind'] == 'rom']
    gate('overdetermined_weak_system', all(x['M'] > x['solved_dimension'] for x in roms),
         [(x['name'], x['M'], x['solved_dimension']) for x in roms
          if x['M'] <= x['solved_dimension']][:5])
    gate('test_ratio_four_times_solved_dimension',
         all(x['M'] == r['config']['test_multiplier'] * x['solved_dimension']
             for x in roms if x['name'] != 'q0_dense_Mmax'),
         sorted({x['name'] for x in roms
                 if x['name'] != 'q0_dense_Mmax'
                 and x['M'] != r['config']['test_multiplier'] * x['solved_dimension']}))
    gate('eq_quadrature_ratio',
         all(x['m'] == r['config']['quadrature_multiplier'] * x['M']
             for x in roms if x['quadrature'] == 'eq'))
    gate('directions_nested_and_available',
         r['directions']['available_rank'] >= max(r['config']['q_ladder']),
         r['directions']['available_rank'])

    if a.fields:
        d = Path(a.fields)
        bad = [x['artifact'] for x in inv if not (d / x['artifact']).exists()]
        gate('artifacts_present', not bad, bad[:5])
        if not bad:
            refs = {e['case']: np.load(d / e['artifact'])['fields'] for e in r['reference']}
            base = {x['case']: x['artifact'] for x in inv if x['name'] == 'fft_tight'}
            cache = {}

            def fields_of(name):
                if name not in cache:
                    cache[name] = np.load(d / name)['fields']
                return cache[name]

            worst = 0.
            for x in inv:
                truth = refs[x['case']]
                f = fields_of(x['artifact'])
                n0 = np.linalg.norm(truth[0])
                err = float(np.max(np.linalg.norm((f - truth).reshape(len(f), -1), axis=1)) / n0)
                worst = max(worst, abs(err - x['error']['fixed_initial_max'])
                            / max(x['error']['fixed_initial_max'], 1e-300))
                g = fields_of(base[x['case']])
                x['same_grid_max'] = float(
                    np.max(np.linalg.norm((f - g).reshape(len(f), -1), axis=1)) / n0)
            gate('recorded_errors_recomputed_from_saved_fields', worst < 1e-9, worst)

    # ---------------------------------------------------- per-arm aggregates
    table = {}
    for x in inv:
        t = table.setdefault(x['name'], dict(arm=x['name'], kind=x['kind'], q=x.get('q'),
                                             solved_dimension=x.get('solved_dimension'),
                                             M=x.get('M'), m=x.get('m'),
                                             quadrature=x.get('quadrature'), dt=x.get('dt'),
                                             gpu_ms=[], host_ms=[], case_error={}, case_same_grid={},
                                             iterations=[], stationarity=[], stationary=[],
                                             completed=[], budget_exits=[]))
        t['gpu_ms'].append(x['gpu_seconds'] * 1e3)
        t['host_ms'].append(x['host_seconds'] * 1e3)
        t['case_error'][x['case']] = x['error']['fixed_initial_max']
        if 'same_grid_max' in x:
            t['case_same_grid'][x['case']] = x['same_grid_max']
        t['iterations'] += x['iterations']
        if x['kind'] == 'rom':
            t['stationarity'].append(max(max(x['step_stationarity']), x['ic_stationarity']))
            t['stationary'].append(x['stationary'])
            t['completed'].append(x['completed'])
            t['budget_exits'].append(x['budget_exits'])
    recon = {x['q']: x for x in r['reconstruction']}
    rows = []
    for name, t in table.items():
        errs = np.array([t['case_error'][c] for c in sorted(t['case_error'])])
        sg = np.array([t['case_same_grid'][c] for c in sorted(t['case_same_grid'])]) \
            if t['case_same_grid'] else None
        rc = recon.get(t['q'])
        rows.append(dict(arm=name, kind=t['kind'], q=t['q'], solved_dimension=t['solved_dimension'],
                         M=t['M'], m=t['m'], quadrature=t['quadrature'], dt=t['dt'],
                         worst_reference_percent=float(np.max(errs) * 100),
                         median_reference_percent=float(np.median(errs) * 100),
                         worst_same_grid_percent=(float(np.max(sg) * 100) if sg is not None else None),
                         median_same_grid_percent=(float(np.median(sg) * 100) if sg is not None else None),
                         worst_bank_projection_percent=(float(rc['worst_bank_projection'] * 100) if rc else None),
                         worst_best_found_percent=(float(rc['worst_best_found'] * 100) if rc else None),
                         median_gpu_ms=median(t['gpu_ms']), median_host_ms=median(t['host_ms']),
                         gpu_ms_repetitions=len(t['gpu_ms']),
                         median_iterations=median(t['iterations']),
                         max_iterations=float(np.max(t['iterations'])),
                         max_stationarity=(float(np.max(t['stationarity'])) if t['stationarity'] else None),
                         all_stationary=(bool(all(t['stationary'])) if t['stationary'] else None),
                         all_completed=(bool(all(t['completed'])) if t['completed'] else None),
                         total_budget_exits=(int(np.sum(t['budget_exits'])) if t['budget_exits'] else None)))
    rows.sort(key=lambda x: (x['kind'] != 'rom', x['q'] if x['q'] is not None else 0, x['arm']))
    checks['arm_table'] = rows

    # --------------------------- q = 0 against the head-ablation job's arm (a)
    if ABLATION.exists():
        ab = json.loads(ABLATION.read_text())
        want = {x['case']: x['error']['fixed_initial_max'] for x in ab['invocations']
                if x['name'] == 'a_neural_eq' and x['intervals'] == r['intervals']}
        got = {x['case']: x['error']['fixed_initial_max'] for x in inv if x['name'] == 'q0_eq'}
        wh = {x['case']: x['field_sha256'] for x in ab['invocations']
              if x['name'] == 'a_neural_eq' and x['intervals'] == r['intervals']}
        gh = {x['case']: x['field_sha256'] for x in inv if x['name'] == 'q0_eq'}
        shared = sorted(set(want) & set(got))
        deltas = [abs(got[c] - want[c]) / max(want[c], 1e-300) for c in shared]
        identical = sum(1 for c in shared if wh[c] == gh[c])
        # Code equivalence is proved bitwise by the local smoke against the incumbent
        # solver. This is a cross-JOB reproduction on possibly different A100 models,
        # where XLA may select different kernels, so the tolerance is 1e-6 relative and
        # the bitwise count is reported rather than required.
        gate('q0_reproduces_head_ablation_arm_a', bool(shared) and max(deltas) < 1e-6,
             dict(compared=len(shared), worst_relative_delta=(max(deltas) if deltas else None),
                  bitwise_identical_fields=identical, tolerance=1e-6,
                  ablation_gpu=ab.get('gpu'), ladder_gpu=r.get('gpu'),
                  note=('ladder q0_eq versus head-ablation job arm a_neural_eq, same cases and mesh; '
                        'bitwise code equivalence is established separately by the local smoke')))
        gate('cases_identical',
             bool(np.allclose(np.asarray(ab['physical_cases']), np.asarray(r['physical_cases']))))

    out = dict(result=str(a.result), result_sha256=hashlib.sha256(Path(a.result).read_bytes()).hexdigest(),
               job_id=r.get('job_id'), commit=r.get('commit'), gpu=r.get('gpu'),
               failed=fail, passed=not fail, checks=checks)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'checks'}, indent=2))
    for name, v in checks.items():
        if isinstance(v, dict) and 'passed' in v:
            print(('PASS ' if v['passed'] else 'FAIL '), name, v['detail'])
    assert not fail, fail


if __name__ == '__main__':
    main()
