"""Independent NumPy audit of the Poisson head-refinement job. No JAX, no GPU."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_common import head_np, median

HERE = Path(__file__).resolve().parent
GATE = HERE / 'gate-reference-poisson.json'


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
    gate('spatial_bank_frozen', r['spatial_bank_frozen'] is True)
    gate('final_cohort_unopened', r['final_cohort_unopened'] is True)
    gate('calibration_case_not_evaluation',
         r['calibration_case']['sha256'] not in
         {hashlib.sha256(np.ascontiguousarray(np.asarray(c)).tobytes()).hexdigest()
          for c in r['cohort']['parameters']})

    inv = r['invocations']
    reps = r['config']['repetitions']
    counts, hashes = {}, {}
    for x in inv:
        counts[(x['name'], x['case'])] = counts.get((x['name'], x['case']), 0) + 1
        hashes.setdefault((x['name'], x['case']), set()).add(x['field_sha256'])
    gate('every_subject_case_has_all_reps', set(counts.values()) == {reps}, sorted(set(counts.values())))
    gate('repetition_output_identical', all(len(v) == 1 for v in hashes.values()),
         [k for k, v in hashes.items() if len(v) > 1][:5])
    gate('every_invocation_paired',
         all('physical_error' in x and x['total_seconds'] > 0 and x['finite'] for x in inv))
    roms = [x for x in inv if x['kind'] == 'rom']
    gate('overdetermined_weak_system', all(x['M'] > x['k'] for x in roms))
    gate('one_step_size_everywhere', len({x['alpha'] for x in roms}) == 1,
         sorted({x['alpha'] for x in roms}))
    gate('declared_anchor_weights_only',
         {x['mu'] for x in roms if x['n'] > 0} == set(r['config']['mu'].values()))
    gate('n_zero_has_no_drift', all(x['drift'] == 0.0 for x in roms if x['n'] == 0))
    gate('refined_arms_moved', all(x['drift'] > 0.0 for x in roms if x['n'] > 0))

    redecode = None
    if a.fields:
        d = Path(a.fields)
        bad = [x['artifact'] for x in inv if not (d / x['artifact']).exists()]
        gate('artifacts_present', not bad, bad[:5])
        if not bad:
            cache = {}

            def field_of(name):
                if name not in cache:
                    cache[name] = np.load(d / name)['field']
                return cache[name]

            refs = {e['case']: np.load(d / e['artifact'])['field'] for e in r['references']}
            worst = 0.
            for x in inv:
                f = field_of(x['artifact'])
                t = refs[x['case']]
                err = float(np.linalg.norm(f - t) / np.linalg.norm(t))
                worst = max(worst, abs(err - x['physical_error'])
                            / max(x['physical_error'], 1e-300))
            gate('recorded_errors_recomputed_from_saved_fields', worst < 1e-9, worst)

            base = {x['case']: x['artifact'] for x in inv if x['name'] == 'dst_direct'}
            for x in inv:
                f, g = field_of(x['artifact']), field_of(base[x['case']])
                # the direct transform solve IS the exact discrete solution on this mesh,
                # so this isolates reduction error from discretisation error
                x['same_grid_error'] = float(np.linalg.norm(f - g) / max(np.linalg.norm(g), 1e-300))

            rd = np.load(d / r['redecode']['artifact'])
            nodes, Bsub, theta0 = rd['nodes'], rd['bank'], rd['theta0']
            shapes = r['theta_layout']['shapes']
            gate('theta_layout_total',
                 int(np.sum([int(np.prod(s)) for s in shapes])) == len(theta0))
            n = r['intervals']
            seen, worst_rd = 0, 0.
            for entry in r['reconstruction']:
                art = entry.get('weights_artifact')
                W = np.load(d / art) if art else None
                for row in entry['cases']:
                    case = row['case']
                    x = next(y for y in inv if y['name'] == entry['arm'] and y['case'] == case)
                    theta = np.asarray(W[f'case{case}']) if W is not None else theta0
                    got = Bsub @ head_np(theta, np.asarray(x['latent']), shapes)
                    want = field_of(x['artifact'])[1:-1, 1:-1].ravel()[nodes]
                    worst_rd = max(worst_rd, float(np.linalg.norm(got - want)
                                                   / max(np.linalg.norm(want), 1e-300)))
                    seen += 1
            redecode = dict(compared=seen, worst_relative=worst_rd, tolerance=1e-10,
                            nodes=int(len(nodes)))
            gate('refined_weights_redecode', worst_rd < 1e-10, redecode)

    table = {}
    for x in inv:
        t = table.setdefault(x['name'], dict(arm=x['name'], kind=x['kind'], variant=x.get('variant'),
                                             n=x.get('n'), mu=x.get('mu'), mu_key=x.get('mu_key'),
                                             k=x.get('k'), M=x.get('M'), ms=[], device_ms=[],
                                             case_error={}, case_same_grid={}, iterations=[],
                                             refine_iterations=[], stationarity=[], drift=[],
                                             completed=[], stationary=[], budget_exits=[],
                                             rejected_exits=[], reasons=[]))
        t['ms'].append(x['total_seconds'] * 1e3)
        if 'fused_device_seconds' in x:
            t['device_ms'].append(x['fused_device_seconds'] * 1e3)
        t['case_error'][x['case']] = x['physical_error']
        if 'same_grid_error' in x:
            t['case_same_grid'][x['case']] = x['same_grid_error']
        if x['kind'] == 'rom':
            t['iterations'].append(x['first_iterations'])
            t['refine_iterations'] += x['refine_iterations']
            t['stationarity'] += [x['stationarity'], x['first_stationarity']] + x['refine_stationarity']
            t['drift'].append(x['drift'])
            t['completed'].append(x['completed'])
            t['stationary'].append(x['stationary'])
            t['budget_exits'].append(x['budget_exits'])
            t['rejected_exits'].append(x['rejected_exits'])
            t['reasons'].append(x['reason'])
    recon = {e['arm']: e for e in r['reconstruction']}
    rows = []
    for name, t in table.items():
        errs = np.array([t['case_error'][c] for c in sorted(t['case_error'])])
        sg = (np.array([t['case_same_grid'][c] for c in sorted(t['case_same_grid'])])
              if t['case_same_grid'] else None)
        rc = recon.get(name)
        rows.append(dict(
            arm=name, kind=t['kind'], variant=t['variant'], n=t['n'], mu=t['mu'],
            mu_key=t['mu_key'], k=t['k'], M=t['M'],
            worst_error_percent=float(np.max(errs) * 100),
            median_error_percent=float(np.median(errs) * 100),
            worst_same_grid_percent=(float(np.max(sg) * 100) if sg is not None else None),
            median_same_grid_percent=(float(np.median(sg) * 100) if sg is not None else None),
            worst_bank_projection_percent=(float(rc['worst_bank_projection'] * 100) if rc else None),
            worst_best_found_percent=(float(rc['worst_best_found'] * 100) if rc else None),
            worst_drift=(float(np.max(t['drift'])) if t['drift'] else None),
            median_query_ms=median(t['ms']),
            median_device_ms=(median(t['device_ms']) if t['device_ms'] else None),
            query_repetitions=len(t['ms']),
            median_iterations=(median(t['iterations']) if t['iterations'] else None),
            median_refine_iterations=(median(t['refine_iterations'])
                                      if t['refine_iterations'] else None),
            max_stationarity=(float(np.max(t['stationarity'])) if t['stationarity'] else None),
            all_stationary=(bool(all(t['stationary'])) if t['stationary'] else None),
            all_completed=(bool(all(t['completed'])) if t['completed'] else None),
            total_budget_exits=(int(np.sum(t['budget_exits'])) if t['budget_exits'] else None),
            total_rejected_exits=(int(np.sum(t['rejected_exits'])) if t['rejected_exits'] else None),
            stop_reasons=sorted(set(t['reasons'])) if t['reasons'] else None))
    rows.sort(key=lambda x: (x['kind'] != 'rom', x['n'] if x['n'] is not None else 0,
                             x['mu'] or 0, x['arm']))
    checks['arm_table'] = rows
    checks['redecode'] = redecode

    if GATE.exists():
        g = json.loads(GATE.read_text())
        want = {x['case']: x for x in g['cases']}
        got = {x['case']: x for x in inv if x['name'] == 'n0'}
        shared = sorted(set(want) & set(got))
        deltas = [abs(got[c]['physical_error'] - want[c]['physical_error'])
                  / max(want[c]['physical_error'], 1e-300) for c in shared]
        gate('n0_reproduces_head_ablation_arm_a', bool(shared) and max(deltas) <= 1e-6,
             dict(compared=len(shared), worst_relative_delta=(max(deltas) if deltas else None),
                  bitwise_identical_fields=int(sum(1 for c in shared if
                                                   want[c]['field_sha256'] == got[c]['field_sha256'])),
                  tolerance=1e-6, reference_job=g['job_id'], reference_gpu=g['gpu'],
                  this_gpu=r.get('gpu')))
        gate('cohort_identical', bool(np.allclose(np.asarray(g['cohort']['parameters']),
                                                  np.asarray(r['cohort']['parameters']))))

    out = dict(result=str(a.result),
               result_sha256=hashlib.sha256(Path(a.result).read_bytes()).hexdigest(),
               job_id=r.get('job_id'), commit=r.get('commit'), gpu=r.get('gpu'),
               selected_alpha=r['calibration']['selected_alpha'],
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
