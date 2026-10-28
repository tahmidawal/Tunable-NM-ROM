"""Recompute the fixed cost gate from retained results; no PDE/model execution."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('result'); ap.add_argument('--out', required=True)
    a = ap.parse_args(); path = Path(a.result); x = json.loads(path.read_text())
    assert x['complete'] and x['settings']['repetitions'] == 7
    assert x['settings']['L'] == 1024 and x['settings']['M'] == 256 and x['settings']['fine_K'] == 64
    assert x['settings']['active_K'] == 16 and x['settings']['rule'] == 'gauss128'
    assert x['backend'] == 'gpu' and x['x64'] and x['precision'] == 'highest'
    arms = ('direct', 'affine')
    timings = {(t['arm'], t['case'], t['block']): t for t in x['timings']}
    assert len(timings) == len(x['timings']) == 84
    assert set(timings) == {(arm, c, b) for arm in arms for c in range(6) for b in range(7)}
    records = {(r['arm'], r['case']): r for r in x['records']}
    assert len(records) == len(x['records']) == 12
    # Read each witness once; every repetition's hash must name its actual arrays.
    witness_meta = {}
    for t in x['timings']:
        wpath = t['array_witness']
        if wpath not in witness_meta:
            with np.load(path.parent/wpath) as z:
                witness_meta[wpath] = (sha(z['W']), sha(z['fields']), int(z['it_sum']))
        assert witness_meta[wpath] == (t['W_sha256'], t['fields_sha256'], t['fine_corrections'])
    quality = True
    for arm in arms:
        for c in range(6):
            r = records[(arm, c)]
            with np.load(path.parent/f'{arm}_case{c}.npz') as z:
                fresh_ratio = z['residual_norm_fresh']/(r['initial_scale']*1e-9+1e-300)
                stop = bool(np.all(np.isfinite(z['gamma_fresh'])) and
                    np.all((z['gamma_fresh']<=1e-3)|(fresh_ratio<=1)))
                np.testing.assert_allclose(z['gamma_fresh'], z['step_gn'], rtol=1e-7, atol=1e-12)
                np.testing.assert_allclose(fresh_ratio, z['step_tolratio'], rtol=1e-7, atol=1e-12)
                assert stop == r['fresh_stationarity']
                valid = stop and int(z['nfail']) == 0 and int(z['ic_reason']) in (1, 4)
                assert valid or not r['valid']
                assert int(z['it_sum']) == r['fine_corrections'] and 50+int(z['it_sum']) == r['fine_J']
                if arm == 'affine':
                    info = z['seed_info']; accepted = info[:, 0].astype(bool)
                    assert np.all(info[:, 1] <= 600) and np.all(info[:, 12] == 0)
                    assert int(info[:, 9].sum()) == r['coarse_J'] == 50+int(info[:, 1].sum())
                    assert int(info[:, 10].sum()) == r['fine_guard_residuals'] == 100
                    assert np.all(np.isin(info[accepted, 2], [1, 4]))
                    assert np.all(info[accepted, 8] < info[accepted, 7])
                    assert np.all(info[accepted, 8] <= (1-1e-8)*info[accepted, 7])
                    assert np.array_equal(z['seed_codes'][:, 16:64], z['seed_codes'][:, 80:128])
                    with np.load(path.parent/f'diagnostics_affine_case{c}.npz') as dg:
                        assert np.array_equal(accepted, dg['accepted'])
                        assert np.all(dg['tail_displacements'] == 0) and np.max(dg['predictor_gaps']) <= 1e-12
                        np.testing.assert_allclose(info[:, 7:9], dg['objectives'], rtol=1e-7, atol=1e-12, equal_nan=True)
            quality = quality and r['valid']
    assert {(r['check'], r['case']) for r in x['parity']} == {('disabled', c) for c in range(6)} | {('forced_failure', 0)}
    controls = all(r['passed'] for r in x['parity']) and len(x['diagnostics']) == 6 and all(r['passed'] for r in x['diagnostics'])
    assert len(x['checks']['exact_prefix']) == 2 and all(r['passed'] for r in x['checks']['exact_prefix'])
    assert x['checks']['expansion']['passed'] and all(r['passed'] for r in x['checks']['active_jacobian'])
    identity = all(t['identity'] and t['online_valid'] for t in x['timings'])
    ts = {arm: np.array([[timings[(arm, c, b)]['seconds'] for b in range(7)] for c in range(6)]) for arm in arms}
    ratio = ts['affine']/ts['direct']; med = np.median(ratio, axis=1)
    passed = bool(quality and controls and identity and np.median(med)<=.85 and np.max(med)<=1.10)
    np.testing.assert_allclose(med, x['gate']['per_case_paired_median_ratios'], rtol=1e-14, atol=1e-14)
    assert passed == x['gate']['passed'] == x['continue_accuracy']
    summary = dict(scope=x['scope'], source_result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        job_id=x['job_id'], gpu=x['gpu'], commit=x['commit'], settings=x['settings'],
        model_manifest_sha256=x['model_manifest_sha256'], gate=x['gate'],
        all_fresh_certificates=quality, all_controls=controls, all_timing_identities=identity,
        paired_ratio_arrays=ratio.tolist(), slower_pairs=int(np.sum(ratio>1)), total_pairs=42,
        strong_accuracy_checks=x['strong_accuracy_checks'],
        interpretation='Affine16-coordinate initialization; last48 fine predictor coordinates are held fixed. Not a standalone D16 trajectory or multigrid.',
        elapsed_program_seconds=x['elapsed_program_seconds'], setup=x['setup'], arms={})
    for arm in arms:
        rec = [records[(arm, c)] for c in range(6)]
        summary['arms'][arm] = dict(query_seconds=ts[arm].tolist(),
            per_case_median_seconds=np.median(ts[arm], axis=1).tolist(),
            median_of_case_median_seconds=float(np.median(np.median(ts[arm], axis=1))),
            total_fine_corrections=sum(r['fine_corrections'] for r in rec),
            median_fine_corrections=float(np.median([r['fine_corrections'] for r in rec])),
            total_fine_J=sum(r['fine_J'] for r in rec),
            total_aux_corrections=int(sum(r['auxiliary_sums']['aux_it'] for r in rec)),
            total_aux_J=sum(r['coarse_J'] for r in rec),
            total_aux_failures=sum(r['auxiliary_failures'] for r in rec),
            total_guard_residuals=sum(r['fine_guard_residuals'] for r in rec),
            total_predictor_residuals=6*50*3,
            accepted_candidates=int(sum(r['auxiliary_sums']['accepted'] for r in rec)),
            rejected_candidates=0 if arm == 'direct' else 300-int(sum(r['auxiliary_sums']['accepted'] for r in rec)),
            initial_fit_J_count='Not exposed by unchanged initializer; full initial-fit runtime included in each query.')
    opath = Path(a.out); opath.parent.mkdir(parents=True, exist_ok=True)
    opath.write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
