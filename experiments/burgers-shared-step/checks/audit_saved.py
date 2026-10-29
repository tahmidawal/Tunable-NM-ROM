"""Saved-array audit and descriptive timing counts; never imports a PDE/model."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(path):
    rows = []
    for line in path.read_text().splitlines():
        digest, name = line.split('  ', 1)
        assert sha(path.parent/name) == digest, name
        rows.append(dict(path=name, sha256=digest))
    return rows


def load(path):
    with np.load(path) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--artifact', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); root = Path(a.artifact); data = root/'output/dev6'
    x = json.loads((data/'result.json').read_text())
    assert x['complete'] and not x['continue_accuracy'] and not x['gate']['passed']
    inputs = manifest(root/'MANIFEST.sha256'); outputs = manifest(root/'OUTPUTS.sha256')
    assert {r['path'] for r in outputs} == {str(p.relative_to(root)) for p in (root/'output').rglob('*') if p.is_file()}
    logs = '\n'.join(p.read_text() for p in (root/'logs').glob('*'))
    assert 'jax_backend=gpu x64=True precision=highest' in logs and 'ALL-DONE' in logs
    assert not (root/'logs'/f"{x['job_id']}.err").read_text().strip()
    accounting = (root/'ACCOUNTING.txt').read_text().splitlines()
    keys = accounting[0].split('|')
    job = next(dict(zip(keys, row.split('|'))) for row in accounting[1:] if row.split('|')[0] == x['job_id'])
    assert job['State'] == 'COMPLETED' and job['ExitCode'] == '0:0' and 'gres/gpu=1' in job['AllocTRES']
    records = {(r['arm'], r['case']): r for r in x['records']}
    times = {(r['arm'], r['case'], r['block']): r for r in x['timings']}
    assert len(records) == 12 and len(times) == 84
    decisions = ('step_it', 'step_reason', 'it_sum', 'it_max', 'rej', 'exits', 'nfail', 'ic_it', 'ic_reason', 'ic_idx')
    per_case = []
    for c in range(6):
        arrays = {}; row = dict(case=c)
        for arm in ('direct', 'affine'):
            z = load(data/f'{arm}_case{c}.npz'); witness = load(data/f'timed_identity_{arm}_case{c}.npz')
            arrays[arm] = z; r = records[(arm, c)]
            assert np.isfinite(z['states']).all() and np.isfinite(z['fields']).all()
            np.testing.assert_array_equal(z['W'], z['states'][::10])
            assert int(z['it_sum']) == int(z['step_it'].sum()) == r['fine_corrections']
            assert int(z['it_max']) == int(z['step_it'].max())
            np.testing.assert_array_equal(z['exits'], np.bincount(z['step_reason'], minlength=5))
            assert int(z['nfail']) == 0 and int(z['ic_reason']) in (1, 4)
            assert np.all(np.isin(z['step_reason'], [1, 4]))
            np.testing.assert_allclose(z['gamma_fresh'], z['step_gn'], rtol=1e-7, atol=1e-12)
            tolratio = z['residual_norm_fresh']/(r['initial_scale']*1e-9+1e-300)
            np.testing.assert_allclose(tolratio, z['step_tolratio'], rtol=1e-7, atol=1e-12)
            assert np.all((z['gamma_fresh'] <= 1e-3) | (tolratio <= 1))
            for key in ('W', 'fields'):
                np.testing.assert_allclose(witness[key], z[key], rtol=1e-12, atol=1e-13)
            for key in decisions:
                np.testing.assert_array_equal(witness[key], z[key])
            np.testing.assert_allclose(witness['step_gn'], z['gamma_fresh'], rtol=1e-7, atol=1e-12)
            np.testing.assert_allclose(witness['step_tolratio'], tolratio, rtol=1e-7, atol=1e-12)
            elapsed = np.array([times[(arm, c, b)]['seconds'] for b in range(7)])
            assert np.isfinite(elapsed).all() and np.all(elapsed > 0)
            q1, q3 = np.quantile(elapsed, [.25, .75]); spread = q3-q1
            outlier = (elapsed < q1-1.5*spread) | (elapsed > q3+1.5*spread)
            row[arm] = dict(median_ms=float(np.median(elapsed)*1e3), seconds=elapsed.tolist(),
                descriptive_tukey_outliers=int(outlier.sum()), outlier_blocks=np.flatnonzero(outlier).tolist(),
                fine_corrections=int(z['it_sum']), fine_J=50+int(z['it_sum']), fine_rejections=int(z['rej']),
                fine_exits=z['exits'].tolist(), initial_iterations=int(z['ic_it']), initial_reason=int(z['ic_reason']),
                gamma_max=float(z['gamma_fresh'].max()))
            if arm == 'affine':
                info = z['seed_info']; good = info[:, 0].astype(bool)
                np.testing.assert_allclose(witness['seed_info'], info, rtol=1e-12, atol=1e-13)
                assert np.all(np.isfinite(info))
                assert np.all(np.isin(info[:, 2], [1, 4]))
                assert np.all((info[:, 3] <= 1e-3) | (info[:, 5] <= r['initial_scale']*1e-9))
                assert np.all(info[:, 1] <= 600) and np.all(info[:, 9] == info[:, 1]+1)
                assert np.all(info[:, 10] == 2) and np.all(info[:, 12] == 0)
                np.testing.assert_array_equal(z['seed_codes'][:, 16:64], z['seed_codes'][:, 80:128])
                dg = load(data/f'diagnostics_affine_case{c}.npz')
                np.testing.assert_array_equal(good, dg['accepted'])
                np.testing.assert_allclose(info[:, 7:9], dg['objectives'], rtol=1e-7, atol=1e-12)
                assert np.all(dg['tail_displacements'] == 0) and np.max(dg['predictor_gaps']) <= 1e-12
                assert np.all(dg['objectives'][good, 1] < dg['objectives'][good, 0])
                assert np.all(dg['objectives'][good, 1] <= (1-1e-8)*dg['objectives'][good, 0])
                row[arm].update(aux_corrections=int(info[:, 1].sum()), aux_J=int(info[:, 9].sum()),
                    aux_rejections=int(info[:, 4].sum()), aux_exits=np.bincount(info[:, 2].astype(int), minlength=5).tolist(),
                    accepted=int(good.sum()), rejected=int((~good).sum()), guard_residuals=int(info[:, 10].sum()))
        np.testing.assert_allclose(arrays['direct']['W'][0], arrays['affine']['W'][0], rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(arrays['direct']['fields'][0], arrays['affine']['fields'][0], rtol=1e-12, atol=1e-13)
        for key in ('ic_it', 'ic_reason', 'ic_idx'):
            np.testing.assert_array_equal(arrays['direct'][key], arrays['affine'][key])
        paired = np.asarray(row['affine']['seconds'])/np.asarray(row['direct']['seconds'])
        row.update(paired_ratios=paired.tolist(), median_paired_ratio=float(np.median(paired)),
                   slower_pairs=int((paired > 1).sum()), faster_pairs=int((paired < 1).sum()))
        per_case.append(row)
    ratios = [r['median_paired_ratio'] for r in per_case]
    np.testing.assert_allclose(ratios, x['gate']['per_case_paired_median_ratios'], rtol=1e-14, atol=1e-14)
    audit = dict(status='PASS: accept scoped negative cost result and stop',
        result_sha256=sha(data/'result.json'), model_manifest_sha256=sha(root/'model/MODEL.json'),
        source_commit=x['commit'], job=job, gpu=x['gpu'], settings=x['settings'],
        input_manifest_files=len(inputs), output_manifest_files=len(outputs),
        all_recorded_timed_agreements=True, all_count_and_guard_checks=True,
        all_cases_production_stationary=True, all_timings_retained=True, excluded_timings=0,
        outlier_definition='Descriptive Tukey fences per arm/case (1.5 times IQR); no exclusion or gate change.',
        per_case=per_case, median_paired_ratio=float(np.median(ratios)), worst_case_ratio=max(ratios),
        slower_pairs=sum(r['slower_pairs'] for r in per_case), total_pairs=42,
        limits=['Only the six exposed development cases and fixed affine16 slice were tested.',
                'No tight/tighter trajectory agreement, successor-quadrature or physical accuracy qualification followed cost failure.',
                'The zero-tail standalone16 replay is a separate experiment; this test is not multigrid.',
                'Initial-fit Jacobian counts are not exposed; their full runtime is included.'])
    Path(a.out).write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:audit[k] for k in ('status', 'median_paired_ratio', 'worst_case_ratio', 'slower_pairs', 'total_pairs')}, indent=2))


if __name__ == '__main__': main()
