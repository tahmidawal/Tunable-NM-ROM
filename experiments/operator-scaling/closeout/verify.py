"""Independently recalculate the reported test64 statistics from saved error/timing arrays."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
TARGET_PERCENT = 0.0264


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, default=HERE.parent / 'inputs' / 'test')
    parser.add_argument('--output', type=Path, default=HERE / 'audit.json')
    args = parser.parse_args()
    manifest_path = HERE / 'selection.json'
    manifest = read(manifest_path)
    assert not manifest['test_cohort_seen']
    measured = {}
    jobs = set()
    parameter_hashes = set()
    files = sorted(args.inputs.glob('*.L256.json'))
    for path in files:
        row = read(path)
        key = row['key']
        assert row['manifest_sha256'] == digest(manifest_path) and key not in measured
        assert row['mesh'] == 256 and row['cohort'] == 'test64' and 'A100' in row['gpu']
        assert (row['network'],row['scoring'],row['tf32']) == ('float32','float64',False)
        frozen = manifest['runs'][key]
        for field in ['best_sha256', 'best_step', 'checkpoint', 'validation', 'arm', 'rung', 'seed', 'real_parameters']:
            assert frozen[field] == row[field], (key, field)
        jobs.add(row['job_id'])
        parameter_hashes.add(row['parameter_sha256'])
        data = row['L256']
        errors = np.asarray(data['per_case_per_time'])
        times = np.asarray(data['timing_repetitions_ms'])
        assert errors.shape == times.shape == (64, 5)
        assert np.isfinite(errors).all() and (errors >= 0).all()
        assert np.isfinite(times).all() and (times > 0).all()
        peaks = errors.max(axis=1) * 100
        for stat, value in [('worst_percent',peaks.max()),('median_percent',np.median(peaks)),
                            ('mean_percent',peaks.mean())]:
            np.testing.assert_allclose(data[stat], value, rtol=5e-15)
        for stat, value in [('median_ms',np.median(times)),('mean_ms',times.mean()),
                            ('p05_ms',np.percentile(times,5)),('p95_ms',np.percentile(times,95)),
                            ('min_ms',times.min()),('max_ms',times.max())]:
            np.testing.assert_allclose(data['timing'][stat], value, rtol=5e-15)
        measured[key] = dict(median=float(np.median(peaks)), worst=float(peaks.max()),
                             worst_case_index=int(peaks.argmax()), ms=float(np.median(times)),
                             cases_over_target=int(np.count_nonzero(peaks>TARGET_PERCENT)),
                             sha256=digest(path))
    assert len(jobs) == 1 and set(measured) == set(manifest['runs'])
    assert parameter_hashes == {'cd058fd2a297c20296b897e54cb186033a44a8db9527d98dcb2c2dad6044527c'}
    fam_results = {}
    for family, selection in manifest['selection'].items():
        cand = {k:v['validation']['mean'] for k,v in manifest['runs'].items()
                if v['family']==family and v['budget']=='final'}
        assert cand == selection['candidates']
        selected = min(cand,key=cand.get)
        assert selected == selection['selected']
        tested = {k:v for k,v in measured.items() if manifest['runs'][k]['family']==family}
        best_test = min(tested,key=lambda k: tested[k]['median'])
        fam_results[family] = dict(selected=selected, headline=measured[selected], lowest_test_key=best_test,
                                   lowest_test=tested[best_test], reached_target=[k for k,v in tested.items() if v['median']<=TARGET_PERCENT])
    info = read(HERE / 'test64-L256.json')
    assert info['physical_sha256'] == next(iter(parameter_hashes)) and info['cases']==64
    assert info['dtype']=='float64' and info['mesh']==256 and info['max_relative_residual'] <= 1e-6*(1+1e-9)
    result = dict(job=next(iter(jobs)), checkpoint_count=len(measured), test_cases=info['cases'],
                  target_median_percent=TARGET_PERCENT, families=fam_results, per_checkpoint=measured,
                  parameter_sha256=info['physical_sha256'], fields_sha256=info['fields_sha256'],
                  selection_sha256=digest(manifest_path))
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='per_checkpoint'},indent=2))


if __name__=='__main__':
    main()
