"""Verify the collected sparse-lane outputs, historical inputs and optional full archives."""
import argparse
import hashlib
import json
import math
import statistics
import subprocess
import tarfile
from pathlib import Path

AUDIT = Path(__file__).resolve().parent
LANE = AUDIT.parent.parent


def hashes(path):
    return {name: digest for digest, name in (line.split('  ', 1) for line in path.read_text().splitlines())}


def verify_archives(directory):
    archive_root = AUDIT / 'archives'
    members = {}
    for name, expected_hash in hashes(archive_root / 'archives.sha256').items():
        path = directory / name
        with path.open('rb') as archive:
            assert hashlib.file_digest(archive, 'sha256').hexdigest() == expected_hash, name
        expected = hashes(archive_root / name.replace('.tar.zst', '.members.sha256'))
        actual = {}
        with subprocess.Popen(['zstd', '-dqc', path], stdout=subprocess.PIPE) as zstd:
            with tarfile.open(fileobj=zstd.stdout, mode='r|') as bundle:
                for item in bundle:
                    if item.isdir():
                        continue
                    assert item.isfile() and item.name not in actual, item.name
                    with bundle.extractfile(item) as f:
                        actual[item.name] = hashlib.file_digest(f, 'sha256').hexdigest()
            # tar may stop at its first end marker. Drain any trailing padding before wait.
            while zstd.stdout.read(1024 * 1024):
                pass
            assert zstd.wait() == 0
        assert actual == expected, name
        members[name] = len(actual)
    return members


def verify_source():
    count = 0
    for manifest in sorted(AUDIT.glob('ours*/*/PROVENANCE.json')):
        for row in json.loads(manifest.read_text()):
            spec = row['commit'] + ':' + row['source']
            blob = subprocess.check_output(['git', 'cat-file', 'blob', spec], cwd=LANE)
            assert hashlib.sha256(blob).hexdigest() == row['sha256'], spec
            count += 1
    amgx = AUDIT / 'amgx-2d-small'
    revision = (amgx / 'SOURCE_COMMIT.txt').read_text().strip()
    for name, value in hashes(amgx / 'SOURCE-SHA256SUMS').items():
        spec = revision + ':experiments/sparse-solvers/' + name
        blob = subprocess.check_output(['git', 'cat-file', 'blob', spec], cwd=LANE)
        assert hashlib.sha256(blob).hexdigest() == value, spec
        count += 1
    return count


def verify_amgx(r):
    assert r['dtype'] == 'float64' and r['mode'] == 'dDDI' and r['status'] == 'ok'
    assert r['reference']['all_converged'] and r['reference']['cohort_sha_matches_paper']
    assert all(check['passed'] for check in r['checks'].values())
    assert len(r['settings']) == 32 and r['cases'] == 16
    for setting in r['settings'].values():
        assert setting['status'] == 'ok' and setting['finite']
        rows = setting['per_case']
        assert [v['case'] for v in rows] == list(range(r['cases']))
        assert all(len(v['times']) == setting['reps_per_case'] for v in rows)
        assert all(v['unconverged_steps'] == v['linear_not_converged'] == 0 for v in rows)
        times = [t for v in rows for t in v['times']]
        errors = [v['error'] for v in rows]
        assert all(math.isfinite(t) and t > 0 for t in times)
        assert all(math.isfinite(e) and e >= 0 for e in errors)
        for key, reduced in [('median', statistics.median(times)), ('min', min(times)), ('max', max(times))]:
            assert math.isclose(reduced, setting['time_s'][key], rel_tol=1e-12)
        assert math.isclose(statistics.median(errors), setting['median_error'], rel_tol=1e-12)
        assert max(errors) == setting['worst_error']
    return {'settings': len(r['settings']), 'cases': r['cases'], 'job': r['slurm_job']}


def verify_rom(r, label, paper, summary):
    assert r['complete'] and r['backend'] == 'gpu' and r['x64'] and r['matmul_precision'] == 'highest'
    poisson = label.startswith('p2')
    if poisson:
        L = str(r['intervals'])
        ratios = [v['ratio'] for v in r['neighbour_gate']['rows']]
        passed = max(ratios) <= r['neighbour_gate']['limit']
        assert r['gates']['neighbour'] == passed
        assert all(v for k, v in r['gates'].items() if k != 'neighbour')
        assert passed == (L != '256')
        rows = r['invocations'] + r['slow_invocations']
        problem = 'poisson2d'
    else:
        L = str(r['intervals'])
        assert all(v['identical_to_quick'] for v in r['invocations'])
        for name, gate in r['gates'].items():
            assert all(item['passed'] for item in gate) if isinstance(gate, list) else gate['passed'], name
        rows = r['invocations']
        passed = True
        problem = 'burgers2d'
    for role in ('accurate', 'fast'):
        target = paper[problem][L][role]
        selected = [v for v in rows if v['name'] == target['arm']]
        if poisson:
            assert all(v['matches_saved_field'] for v in selected)
            errors = {v['case']: 100 * v['same_grid_error'] for v in selected}
            seconds = [v['fused_device_seconds'] for v in selected]
        else:
            errors = {v['case']: 100 * v['same_grid_evolved'] for v in r['quick'] if v['name'] == target['arm']}
            seconds = [v['gpu_seconds'] for v in selected]
        assert set(errors) == {v['case'] for v in selected}
        assert all(math.isfinite(t) and t > 0 for t in seconds)
        assert math.isclose(max(errors.values()), target['worst_percent'], rel_tol=1e-6)
        assert math.isclose(statistics.median(errors.values()), target['median_percent'], rel_tol=1e-6)
        reported = next(v for v in summary['_all'] if v['problem'] == problem and v['L'] == int(L) and
                        v['role'] == role and v['job_id'] == r['job_id'])
        assert reported['timing_usable'] == passed
        assert math.isclose(1000 * statistics.median(seconds), reported['ms'], rel_tol=1e-12)
        assert bool(summary.get(problem, {}).get(L, {}).get(role)) == passed
    return {'timing_usable': passed, 'job': r['job_id'], 'cases': len(errors),
            'failed_controls': reported['failed_controls']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives', type=Path)
    args = parser.parse_args()
    paper = json.loads((LANE / 'ours_burgers.json').read_text())
    paper.update(json.loads((LANE / 'ours_poisson2d.json').read_text()))
    stats = json.loads((LANE / 'results/ours/same_gpu.json').read_text())
    verdict = {'verified_source_entries': verify_source(), 'outputs': {}}
    for remote_file, digest in hashes(AUDIT / 'remote-results.sha256').items():
        location = Path(remote_file)
        if location.parts[0] == 'amgx-2d-small':
            f = LANE / 'results/step/h200' / location.name
        else:
            f = LANE / 'results/ours/h200' / (location.parts[1] + '.json')
        raw = f.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest, f
        r = json.loads(raw)
        details = verify_amgx(r) if f.name.startswith('amgx_') else verify_rom(r, f.stem, paper, stats)
        verdict['outputs'][str(f.relative_to(LANE))] = details
    assert len(verdict['outputs']) == 7
    if args.archives:
        verdict['verified_archive_members'] = verify_archives(args.archives)
    print(json.dumps(verdict, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
