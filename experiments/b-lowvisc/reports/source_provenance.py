"""Reconstruct report attribution from retained archives, never from the current HEAD.

This checks archived source bytes against both their manifests and named Git objects.
It does not rerun or approve scientific measurements.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
LANE = ROOT / 'experiments/b-lowvisc'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path):
    return dict(path=os.path.relpath(path, ROOT), sha256=digest(path))


def checked_archive(base):
    commit_path = base / 'COMMIT.txt'
    if not commit_path.exists():
        commit_path = base.parent / 'COMMIT.txt'
    commit = commit_path.read_text().strip()
    manifest = base / 'PROVENANCE.json'
    checked = []
    for item in json.loads(manifest.read_text()):
        # Some historical b-panel external dependencies have no Git object. Their
        # archive hash is still checked, but they cannot certify a source commit.
        path = base / item.get('staged_as', item['source'])
        git_source = item['source']
        external = item.get('provenance', '').startswith('out-of-band copy')
        no_git_object = 'has no Git object' in item.get('provenance', '')
        if external:
            path = base / ('fnockpt' if no_git_object else 'fnocode') / Path(item['source']).name
            git_source = item['source'].split('/', 1)[1]
        if not path.exists():
            path = base.parent / path.relative_to(base)
        if no_git_object and not path.exists():
            checked.append(dict(source=item['source'], sha256=item['sha256'],
                                source_commit=None, git_blob_checked=False,
                                archive_bytes_checked=False,
                                status='Excluded external operator checkpoint; manifest hash only.'))
            continue
        assert digest(path) == item['sha256'], path
        git_checked = False
        if item.get('commit') and not no_git_object:
            blob = subprocess.check_output(
                ['git', '-C', str(ROOT), 'show', f"{item['commit']}:{git_source}"])
            assert hashlib.sha256(blob).hexdigest() == item['sha256'], path
            git_checked = True
        checked.append(dict(source=item['source'], sha256=item['sha256'],
                            source_commit=item.get('commit'), git_blob_checked=git_checked,
                            archive_bytes_checked=True))
    assert any(x['git_blob_checked'] and x['source_commit'] == commit for x in checked)
    return dict(source_commit=commit, manifest=record(manifest),
                commit_record=record(commit_path), checked_sources=checked)


def build():
    stages, sources = {}, {}
    for stage, attempt, result_name in (
            ('gate', 'lvg01', 'output/gate/result.json'),
            ('train', 'lvt01', 'output/train/hfit_full.json'),
            ('panel', 'lvp01', 'output/panel/result.json')):
        base = LANE / 'runs' / attempt / 'archive'
        entry = checked_archive(base)
        path = base / result_name
        result = json.loads(path.read_text())
        job = str(result['config']['slurm_job'] if stage == 'train' else result['job_id'])
        if stage != 'train':
            assert result['commit'] == entry['source_commit']
        entry.update(job_id=job, result=record(path))
        stages[stage] = job
        sources[job] = entry

    for seed in (1, 2, 3):
        other = ROOT.parent / '2026-09-17-b-seeds/experiments/b-seeds'
        entry = checked_archive(other / f'runs/s{seed}/archive')
        paths = [other / f'artifacts/s{seed}/train-{name}' for name in
                 ('sep_coeff_N256_K16_R512.json', 'hfit_full.json')]
        jobs = {str(json.loads(p.read_text())['config']['slurm_job']) for p in paths}
        assert len(jobs) == 1, jobs
        entry.update(job_id=jobs.pop(), result_records=[record(p) for p in paths])
        sources[f'b-seeds-s{seed}'] = entry

    other = ROOT.parent / '2026-09-17-b-panel/experiments/b-panel'
    entry = checked_archive(other / 'runs/bpn101/archive')
    result_path = other / 'artifacts/bpn101/result.json'
    result = json.loads(result_path.read_text())
    assert result['commit'] == entry['source_commit']
    comparator = LANE / 'comparators/bpn301-summary.json'
    comparator_rows = json.loads(comparator.read_text())['rows']
    job = str(result['job_id'])
    assert all(digest(result_path) == r['source_sha']
               for r in comparator_rows if str(r['job_id']) == job)
    entry.update(job_id=job, result=record(result_path), comparator=record(comparator))
    sources[job] = entry

    # This older extraction/head run retains result hashes and a staged-file
    # manifest, but no run COMMIT.txt or source-commit field. A matching file in a
    # later checkout would not establish which checkout actually ran the job.
    base = ROOT / 'experiments/separable-decoder/runs/dn256b'
    paths = [base / 'out' / name for name in
             ('sep_coeff_N256_K16_R512.json', 'hfit_dn256b_full.json')]
    jobs = {str(json.loads(p.read_text())['config']['slurm_job']) for p in paths}
    assert len(jobs) == 1, jobs
    job = jobs.pop()
    sources[job] = dict(job_id=job, source_commit=None,
                       status='source commit unavailable in retained run metadata',
                       manifest=record(base / 'MANIFEST.sha256'),
                       result_records=[record(p) for p in paths])
    return dict(stages=stages, sources=sources)


if __name__ == '__main__':
    out = Path(__file__).with_name('source-provenance.json')
    out.write_text(json.dumps(build(), indent=1) + '\n')
    print(out)
