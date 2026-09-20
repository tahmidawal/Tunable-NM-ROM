"""Check that the attribution repair changes no measurement or scientific verdict."""
import hashlib
import json
from pathlib import Path
import subprocess

import source_provenance

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASELINE = 'df92e40df68964cf50579a496c00b6cdcb32d5b9'
SUMMARY = 'experiments/b-lowvisc/reports/summary.json'


def without_provenance(value):
    if isinstance(value, dict):
        return {key: without_provenance(item) for key, item in value.items()
                if not key.startswith('source_')}
    if isinstance(value, list):
        return [without_provenance(item) for item in value]
    return value


def main():
    old = json.loads(subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{BASELINE}:{SUMMARY}']))
    new = json.loads((ROOT / SUMMARY).read_text())
    a, b = without_provenance(old), without_provenance(new)
    assert a == b, 'A numerical, structural, or scientific field changed.'
    sources = source_provenance.build()
    assert sources == json.loads((HERE / 'source-provenance.json').read_text())
    corrections, unknown = [], []
    for before, after in zip(old['rows'], new['rows']):
        expected = sources['sources'][after['source_provenance_key']]['source_commit']
        assert after['source_commit'] == expected
        if before['source_commit'] != expected:
            corrections.append(dict(job_id=after['job_id'], arm=after['arm'],
                                    metric=after['metric'], before=before['source_commit'], after=expected))
        if expected is None:
            unknown.append(dict(job_id=after['job_id'], arm=after['arm'], metric=after['metric']))
    panel = json.loads((ROOT / 'experiments/b-lowvisc/runs/lvp01/archive/output/panel/result.json').read_text())
    panel_rows = [row for row in new['rows'] if row['job_id'] == str(panel['job_id'])]
    assert all(row['source_commit'] == panel['commit'] for row in panel_rows)
    report = dict(passed=True, baseline_commit=BASELINE,
                  summary_sha256=hashlib.sha256((ROOT / SUMMARY).read_bytes()).hexdigest(),
                  non_provenance_content_sha256=hashlib.sha256(json.dumps(a, sort_keys=True).encode()).hexdigest(),
                  rows_checked=len(new['rows']), corrected_rows=len(corrections),
                  panel_rows_checked=len(panel_rows), unknown_source_rows=unknown,
                  corrections=corrections,
                  scope='Exact recursive equality after removing only source_* metadata; '
                        'archive source hashes and Git objects rechecked. No new GPU or field audit.')
    out = HERE / 'provenance-repair-audit.json'
    out.write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('corrections', 'unknown_source_rows')}, indent=1))
    print('Unknown source rows:', len(unknown))


if __name__ == '__main__':
    main()
