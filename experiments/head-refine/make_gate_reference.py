"""Extract the small in-job fidelity-gate references from the retained archives.

The head-ablation raw results are megabytes; the job only needs arm (a)'s recorded
errors, field digests and cohort on the meshes this cell runs. Extracting them here
keeps the staged tree small and makes the gate comparison auditable from a file that
is itself committed.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ABL = ROOT / 'experiments/head-ablation/artifacts/abl01/result.json'
PABL = ROOT / 'experiments/head-ablation/artifacts/pabl01/result.json'
HERE = Path(__file__).resolve().parent


def burgers(intervals):
    r = json.loads(ABL.read_text())
    rows = {}
    for x in r['invocations']:
        if x['name'] == 'a_neural_eq' and x['intervals'] == intervals:
            rows.setdefault(x['case'], dict(case=x['case'], field_sha256=x['field_sha256'],
                                            fixed_initial_max=x['error']['fixed_initial_max'],
                                            fixed_initial_per_time=x['error']['fixed_initial_per_time']))
    assert rows, intervals
    return dict(source='experiments/head-ablation/artifacts/abl01/result.json',
                source_sha256=hashlib.sha256(ABL.read_bytes()).hexdigest(),
                job_id=r['job_id'], gpu=r['gpu'], commit=r['commit'], arm='a_neural_eq',
                intervals=intervals, physical_cases=r['physical_cases'],
                cohort_roles=r['cohort_roles'], checkpoint_sha256=r['checkpoint_sha256'],
                cases=[rows[c] for c in sorted(rows)])


def poisson(intervals):
    r = json.loads(PABL.read_text())
    rows = {}
    for x in r['invocations']:
        if x['name'] == 'a_neural' and x['intervals'] == intervals:
            rows.setdefault(x['case'], dict(case=x['case'], field_sha256=x['field_sha256'],
                                            physical_error=x['physical_error'],
                                            iterations=x['iterations'], reason=x['reason']))
    assert rows, intervals
    return dict(source='experiments/head-ablation/artifacts/pabl01/result.json',
                source_sha256=hashlib.sha256(PABL.read_bytes()).hexdigest(),
                job_id=r['job_id'], gpu=r['gpu'], commit=r['commit'], arm='a_neural',
                intervals=intervals, cohort=r['cohort'], checkpoint_sha256=r['checkpoint_sha256'],
                basis_sha256=r['basis_sha256'], cases=[rows[c] for c in sorted(rows)])


if __name__ == '__main__':
    for name, value in (('gate-reference-burgers.json', burgers(256)),
                        ('gate-reference-poisson.json', poisson(1024))):
        (HERE / name).write_text(json.dumps(value, indent=2) + '\n')
        print(name, len(value['cases']), 'cases')
