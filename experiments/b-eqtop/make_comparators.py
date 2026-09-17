"""Build `checks/comparators.json` from the qrg304 audit: both reported metrics for every
arm of the parent job, read from its committed audit JSON. Nothing is typed by hand."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
AUDIT = ROOT / 'experiments/q-ridge/checks/qrg304-audit.json'
KEEP = ('arm', 'q', 'M', 'm', 'rule_kind', 'quadrature', 'converged', 'total_budget_exits',
        'max_joint_stationarity', 'median_gpu_ms', 'worst_reference_percent',
        'worst_all_times_percent', 'worst_evolved_percent', 'worst_t0_compression_percent',
        'rho_max', 'rho_p95', 'certified_primary', 'relative_fit')


def main():
    a = json.loads(AUDIT.read_text())
    arms = {x['arm']: {k: x.get(k) for k in KEEP} for x in a['arms']}
    out = dict(qrg304=dict(job_id=a['job_id'], gpu=a['gpu'], commit=a['commit'],
                           source=str(AUDIT.relative_to(ROOT)), arms=arms))
    (HERE / 'checks').mkdir(exist_ok=True)
    (HERE / 'checks/comparators.json').write_text(json.dumps(out, indent=2) + '\n')
    print(len(arms), 'comparator arms from job', a['job_id'])


if __name__ == '__main__':
    main()
