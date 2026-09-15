"""Freeze a shortlist of at most four settings from the calibration pass.

Selection is mechanical: the Pareto front of (worst calibration physical error,
median GPU seconds) over every measured reduced-order arm, always keeping the
archived native configuration and the most accurate arm, truncated to the cap.
Quadrature rules the shortlist needs are copied verbatim from the calibration
index so the held-out job rebuilds identical operators without refitting.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cap', type=int, default=4)
    args = parser.parse_args()
    report = json.loads(args.calibration.read_text())
    arms = {}
    for row in report['invocations']:
        if not row.get('arm') or not row.get('error'):
            continue
        arms.setdefault(row['method'], dict(spec=row['arm'], errors=[], times=[]))
        arms[row['method']]['errors'].append(row['error']['maximum'])
        arms[row['method']]['times'].append(row['gpu_seconds'])
    scored = []
    for name, value in arms.items():
        times = sorted(value['times'])
        median = times[len(times) // 2] if len(times) % 2 else .5 * (times[len(times) // 2 - 1] + times[len(times) // 2])
        scored.append(dict(name=name, spec=value['spec'], worst_error=max(value['errors']), median_gpu_seconds=median))
    front = [a for a in scored if not any(b['worst_error'] <= a['worst_error']
                                          and b['median_gpu_seconds'] <= a['median_gpu_seconds']
                                          and (b['worst_error'] < a['worst_error']
                                               or b['median_gpu_seconds'] < a['median_gpu_seconds'])
                                          for b in scored)]
    keep = []
    native = next((a for a in scored if a['spec']['setting'] == 'native' and a['spec']['quadrature'] == 'm256'), None)
    accurate = min(scored, key=lambda a: a['worst_error'])
    for candidate in [native, accurate] + sorted(front, key=lambda a: a['median_gpu_seconds']):
        if candidate and candidate['name'] not in [k['name'] for k in keep] and len(keep) < args.cap:
            keep.append(candidate)
    needed = {a['spec']['quadrature'] for a in keep}
    rules = []
    for rule in report['eq_rules']:
        name = 'full' if rule.get('requested_m') is None else f"m{rule['requested_m']}"
        if name in needed and rule.get('eq_indices'):
            rules.append(dict(name=name, eq_indices=rule['eq_indices'], eq_weights=rule['eq_weights'],
                              info={k: v for k, v in rule.items() if k not in ('eq_indices', 'eq_weights')}))
    shortlist = dict(schema_version=1, frozen_on='calibration cases only; held-out cases were not consulted',
                     selection=('Pareto front of worst calibration physical error against median GPU seconds, '
                                'always retaining the archived native configuration and the most accurate arm, '
                                f'truncated to {args.cap} settings'),
                     calibration_index_sha256=hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
                     calibration_job_id=report['provenance']['job_id'],
                     candidates=scored, pareto_front=[a['name'] for a in front],
                     arms=[dict(name=a['name'], quadrature=a['spec']['quadrature'], setting=a['spec']['setting'],
                                calibration_worst_error=a['worst_error'],
                                calibration_median_gpu_seconds=a['median_gpu_seconds']) for a in keep],
                     rules=rules)
    args.out.write_text(json.dumps(shortlist, indent=2, allow_nan=False) + '\n')
    print(json.dumps([a['name'] for a in keep]))
    print(args.out, hashlib.sha256(args.out.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
