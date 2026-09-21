"""Generate SPEED-LOG rows (same-job before -> after) from one audit summary; no number is typed.

    python checks/speedlog_rows.py checks/hb4k03-summary.json H5:chol:<before>:<after> ...

Each pair is HYP:LABEL:BEFORE_ARM:AFTER_ARM. Prints markdown rows in SPEED-LOG's format; the
kept/reverted column is the author's decision and is left as `?` for them to fill in.
"""
import json
import sys
from pathlib import Path

LABEL = {
    'chol': 'Cholesky',
    'clip': '`clip` (shorten an over-long z-step onto the trust radius)',
    'lamcarry': '`lamcarry` (carry the damping between steps)',
    'pred2': '`pred2` (quadratic-extrapolation guard, one batched residual)',
    'gtol': 'looser LM stationarity tolerance (labelled tolerance arm)',
    'M': 'test count M',
}


def main():
    path = Path(sys.argv[1])
    s = json.loads(path.read_text())
    t = s['table']
    gpu = s['nvidia_smi'][0].split(' (UUID')[0].split(': ')[-1] if s['nvidia_smi'] else s['gpu']
    tag = f"{s['attempt']} ({s['job_id']}, {gpu.replace('NVIDIA ', '')}, {s['intervals']}², {'hold64' if (s.get('cohort') or '').startswith('hold64') else 'dev6'})"
    for spec in sys.argv[2:]:
        hyp, lab, a, b = spec.split(':')
        x, y = t[a], t[b]
        gate = (f"algorithmic arm, error measured directly: error {x['worst_evolved_percent']:.4f} → {y['worst_evolved_percent']:.4f} %, "
                f"stalled {x['stalled_exits']} → {y['stalled_exits']}, rejected trial steps {x['damping_retries']} → {y['damping_retries']}, "
                f"median LM iterations per query {x['total_iterations_median']:g} → {y['total_iterations_median']:g}, "
                f"worst exit stationarity {x['worst_joint_stationarity']:.2e} → {y['worst_joint_stationarity']:.2e}, "
                f"certified {x.get('certified_primary')} → {y.get('certified_primary')}")
        print(f"| 2026-09-21 | {tag} | {hyp} | {LABEL.get(lab, lab)}: `{a}` → `{b}` | {gate} | "
              f"{x['median_gpu_ms']:.2f} → {y['median_gpu_ms']:.2f} ms ({x['median_gpu_ms'] / y['median_gpu_ms']:.2f}×); "
              f"host-inclusive {x['median_host_ms']:.1f} → {y['median_host_ms']:.1f} ms | ? |")


if __name__ == '__main__':
    main()
