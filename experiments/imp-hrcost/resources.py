"""imp-hrcost smoke-derived cost certificate (DESIGN A1.9, A3.5): projects the production job from the smoke's measured
phase times, scaled by case counts, and refuses submission unless spent + projection <= cap.

    python resources.py <smoke result.json> <production config> [--spent H] -> checks/resources-<attempt>.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAP = 3.0
SAFETY = 1.5


def project(smk, prod_cfg):
    ph = smk['phases']
    sc = smk['config']
    cases = {'all38': 38, 'sub16': 16, 'dev6': 6, 'dev2': 2}
    f_case = cases[prod_cfg.get('cases', 'all38')] / cases[sc.get('cases', 'all38')]
    f_q2 = cases[prod_cfg.get('q2_cases', 'sub16')] / cases[sc.get('q2_cases', 'sub16')]
    f_tim = cases[prod_cfg.get('timing', {}).get('cases', 'all38')] / cases[sc.get('timing', {}).get('cases', 'all38')]
    f_tim *= (prod_cfg.get('timing', {}).get('reps', 3) + prod_cfg.get('timing', {}).get('burn', 2)) / \
        (sc.get('timing', {}).get('reps', 3) + sc.get('timing', {}).get('burn', 2))
    n_lad_s = len(sc.get('gauss', [])) + (5 if sc.get('fib', True) else 0)
    n_lad_p = len(prod_cfg.get('gauss', [64, 80, 96, 112, 128, 160, 192])) + (5 if prod_cfg.get('fib', True) else 0)
    est = dict(setup=ph['setup'],
               R=ph['R'] * max(f_case, f_q2),
               F2_q2=ph.get('F2_q2', 0.) * f_q2 * n_lad_p / max(n_lad_s, 1),
               F2_q1=ph.get('F2_q1', 0.) * f_case * 2.,                       # up to two F2 Q1 blocks (m*, 128^2)
               fits=sum(v for k, v in ph.items() if k.startswith('fit_')) * max(f_case, f_q2),
               timing=ph.get('timing', 0.) * f_tim)
    sec = sum(est.values()) * SAFETY
    return est, sec / 3600.


def main():
    smk = json.loads(Path(sys.argv[1]).read_text())
    cfgp = Path(sys.argv[2])
    cfg = json.loads(cfgp.read_text())
    spent = float(sys.argv[sys.argv.index('--spent') + 1]) if '--spent' in sys.argv else 0.
    est, hours = project(smk, cfg)
    ok = spent + hours <= CAP
    out = dict(attempt=cfg['attempt'], smoke=str(sys.argv[1]), estimate_seconds=est, safety=SAFETY,
               projection_hours=hours, spent_hours=spent, cap=CAP, ok=ok,
               config_sha256=hashlib.sha256(cfgp.read_bytes()).hexdigest())
    (HERE / 'checks' / f'resources-{cfg["attempt"]}.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
