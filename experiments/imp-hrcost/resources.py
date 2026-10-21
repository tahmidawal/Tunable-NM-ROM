"""imp-hrcost smoke-derived cost certificate (DESIGN A1.9, A3.5, A4.2; code audit 1):

    python resources.py <smoke run dir> <production config> -> checks/resources-<attempt>.json (exit 1 if refused)

Requires a COMPLETE real-size smoke (same mesh, r, dt) with an accepted smoke-acceptance file; reads the spent GPU-hours
from runs/GPU-LEDGER.json; projects every phase from the smoke's measured times scaled by the work counts of the
production config (worst case for the F2 Q1 search: every candidate); applies the safety factor; checks the device-memory
peak; applies the registered defer order (§8) only by dropping whole items, and records it.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAP = 3.0
SAFETY = 1.5
NCASE = {'all38': 38, 'sub16': 16, 'dev6': 6, 'dev2': 2}
FIB = 5


def n_ladder(cfg):
    return len(cfg.get('gauss', [64, 80, 96, 112, 128, 160, 192])) + (FIB if cfg.get('fib', True) else 0)


def n_cands(cfg):
    g = [p for p in cfg.get('gauss', [64, 80, 96, 112, 128, 160, 192]) if p * p <= cfg.get('mmax', 16384)]
    return len(g) + (3 if cfg.get('fib', True) else 0)


def project(R, cfg):
    ph, sc = R['phases'], R['config']
    for k in ('setup', 'R', 'F2_q2', 'F2_q1', 'fit_F3a', 'fit_F3b', 'fit_F2F3', 'timing'):
        assert k in ph, f'smoke lacks phase {k}'
    f_case = NCASE[cfg['cases']] / NCASE[sc['cases']]
    f_q2 = NCASE[cfg['q2_cases']] / NCASE[sc['q2_cases']]
    tc, ts = cfg.get('timing', {}), sc.get('timing', {})
    f_tim = NCASE[tc.get('cases', 'all38')] / NCASE[ts.get('cases', 'all38')] * \
        (tc.get('reps', 5) + tc.get('burn', 3)) / (ts.get('reps', 5) + ts.get('burn', 3))
    n_q1_smoke = max(1, sum(1 for n in R['q1'] if n.startswith('F2_')))
    per_q1 = ph['F2_q1'] / n_q1_smoke
    n_arm_s = len(R['timing'])
    n_arm_p = n_arm_s + 2                                         # up to two more F2 timing arms (m*, projected m*)
    est = dict(setup=ph['setup'], R=ph['R'] * max(f_case, f_q2),
               F2_q2=ph['F2_q2'] * f_q2 * n_ladder(cfg) / max(n_ladder(sc), 1),
               F2_q1=per_q1 * f_case * (n_cands(cfg) + 1),
               fits=sum(ph[k] for k in ('fit_F3a', 'fit_F3b', 'fit_F2F3')) * max(f_case, f_q2),
               timing=ph['timing'] * f_tim * n_arm_p / n_arm_s)
    return est, sum(est.values()) * SAFETY / 3600.


def main():
    run, cfgp = Path(sys.argv[1]), Path(sys.argv[2])
    R = json.loads((run / 'result.json').read_text())
    cfg = json.loads(cfgp.read_text())
    sc = R['config']
    assert (run / 'COMPLETE').exists(), 'smoke not complete'
    assert (sc['mesh'], sc['Rp'], sc.get('dt', .005)) == (cfg['mesh'], cfg['Rp'], cfg.get('dt', .005)), 'smoke size differs'
    acc = json.loads((HERE / 'checks' / f'smoke-accept-{sc["attempt"]}.json').read_text())
    assert acc['accepted'], 'smoke not accepted'
    led = json.loads((HERE / 'runs' / 'GPU-LEDGER.json').read_text())
    spent = float(led['total_gpu_hours'])
    peak = max(v['peak'] for v in R['mem'].values() if v['peak'] > 0)
    limit = max(v['limit'] for v in R['mem'].values())
    mem_ok = peak <= .9 * limit
    est, hours = project(R, cfg)
    deferred = []
    if spent + hours > CAP and cfg.get('fib', True):
        deferred.append('Fibonacci ladder (defer item 4); F2 claim restricted to the Gauss ladder (A2.3)')
        cfg = dict(cfg, fib=False)
        est, hours = project(R, cfg)
    ok = bool(spent + hours <= CAP and mem_ok and not deferred)
    out = dict(attempt=cfg['attempt'], smoke=str(run), smoke_attempt=sc['attempt'], estimate_seconds=est, safety=SAFETY,
               projection_hours=hours, spent_hours=spent, cap=CAP, device_peak_bytes=peak, device_limit_bytes=limit,
               mem_ok=mem_ok, deferred=deferred, ok=ok,
               note='deferred items require a new config; the certificate refuses a config that still contains them',
               config_sha256=hashlib.sha256(cfgp.read_bytes()).hexdigest(),
               memory_semantics='allocator high-water marks (running maximum, phase order recorded; A4.2)')
    (HERE / 'checks' / f'resources-{cfg["attempt"]}.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
