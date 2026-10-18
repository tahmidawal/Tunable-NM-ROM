"""Smoke-derived cost certificates (DESIGN 9.1, A1.11; pattern of jcp-speed3d resources.py). Writes
checks/resources-<attempt>.json with projection_hours; cluster/stage.py requires time limit >= 1.5 x projection.

Measurements: the smoke's p1acc log ("RUN <config> <scheme> <cohort> <case> <elapsed>": per-case wall time of the whole
per-case block - prod + certification + tight + tighter + q+ + q++ + Q3), its configs' setup seconds, the smoke timing
job's per-triplet wall time, the kernel jobs' elapsed time, and the H200 pilot's per-run seconds at 4096^2.
Proxies are conservative ESTIMATES (not established bounds; iteration counts and compilation are not bounded): a configuration not measured takes the measured configuration with the next larger
(r, m), scaled by the point-count ratio m / m_measured when larger; 256^2 takes the 1024^2 time; 4096^2 multiplies by the
pilot's 4096^2 / smoke 1024^2 per-run ratio (>= 1). Every projection is an ESTIMATE x 1.5 in the time limit, enforced by
the scheduler (the reservation is the time limit, not the projection).

    python resources.py --smoke runs/<smoke>/archive/output --pilot runs/<pilot>/archive/output --attempt acc1024
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def rule_m(rule):
    if rule.startswith('gauss'):
        return int(rule[5:]) ** 2
    if rule.startswith('fib'):
        return int(rule[3:])
    raise ValueError(rule)


NCASE = {'all38': 38, 'sub16': 16, 'dev6': 6, 'dev2': 2}


def acc_projection(cfg, smoke_dir, pilot):
    """Measured wall seconds of the smoke (rep['elapsed']: per (config, scheme) including compilation, per setup, per
    block) scaled to the production workload: x (cases / smoke cases) (compilation scaled too: conservative),
    x max(1, m / m_proxy) for the rule, x the pilot's 4096^2 / 1024^2 ratio at 4096^2. Estimates, not bounds."""
    r = json.loads((smoke_dir / 'p1acc_smkacc' / 'result.json').read_text())
    assert r['complete'], 'smoke accuracy job incomplete'
    el = r['elapsed']
    ccfg = {c['name']: c for c in r['config']['configs']}
    meas = {}
    for name, c in ccfg.items():
        for sc in c['schemes']:
            meas[(c['Rp'], c['rule'], sc)] = (el[f'{name}|{sc}'] / NCASE[c['cases']], el[f'{name}|setup'])
    mesh_f = max(1., pilot['run_ratio_4096_over_1024']) if cfg['mesh'] == 4096 else 1.
    total, detail = 0., {}
    for c in cfg['configs']:
        n = NCASE[c['cases']]
        for sc in c['schemes']:
            cand = [(k, v) for k, v in meas.items() if k[0] >= c['Rp'] and k[2] == sc] or \
                   [(k, v) for k, v in meas.items() if k[0] >= c['Rp']]
            k, (t_case, t_setup) = min(cand, key=lambda kv: (kv[0][0], -rule_m(kv[0][1])))
            f = max(1., max(rule_m(c['rule']), rule_m(c.get('qp', c['rule']))) / rule_m(k[1]))
            sec = t_setup * f + n * t_case * f * mesh_f
            detail[f"{c['name']}|{sc}"] = dict(proxy=list(k), factor=f, mesh_factor=mesh_f, seconds=sec)
            total += sec
    for b in cfg.get('blocks', []):
        meas_b = el.get(f"{b['config'] if b['kind'] != 'p1e' else 'R128'}|block|{b['kind']}")
        assert meas_b is not None, b
        scale = 3. * 4. if b['kind'] == 'p1e' else 1.     # smoke p1e: dev2 at R128; production: dev6 at R256 (x3 cases, x4 work)
        detail[f"block|{b['kind']}"] = meas_b * scale
        total += meas_b * scale
    setup = r['phases']['setup'] * mesh_f + 600.
    return (total + setup) / 3600, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='append', required=True, help='smoke archive output dirs (smkA, smkB)')
    ap.add_argument('--pilot', required=True)
    ap.add_argument('--attempt', required=True)
    ap.add_argument('--run', action='append', required=True, help='driver:config exactly as passed to cluster/stage.py')
    a = ap.parse_args()
    smokes = [Path(x) for x in a.smoke]
    find = lambda sub: next(d for d in smokes if (d / sub / 'result.json').exists())
    smoke = find('p1acc_smkacc')
    pil = json.loads((Path(a.pilot) / 'pilot_pilot' / 'result.json').read_text())
    assert pil['complete'], 'pilot incomplete: no certificate (DESIGN A5.2)'
    # comparable measurement (code audit 3): the pilot's own warm recording fused query, R384-gauss128-frozen-CN-R,
    # 4096^2 vs 1024^2 on the same H200, times the H200/A100 ratio is NOT applied (A100 projections stay A100)
    f = {x['mesh']: x['fused_seconds'] for x in pil['runs'] if x.get('Rp') == 384 and x.get('rule') == 'gauss128'
         and x.get('policy') == 'frozen' and x.get('scheme') == 'CNR' and 'fused_seconds' in x}
    assert 4096 in f and 1024 in f and np.isfinite([f[4096], f[1024]]).all() and min(f.values()) > 0, f
    pilot = dict(run_ratio_4096_over_1024=max(1., float(f[4096] / f[1024])), fused_seconds=f)
    cfg = json.loads((HERE / 'configs' / f'{a.attempt}.json').read_text())
    if cfg['driver'] == 'p1acc':
        h, detail = acc_projection(cfg, smoke, pilot)
    elif cfg['driver'] == 'p1time':
        t = json.loads((find('p1time_smktime') / 'p1time_smktime' / 'result.json').read_text())
        assert t['complete']
        per_trip = t['elapsed_seconds'] / max(1, len(t['invocations']))
        n = len(cfg['subjects']) * 6 * cfg['reps']
        h = (n * per_trip * pilot['run_ratio_4096_over_1024'] + 1800.) / 3600
        detail = dict(per_triplet_seconds=per_trip, triplets=n)
    elif cfg['driver'] == 'p1kern':
        k = json.loads((find('p1kern_smkkern') / 'p1kern_smkkern' / 'result.json').read_text())
        assert k['complete']
        h = k['elapsed_seconds'] * (6 / 4) * 3 / 3600 + .1
        detail = dict(smoke_seconds=k['elapsed_seconds'])
    else:
        k = json.loads((find('p1three_smkkern3d') / 'p1three_smkkern3d' / 'result.json').read_text())
        assert k['complete']
        h = k['elapsed_seconds'] * (6 / 2) * 2 / 3600 + .2 if cfg['mode'] == 'kern' else k['setup_seconds'] * 4 / 3600 + .5
        detail = dict(smoke_seconds=k['elapsed_seconds'])
    out = dict(attempt=a.attempt, projection_hours=round(h, 3), time_limit_hours=float(np.ceil(1.5 * h * 4) / 4),
               detail=detail, pilot=pilot, smoke=[str(x) for x in smokes], runs=a.run,
               configs={r.split(':', 1)[1]: __import__('hashlib').sha256((HERE / r.split(':', 1)[1]).read_bytes()).hexdigest()
                        for r in a.run})
    (HERE / 'checks').mkdir(exist_ok=True)
    (HERE / 'checks' / f'resources-{a.attempt}.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != 'detail'}))


if __name__ == '__main__':
    main()
