"""Smoke-derived cost certificates (DESIGN 9.1, A1.11; pattern of jcp-speed3d resources.py). Writes
checks/resources-<attempt>.json with projection_hours; cluster/stage.py requires time limit >= 1.5 x projection.

Measurements: the smoke's p1acc log ("RUN <config> <scheme> <cohort> <case> <elapsed>": per-case wall time of the whole
per-case block - prod + certification + tight + tighter + q+ + q++ + Q3), its configs' setup seconds, the smoke timing
job's per-triplet wall time, the kernel jobs' elapsed time, and the H200 pilot's per-run seconds at 4096^2.
Proxies are always upper proxies: a configuration not measured takes the measured configuration with the next larger
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


def per_case_times(log):
    """{(config, scheme): [per-case seconds]} from consecutive RUN marks (the first case of a block includes compile)."""
    marks = [(m.group(1), m.group(2), float(m.group(5))) for m in
             re.finditer(r'RUN (\S+) (\S+) (\S+) (\d+) ([\d.]+)', log)]
    out, prev = {}, None
    for cfg, sc, t in marks:
        if prev is not None and (cfg, sc) == prev[:2]:
            out.setdefault((cfg, sc), []).append(t - prev[2])
        prev = (cfg, sc, t)
    return out


def acc_projection(cfg, smoke_dir, pilot):
    r = json.loads((smoke_dir / 'p1acc_smkacc' / 'result.json').read_text())
    log = '\n'.join(p.read_text() for p in (smoke_dir.parent / 'logs').glob('*.out'))
    pct = per_case_times(log)
    meas = {}
    for name, c in r['configs'].items():
        for sc in c['schemes']:
            if (name, sc) in pct:
                meas[(c['Rp'], c['rules']['main'], sc)] = (float(np.max(pct[(name, sc)])), c['setup_seconds'])
    mesh_f = 1.
    if cfg['mesh'] == 4096:
        mesh_f = max(1., pilot['run_ratio_4096_over_1024'])
    total, detail = 0., {}
    for c in cfg['configs']:
        n = {'all38': 38, 'sub16': 16, 'dev6': 6, 'dev2': 2}[c['cases']]
        for sc in c['schemes']:
            cand = [(k, v) for k, v in meas.items() if k[0] >= c['Rp'] and k[2] == sc] or \
                   [(k, v) for k, v in meas.items() if k[0] >= c['Rp']]
            (k, (t_case, t_setup)) = min(cand, key=lambda kv: (kv[0][0], -rule_m(kv[0][1])))
            f = max(1., max(rule_m(c['rule']), rule_m(c.get('qp', c['rule']))) / rule_m(k[1]))
            sec = t_setup + n * t_case * f * mesh_f
            detail[f"{c['name']}|{sc}"] = dict(proxy=list(k), factor=f, mesh_factor=mesh_f, seconds=sec)
            total += sec
    blocks = 0.
    for b in cfg.get('blocks', []):
        blocks += 900.                    # P1e / order / vendor blocks: <= 15 min each (smoke measured, see report)
    setup = r['phases']['setup'] * (mesh_f if cfg['mesh'] == 4096 else 1.) + 300.
    return (total + blocks + setup) / 3600, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', required=True)
    ap.add_argument('--pilot', required=True)
    ap.add_argument('--attempt', required=True)
    ap.add_argument('--run', action='append', required=True, help='driver:config exactly as passed to cluster/stage.py')
    a = ap.parse_args()
    smoke = Path(a.smoke)
    pil = json.loads((Path(a.pilot) / 'pilot_pilot' / 'result.json').read_text())
    # comparable measurement (code audit 3): the pilot's own warm recording fused query, R384-gauss128-frozen-CN-R,
    # 4096^2 vs 1024^2 on the same H200, times the H200/A100 ratio is NOT applied (A100 projections stay A100)
    f = {x['mesh']: x['fused_seconds'] for x in pil['runs'] if x.get('Rp') == 384 and x.get('rule') == 'gauss128'
         and x.get('policy') == 'frozen' and x.get('scheme') == 'CNR' and 'fused_seconds' in x}
    pilot = dict(run_ratio_4096_over_1024=max(1., float(f[4096] / f[1024])) if 4096 in f and 1024 in f else 3.,
                 fused_seconds=f)
    cfg = json.loads((HERE / 'configs' / f'{a.attempt}.json').read_text())
    if cfg['driver'] == 'p1acc':
        h, detail = acc_projection(cfg, smoke, pilot)
    elif cfg['driver'] == 'p1time':
        t = json.loads((smoke / 'p1time_smktime' / 'result.json').read_text())
        per_trip = t['elapsed_seconds'] / max(1, len(t['invocations']))
        n = len(cfg['subjects']) * 6 * cfg['reps']
        h = (n * per_trip * pilot['run_ratio_4096_over_1024'] + 1800.) / 3600
        detail = dict(per_triplet_seconds=per_trip, triplets=n)
    elif cfg['driver'] == 'p1kern':
        k = json.loads((smoke / 'p1kern_smkkern' / 'result.json').read_text())
        h = k['elapsed_seconds'] * (6 / 4) * 3 / 3600 + .1
        detail = dict(smoke_seconds=k['elapsed_seconds'])
    else:
        k = json.loads((smoke / 'p1three_smkkern3d' / 'result.json').read_text())
        h = k['elapsed_seconds'] * (6 / 2) * 2 / 3600 + .2 if cfg['mode'] == 'kern' else k['setup_seconds'] * 4 / 3600 + .5
        detail = dict(smoke_seconds=k['elapsed_seconds'])
    out = dict(attempt=a.attempt, projection_hours=round(h, 3), time_limit_hours=float(np.ceil(1.5 * h * 4) / 4),
               detail=detail, pilot=pilot, smoke=str(smoke), runs=a.run,
               configs={r.split(':', 1)[1]: __import__('hashlib').sha256((HERE / r.split(':', 1)[1]).read_bytes()).hexdigest()
                        for r in a.run})
    (HERE / 'checks').mkdir(exist_ok=True)
    (HERE / 'checks' / f'resources-{a.attempt}.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != 'detail'}))


if __name__ == '__main__':
    main()
