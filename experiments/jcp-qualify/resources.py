"""jcp-qualify smoke-derived cost certificates (DESIGN 8; P1 resources.py pattern). Writes checks/resources-<attempt>.json
with projection_hours and time_limit_hours = ceil_0.25(1.5 x projection); cluster/stage.py requires the time limit to
cover 1.5 x projection, the configs and runs to match. ESTIMATES, not bounds: the scheduler limit is the enforced bound.

Measurements: smkA (qacc) rep['elapsed'] per setting|arm (wall seconds incl. compilation, over its cases) and per
block; smkB (qtime) elapsed and triplet count; smk3 (q3run) elapsed per member (log 'MEMBER' lines) and setup; the
4096^2 pilot run seconds. Proxies (conservative): an arm not measured takes the measured arm of the same
(form, scheme, factor) at the next larger measured r (R256, R384 -> R512); 256^2 takes the 1024^2 time; 4096^2
multiplies by MESH4096 = 2.0 (P1 ledger: its acc4096 per-rollout time did not exceed acc1024's); case counts scale
linearly (compilation scaled too: conservative). Q2 / tighter / Q3 extras are inside the measured per-case blocks.

    python resources.py --attempt acc1024 --run qacc:configs/acc1024.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import analyze as AN

HERE = Path(__file__).resolve().parent
MESH4096 = 2.0
NC = {'all38': 38, 'sub16': 16, 'dev6': 6, 'dev2': 2}


def ncases(c):
    return NC[c] if isinstance(c, str) else len(c)


def smoke_acc():
    d = AN.out_dir('smkA', 'qacc')
    r = json.loads((d / 'result.json').read_text())
    assert r['complete'], 'smoke incomplete: no certificate'
    per = {}
    for s in r['config']['settings']:
        for arm in s['arms']:
            k = f"{s['name']}|{arm['form']}_{arm['scheme']}_{float(arm['factor']):g}"
            per[(s['Rp'], arm['form'], arm['scheme'], float(arm['factor']))] = r['elapsed'][k] / ncases(arm['cases'])
    return r, per


def acc_projection(cfg):
    r, per = smoke_acc()
    mesh_f = MESH4096 if cfg['mesh'] == 4096 else 1.
    total, det = 0., {}
    for s in cfg['settings']:
        for arm in s['arms']:
            key = (arm['form'], arm['scheme'], float(arm['factor']))
            cand = sorted(rr for (rr, *k) in per if tuple(k) == key and rr >= s['Rp'])
            assert cand, ('no measured proxy', s['Rp'], key)
            t_case = per[(cand[0],) + key]
            sec = ncases(arm['cases']) * t_case * mesh_f
            det[f"{s['name']}|{key}"] = dict(proxy_r=cand[0], seconds=sec)
            total += sec
        total += 120. * mesh_f                     # setting setup (tables, compile of certifiers)
    for b in cfg.get('blocks', []):
        if b['kind'] == 'order':
            sm = r['elapsed'].get('R384|block|order')
            ncase_smoke = 6 + 3                    # smoke: TH06 on dev6 + three one-case arms
            n_prod = sum(ncases(a.get('cases', b['cases'])) for a in b['arms'])
            sec = sm / ncase_smoke * n_prod
            det[f"order|{b['setting']}"] = sec
            total += sec
    setup = r['phases']['setup'] * mesh_f + 900.
    return (total + setup) / 3600, det


def time_projection(cfg):
    d = AN.out_dir('smkB', 'qtime')
    t = json.loads((d / 'result.json').read_text())
    assert t['complete']
    _, per = smoke_acc()
    mesh = [int(m) for m in cfg['meshes'] if int(m) != 1024] or [1024]
    mesh_f = MESH4096 if mesh[0] == 4096 else 1.
    per_trip = t['elapsed_seconds'] / max(1, len(t['invocations']))      # includes certification of the smoke subjects
    n = len(cfg['subjects']) * 6 * cfg['reps']
    cert = 0.
    for sp in cfg['subjects']:
        key = (sp['form'], sp['scheme'], float(sp['factor']))
        cand = sorted(rr for (rr, *k) in per if tuple(k) == key and rr >= sp['Rp'])
        cert += 6 * per[(cand[0],) + key] * mesh_f
    sec = n * per_trip * mesh_f + cert + 1800.
    return sec / 3600, dict(per_triplet_seconds=per_trip, triplets=n, certification_seconds=cert)


def q3_projection(cfg):
    d = AN.out_dir('smk3', 'q3run')
    r = json.loads((d / 'result.json').read_text())
    assert r['complete']
    n_smoke = len(r['members']) + (1 if r.get('control') else 0)
    per_member = r['elapsed_seconds'] / max(1, n_smoke)
    n = len(cfg['members']) + (1 if cfg.get('control') else 0)
    return (n * per_member * 1.5 + 1200.) / 3600, dict(per_member_seconds=per_member, members=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--attempt', required=True)
    ap.add_argument('--run', action='append', required=True)
    a = ap.parse_args()
    h, det = 0., {}
    for run in a.run:
        drv, cf = run.split(':', 1)
        cfg = json.loads((HERE / cf).read_text())
        fn = dict(qacc=acc_projection, qtime=time_projection, q3run=q3_projection)[drv]
        hh, dd = fn(cfg)
        h += hh
        det[run] = dict(hours=hh, detail=dd)
    out = dict(attempt=a.attempt, projection_hours=round(h, 3), time_limit_hours=math.ceil(1.5 * h * 4) / 4, detail=det,
               runs=a.run, configs={r.split(':', 1)[1]: hashlib.sha256((HERE / r.split(':', 1)[1]).read_bytes()).hexdigest()
                                    for r in a.run})
    (HERE / 'checks' / f'resources-{a.attempt}.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'detail'}))


if __name__ == '__main__':
    main()
