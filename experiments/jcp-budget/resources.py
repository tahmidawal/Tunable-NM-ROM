"""Smoke-derived cost certificates (DESIGN 9, A2.7; speed3d / qualify resources.py pattern). ESTIMATES, not bounds: the
scheduler time limit is the enforced bound. Writes checks/resources-<attempt>.json with projection_hours and
time_limit_hours = ceil_0.25(1.5 x projection); cluster/stage.py refuses a production attempt whose time limit does not
cover 1.5 x the projection or whose configs / runs differ from the certificate.

Measurements (smk1 2D, smk2 3D; GPU type recorded, H200):
  references   seconds per (dim, horizon) case from the fd step metas (all attempts of a case, incl. retries);
               T = 1.0 projected as 2.2 x the T = 0.5 case time (twice the stages, no refinement, 10 % margin);
               refinement risk: every T = 0.5 case charged x 1.3; PS from the smoke's PS step if present, else
               PS_CASE_H per case (E0: 0.17 h per case at T = 0.25 -> 0.4 h at T = 0.5 incl. escalation reserve)
  b2run        seconds per (member, case) of every phase from the B2 / B7 npz 'seconds' fields and result.json
               elapsed; an unmeasured member takes the slowest measured member (conservative); the full factorial's
               38 corners vs the pairwise 22 scale the corner time linearly; 256^2 = 1024^2 time; 4096^2 x MESH4096
               plus the pilot's measured bank build; + per-r setup (bank, LS, tables) measured
  b3run        seconds per (member, case) from the rollout rows (seconds_first, incl. compile) x cases

    python resources.py --attempt b1024 --run b2run:configs/b1024.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MESH4096 = 2.0
PS_CASE_H = 0.4
NC = {'all38': 38, 'sub16': 16, 'dev6': 6, 'dev2': 2}


def out_dir(att):
    return HERE / 'runs' / att / 'archive' / 'output'


def ncases(c):
    return NC[c] if isinstance(c, str) else len(c)


def ref_case_seconds(att='smk1', dim=2):
    secs = []
    for js in out_dir(att).glob(f'ref_refrun_*/{dim}d/*/T*/c*.json'):
        m = json.loads(js.read_text())
        secs.append((m['horizon'], m['seconds']))
    assert secs, f'no reference timing in {att}'
    return max(s for _, s in secs), secs


def b2_rates(att='smk1'):
    d = next(out_dir(att).glob('b2run_*b1024*'))
    res = json.loads((d / 'result.json').read_text())
    ncase = ncases(res['config']['cases'])
    per_member, corners = {}, None
    for f in d.glob('B2_*.npz'):
        z = np.load(f)
        corners = len(z['corners'])
        per_member[f.stem[3:]] = float(np.nansum(z['seconds'])) / len(z['case_ids'])          # s per case (all corners)
    b7 = {mk: v['seconds'] / v['n'] for mk, v in res['b7'].items()}
    total = res['elapsed_seconds']
    accounted = sum(per_member.values()) * ncase + sum(v * ncase for v in b7.values())
    overhead = max(0., total - accounted)                       # setup, stars, checks, controls, B8, compile
    return dict(corner_case=max(per_member.values()), corners=corners, b7_case=max(b7.values()), overhead=overhead,
                ncase=ncase, members=len(per_member))


def pilot_seconds(att='smk1'):
    d = next(out_dir(att).glob('b2run_*pilot4096*'), None)
    if d is None:
        return 3600.
    return json.loads((d / 'result.json').read_text())['elapsed_seconds']


def project(drv, cfg):
    if drv == 'ref/refrun':
        dim = int(cfg['dim'])
        smk = 'smk1' if dim == 2 else 'smk2'
        case_s, _ = ref_case_seconds(smk, dim)
        if cfg['mode'] == 'fd':
            f = 1.3 if cfg['horizon'] <= .5 else 2.2
            return len(cfg['cases']) * case_s * f + 600
        if cfg['mode'] == 'ps':
            return len(cfg['cases']) * PS_CASE_H * 3600 * (2.2 if cfg['horizon'] > .5 else 1.) + 900
        return 1800.
    if drv == 'b2run':
        r = b2_rates()
        n = ncases(cfg['cases'])
        corners = 38 if cfg['b2'].get('design') == 'full' else 22
        per_corner = r['corner_case'] / r['corners']
        t = len(cfg['members']) * n * per_corner * corners
        if 'b7' in cfg:
            t += len(cfg['members']) * n * r['b7_case']
        t += r['overhead'] * (n / r['ncase']) * (len(cfg['members']) / r['members'])
        if cfg['mesh'] == 4096:
            t = t * MESH4096 + pilot_seconds()
        return t
    if drv == 'b3run':
        d = next(out_dir('smk2').glob('b3run_*'))
        res = json.loads((d / 'result.json').read_text())
        per = sum(r_['seconds_first'] for r_ in res['rows']) / max(1, len(res['config']['case_subset']))
        per *= len(cfg['members']) / len(res['config']['members'])
        return per * len(cfg['case_subset']) + res['elapsed_seconds'] * .2
    raise KeyError(drv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--attempt', required=True)
    ap.add_argument('--run', action='append', required=True)
    a = ap.parse_args()
    tot, det, cfgs = 0., {}, {}
    for r in a.run:
        drv, cf = r.split(':', 1)
        p = HERE / cf
        cfg = json.loads(p.read_text())
        s = project(drv, cfg)
        det[r] = s / 3600
        tot += s
        cfgs[cf] = hashlib.sha256(p.read_bytes()).hexdigest()
    proj = tot / 3600
    tl = math.ceil(1.5 * proj * 4) / 4
    out = dict(attempt=a.attempt, runs=a.run, configs=cfgs, projection_hours=proj, time_limit_hours=max(tl, .5), detail=det,
               label='estimate from the smokes (not a bound)')
    dest = HERE / 'checks' / f'resources-{a.attempt}.json'
    dest.write_text(json.dumps(out, indent=1) + '\n')
    print(dest, json.dumps(out))


if __name__ == '__main__':
    main()
