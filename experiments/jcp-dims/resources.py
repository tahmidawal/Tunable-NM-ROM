"""jcp-dims smoke-derived cost certificates (DESIGN 8, A2.6; pattern of P1 / qualify resources.py, rewritten for the
lane's drivers). ESTIMATES, not bounds: the scheduler time limit (>= 1.5 x projection, rounded up to 0.25 h) is the
enforced bound; cluster/stage.py refuses a production attempt whose limit is below 1.5 x the certificate's projection or
whose configs differ from the certified ones.

Work model: the per-case cost of a 2D arm / 3D rung-spec scales with w = m * M * r (Jacobian formation, the dominant
term of DESIGN 5), floored by the smallest measured per-case cost of the same driver. Measurements (smokes, all H200):
  smk2  (dacc)  rep['elapsed']['<setting>|<arm>'] / n_cases     -> per-case cost incl. tight, tighter and rho work
                rep['elapsed']['<setting>|setup']                -> per-setting table build (max taken)
  smk2t (dtime) rep['blocks'][*]['seconds'] / (subjects x cases x (1 + reps))   -> per subject-case-rep unit
  smk3  (d3run) rep['blocks'][*] ladder / adaptive seconds / (rungs x specs x cases); panel seconds / invocations
The heaviest measured unit of each kind is the reference (t_ref, w_ref); a production unit costs
max(t_floor, t_ref * w / w_ref) — linear scaling from the heaviest point, conservative for heavier-than-smoke cells.

    python resources.py --attempt f2a --run dacc:configs/f2a.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
NC = {'all38': 38, 'sub16': 16, 'dev6': 6, 'dev2': 2}


def out_dir(att, drv, stem):
    d = HERE / 'runs' / att / 'archive' / 'output' / f'{drv}_{stem}'
    assert (d / 'result.json').exists(), f'missing smoke output {d}'
    return d


SMOL = {2: {8: 1025, 9: 2305, 10: 5121, 11: 11265, 12: 24577, 13: 53249, 14: 114689},
        3: {8: 2559, 9: 6271, 10: 15103, 11: 35839, 12: 83967, 13: 194559}}


def m_of(rule, dim=2):
    import re
    f, n = re.match(r'^([a-z]+)(\d+)', rule).groups()
    n = int(n)
    if f in ('gauss',):
        return n * n
    if f == 'gl':
        return n ** 3
    if f in ('smolyak', 'smol'):             # 2D and 3D Smolyak sizes (DESIGN 2.2, A1.7)
        return SMOL[dim][n]
    return n


def ncases(c):
    return NC[c] if isinstance(c, str) else len(c)


# ------------------------------------------------------------------------------------------------ 2D accuracy
def ref_dacc():
    r = json.loads((out_dir('smk2', 'dacc', 'smk2') / 'result.json').read_text())
    assert r['complete'], 'smoke incomplete'
    pts, setups = [], []
    for s in r['config']['settings']:
        for a in s['arms']:
            k = f"{s['name']}|{a['form']}_{a['scheme']}_{float(a['factor']):g}"
            t = r['elapsed'][k] / ncases(a['cases'])
            pts.append((m_of(s['rule']) * s['M'] * s['Rp'], t, a['form']))
        setups.append(r['elapsed'].get(f"{s['name']}|setup", 0.))
    return pts, max(setups)


def proj_dacc(cfg):
    pts, setup = ref_dacc()
    floor = min(t for _, t, _ in pts)
    tot = 0.
    for s in cfg['settings']:
        tot += setup
        w = m_of(s['rule']) * s['M'] * s['Rp']
        for a in s['arms']:
            same = [p for p in pts if p[2] == a['form']] or pts
            wr, tr, _ = max(same)
            tot += ncases(a['cases']) * max(floor, tr * w / wr)
    for blk in cfg.get('blocks', []):
        tot += 120. if blk['kind'] == 'fixgate' else 600.
    return tot


# ------------------------------------------------------------------------------------------------ 2D timing
def proj_dtime(cfg):
    r = json.loads((out_dir('smk2t', 'dtime', 'smk2t') / 'result.json').read_text())
    assert r['complete'], 'smoke incomplete'
    nc_s, reps_s = ncases(r['config']['cases']), int(r['config']['reps'])
    units = []
    for b in r['blocks']:
        subs = [x for x in r['config']['subjects'] if f"{x['mesh']}|R{x['Rp']}|M{x.get('M')}" in b['subjects'][0]] or r['config']['subjects']
        x = subs[0]
        units.append((m_of(x['rule']) * int(x.get('M', 4 * x['Rp'])) * int(x['Rp']),
                      b['seconds'] / (len(b['subjects']) * nc_s * (1 + reps_s))))
    floor = min(t for _, t in units)
    wr, tr = max(units)
    nc, reps = ncases(cfg['cases']), int(cfg['reps'])
    tot = 300.
    for x in [cfg['A']] + cfg['subjects']:
        w = m_of(x['rule']) * int(x.get('M', 4 * x['Rp'])) * int(x['Rp'])
        tot += nc * (1 + reps) * max(floor, tr * w / wr)
    return tot


# ------------------------------------------------------------------------------------------------ 3D
def ref_d3():
    r = json.loads((out_dir('smk3', 'd3run', 'smk3') / 'result.json').read_text())
    assert r['complete'], 'smoke incomplete'
    cfg = r['config']
    sets = {s['name']: s for s in cfg['settings']}
    lad, pan = [], []
    for b, rb in zip(cfg['blocks'], r['blocks']):
        if b['kind'] in ('ladder', 'adaptive'):
            s = sets[b['setting']]
            M = int(s.get('M') or 4 * s['Rp'])
            nsp = len(b['specs']) if b['kind'] == 'ladder' else len(b['forms'])
            nc = len(b.get('cases') or range(cfg['cohort_count']))
            w = max(m_of(x, 3) for x in b['rungs']) * M * s['Rp']
            lad.append((w, rb['seconds'] / (len(b['rungs']) * nsp * nc), b['kind']))
        elif b['kind'] == 'panel':
            n = len(r['timing'][b['label']]['invocations'])
            pan.append(rb['seconds'] / max(n, 1))
        elif b['kind'] == 'rounds':
            pan_r = rb['seconds'] / max(1, cfg['cohort_count'])
            lad.append((1., pan_r, 'rounds'))
    return lad, max(pan) if pan else 1., r['peak_bytes_job']


def proj_d3(cfg):
    lad, t_inv, _ = ref_d3()
    floor = min(t for _, t, _ in lad)
    sets = {s['name']: s for s in cfg['settings']}
    tot = 600.
    for b in cfg['blocks']:
        if b['kind'] in ('ladder', 'adaptive'):
            s = sets[b['setting']]
            M = int(s.get('M') or 4 * s['Rp'])
            nsp = len(b['specs']) if b['kind'] == 'ladder' else len(b['forms'])
            nc = len(b.get('cases') or range(cfg['cohort_count']))
            same = [p for p in lad if p[2] == b['kind']] or lad
            wr, tr, _ = max(same)
            for rl in b['rungs']:
                tot += nsp * nc * max(floor, tr * m_of(rl, 3) * M * s['Rp'] / wr)
        elif b['kind'] == 'rounds':
            same = [p for p in lad if p[2] == 'ladder']
            wr, tr, _ = max(same)
            for m in b['members']:
                st, rl = m['key'].split('|')[:2]
                s = sets[st]
                w = (m_of(rl, 3) + m_of(m['qp'], 3)) * 4 * s['Rp'] * s['Rp']
                tot += cfg['cohort_count'] * max(floor, tr * w / wr) * 1.2
        elif b['kind'] == 'panel':
            tot += len(b['subjects']) * int(b['cases']) * int(b['reps']) * t_inv * 3
        else:
            tot += 300.
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--attempt', required=True)
    ap.add_argument('--run', action='append', required=True)
    a = ap.parse_args()
    det, total = {}, 0.
    for r_ in a.run:
        drv, cf = r_.split(':', 1)
        cfg = json.loads((HERE / cf).read_text())
        sec = {'dacc': proj_dacc, 'dtime': proj_dtime, 'd3run': proj_d3}[drv](cfg)
        det[cf] = sec / 3600.
        total += sec / 3600.
    tl = math.ceil(1.5 * total * 4) / 4
    cert = dict(attempt=a.attempt, runs=a.run, projection_hours=round(total, 4), time_limit_hours=tl, detail=det,
                configs={r_.split(':', 1)[1]: hashlib.sha256((HERE / r_.split(':', 1)[1]).read_bytes()).hexdigest() for r_ in a.run},
                note='estimate (linear w = m M r scaling from the heaviest smoke unit, floored); not a bound')
    (HERE / 'checks').mkdir(exist_ok=True)
    (HERE / 'checks' / f'resources-{a.attempt}.json').write_text(json.dumps(cert, indent=1) + '\n')
    print(json.dumps(cert))


if __name__ == '__main__':
    main()
