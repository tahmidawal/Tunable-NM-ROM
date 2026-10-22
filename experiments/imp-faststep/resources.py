"""Smoke-derived cost certificate (COMMON-BRIEF 7, A1.10, A2.8, code audit 7): prices a production config's explicit
inventory from a measured smoke result.json on the SAME mesh and GPU type (otherwise refused).

  * per (arm, r) of the target: smoke seconds per case of that (arm, r); a target rank absent from the smoke is priced
    at the most expensive measured rank of that arm (conservative); an arm absent from the smoke is refused;
  * dev6-only work inside the arm loop is included in the smoke per-case cost; it is scaled by cases (conservative);
  * rank setup (Setting + tables), order and classification blocks: measured per rank, scaled by 6 / smoke dev cases;
    controls: measured per rank (refused if the target wants controls the smoke did not run);
  * POD: smoke POD seconds x target cases / smoke cases (refused if the smoke had no POD block);
  * timing: smoke timing seconds x (target subjects / smoke subjects) x (6 / smoke dev cases) x (reps ratio);
  * bank setup once. A 1.5x safety factor; the config's time limit must cover the projection.

    python resources.py <smoke result.json> <production config> --out checks/resources-<job>.json
"""
import json
import sys

import numpy as np
from pathlib import Path

SAFETY = 1.5
ARMS = ('L-CNR', 'G-CN', 'G-E4a', 'G-E4b', 'G-R1', 'L-CNR-M1152')


def main():
    R = json.loads(Path(sys.argv[1]).read_text())
    cfg = json.loads(Path(sys.argv[2]).read_text())
    out = Path(sys.argv[sys.argv.index('--out') + 1])
    sc = R['config']
    if sc['mesh'] != cfg['mesh']:
        sys.exit('refused: smoke mesh differs from the target mesh')
    n_s = len(R['setup']['cases'])
    nd_s = sum(1 for c in R['setup']['cases'] if c.startswith('dev6'))
    n_p = 38 if cfg['cases'] == 'all38' else len(cfg['cases'])
    el = R['elapsed']
    if 'H200' not in R['gpu']:
        sys.exit('refused: the smoke did not run on an H200 (production GPU)')
    sec, inv = dict(setup=el.get('setup', 0.)), []           # includes the bank build (measured from job start)
    ranks_s = [int(r['Rp']) for r in sc['ranks']]
    # per (arm, r): fixed cost (compilation, first case) and marginal per-case cost from the smoke log's RUN timestamps
    # (code audit 7/8): marginal = mean of the later cases' increments; with a single smoke case the marginal is priced
    # at one third of that case's full cost (which includes compilation; conservative).
    logs = sorted((Path(sys.argv[1]).parents[2] / 'logs').glob('*.out'))
    stamps = {}
    for line in logs[0].read_text().splitlines() if logs else []:
        f_ = line.split()
        if f_ and f_[0] == 'RUN':
            stamps.setdefault((f_[1], int(f_[2])), []).append(float(f_[-1]))
    arm_cost = {}
    for k, v in el.items():
        if '|' in k and k.split('|')[0] in ARMS:
            key = (k.split('|')[0], int(k.split('|')[1]))
            ts = stamps.get(key, [])
            inc = np.diff(ts)
            marg = float(np.mean(inc)) if len(inc) else v / 3.
            arm_cost[key] = (max(v - marg * (n_s - 1), 0.), marg)
    sec['arms'] = 0.
    for spec in cfg['ranks']:
        rp = int(spec['Rp'])
        for arm in cfg['arms']:
            if arm == 'L-CNR-M1152' and rp != 384:
                continue
            meas = [arm_cost[(arm, r)] for r in ranks_s if (arm, r) in arm_cost]
            if not meas:
                sys.exit(f'refused: arm {arm} not measured in the smoke')
            fx, mg = arm_cost.get((arm, rp), max(meas, key=lambda t: t[0] + 38 * t[1]))
            c = fx + mg * (n_p - 1)
            sec['arms'] += c
            sec.setdefault('classification_allowance', 0.)
            sec['classification_allowance'] += 2 * 24 * mg + fx  # up to two tripped runs per (arm, r): dt/8 + dt/16 refinements
            inv.append((arm, rp, round(c, 1), 'measured' if (arm, rp) in arm_cost else 'priced at the costliest measured rank'))
    per_rank = lambda pre: [v for k, v in el.items() if k.startswith(pre)]
    rank_over = max([v - sum(el.get(f'{a}|{k.split("|")[1]}', 0.) for a in ARMS) - el.get(f'controls|{k.split("|")[1]}', 0.)
                     for k, v in el.items() if k.startswith('rank|')] + [0.])     # Setting, tables, order, classification
    sec['rank_setup_and_blocks'] = len(cfg['ranks']) * rank_over * (6 / max(nd_s, 1))
    if cfg.get('controls'):
        cm = per_rank('controls|')
        if not cm:
            sys.exit('refused: target runs controls the smoke did not measure')
        sec['controls'] = len(cfg['ranks']) * max(cm)
    if cfg.get('pod'):
        if 'pod' not in el:
            sys.exit('refused: target runs the POD block the smoke did not measure')
        # POD members: per-member fixed (shape compilation) + marginal per case; the marginal is taken from the members of
        # the second time step (already compiled shapes): their duration / smoke cases (code audit 8)
        pts = [float(l.split()[-1]) for l in logs[0].read_text().splitlines() if l.startswith(('POD', 'NATIVE'))]
        durs = np.diff(pts) if len(pts) > 1 else np.array([el['pod']])
        marg_case = float(np.median(durs)) / n_s
        sec['pod'] = el['pod'] + marg_case * len(pts) * (n_p - n_s)
    if cfg.get('timing'):
        subj_s = len((R.get('timing') or {}).get('screen', {})) or 1
        subj_p = sum(len([a for a in cfg['arms'] if a != 'L-CNR-M1152' or int(s['Rp']) == 384]) for s in cfg['ranks'])
        if cfg.get('pod'):
            subj_p += sum(1 for k in (R.get('timing') or {}).get('screen', {}) if not k.split('|')[0] in ARMS)
        sec['timing'] = el.get('timing', 0.) * max(1., subj_p / subj_s) * (6 / max(nd_s, 1)) * \
            (cfg['timing']['reps'] / sc['timing']['reps'])
    total_h = sum(sec.values()) / 3600.
    cert = dict(smoke=sys.argv[1], smoke_gpu=R['gpu'], config=sys.argv[2], seconds=sec, inventory=inv,
                projection_hours=SAFETY * total_h, safety=SAFETY, time_limit_hours=cfg['time_limit_hours'],
                covered=bool(cfg['time_limit_hours'] - 4 / 60 >= SAFETY * total_h))   # driver timeout = limit - 4 min
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cert, indent=1) + '\n')
    print(json.dumps(dict(projection_hours=cert['projection_hours'], covered=cert['covered'], seconds=sec)))


if __name__ == '__main__':
    main()
