"""jcp-fom2d resource certificates (DESIGN §7, A1.6, A2.6): project a production ladder job's wall time from the ladder
smoke's measured pilots (same GPU type) and write checks/resources-<attempt>.json, which cluster/stage.py verifies.

FD: per (family, L) the smoke pilot gives the warm seconds per stage of one dt = 0.005 query at each of the four levels;
s(L) = the largest (applied to every level and step: per-stage Newton work varies with dt, hence the x 1.5 headroom). A group (family, L, dt) costs
  calibration 4 cases x 4 levels + production 34 cases + one recovery rerun of 38 cases (worst case, A1.1)
  + warm 2 cases x (1 + 3) calls, each stages(dt) x s(L), plus the compile allowance (pilot first - warm) once per
  (family, L), plus 0.5 s of CPU scoring/saving per case run.
PS: per N, seconds per step from the warm dt = 0.005 pilot; a configuration costs (38 + 8) x steps(dt) x s(N) plus the
compile allowance per configuration (dt is static in the PS program) plus 0.5 s per case run.
Total x 1.5 must fit the allocation (stage.py asserts hours >= 1.5 x projection). ESTIMATES with headroom, not bounds.

    python resources.py --smoke runs/smk1/archive/output/ladder_smk1 --config configs/fcc.json
"""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
from pathlib import Path

import fomrun2 as FR
import refcore as RC

HERE = Path(__file__).resolve().parent
CPU_PER_RUN = 0.5


def fd_rate(res, fam, L):
    """Seconds per stage: the slowest of the four tolerance levels' warm dt = 0.005 pilots; compile: the solver and the
    lattice map as logged when they compiled; prep: the pilots' measured input preparation."""
    pil, log = res['pilots'], res['compile_log']
    ks = [f'fd|{fam}|L{L}|0.005|t{lev}' for lev in '0123']
    adv = FR.FAM[fam][0]
    comp = log[f'fd|{adv}|L{L}|lim20'] + log.get(f'map|L{L}|256', 0.)
    return (max(pil[k]['seconds_per_unit'] for k in ks), comp, max(pil[k]['warm']['prep_seconds'] for k in ks))


def ps_rate(res, N):
    pil, log = res['pilots'], res['compile_log']
    k = f'ps|N{N}|0.005'
    return pil[k]['seconds_per_unit'], log[f'ps|N{N}|0.005|lattice'], pil[k]['warm']['prep_seconds']


def project(cfg, res):
    tot, detail = 0., {}
    seen = set()
    burn = float(cfg.get('burn', 0.1)) * (len(cfg.get('warm_cases', [])) * int(cfg.get('warm_reps', 3)) + 1)
    for fam, L, dt in cfg.get('fd', []):
        s, comp, prep = fd_rate(res, fam, L)
        st = RC.n_stages(FR.canon_dt(dt), FR.FAM[fam][1])
        runs = 4 * 4 + 34 + 38 + 2 * 4
        nf = max(1., min(3., FR.canon_dt(dt) / 0.005))          # more Newton iterations per stage at larger steps
        sec = runs * (st * s * nf + CPU_PER_RUN + prep) + burn + (comp if (fam, L) not in seen else 0.)
        seen.add((fam, L))
        detail[f'fd|{fam}|L{L}|{dt}'] = sec
        tot += sec
    for N, dt in cfg.get('ps', []):
        s, comp, prep = ps_rate(res, N)
        steps = int(round(.25 / FR.canon_dt(dt)))
        sec = (38 + 8) * (steps * s + CPU_PER_RUN + prep) + burn + comp
        detail[f'ps|N{N}|{dt}'] = sec
        tot += sec
    return tot, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', required=True)
    ap.add_argument('--config', required=True)
    a = ap.parse_args()
    res = json.loads((Path(a.smoke) / 'result.json').read_text())
    assert res.get('complete') is True and res.get('accepted') is True, 'certificates need an accepted smoke'
    cfgp = HERE / a.config
    cfg = json.loads(cfgp.read_text())
    tot, detail = project(cfg, res)
    out = dict(attempt=cfg['attempt'], smoke=str(a.smoke), smoke_job=res.get('job_id'), gpu=res.get('gpu'),
               projection_seconds=tot, projection_hours=tot / 3600, required_limit_hours=1.5 * tot / 3600,
               configs={a.config: hashlib.sha256(cfgp.read_bytes()).hexdigest()}, runs=[f'ladder:{a.config}'],
               detail=detail, note='estimates with headroom (x1.5), not demonstrated upper bounds')
    (HERE / 'checks' / f"resources-{cfg['attempt']}.json").write_text(json.dumps(out, indent=1) + '\n')
    print(cfg['attempt'], 'projection h', round(tot / 3600, 3), 'required limit h', round(1.5 * tot / 3600, 3))


if __name__ == '__main__' and '--c2' not in __import__('sys').argv:
    main()


# ------------------------------------------------------------------------------------------------- C2 jobs -----

def project_c2(cfg, smkc2, match, ladder_res):
    """C2 job projection (DESIGN A1.6): per FOM candidate a full-field call costs max(2 x its lattice screening cost,
    screening cost + 0.02 s) (the C2 smoke measured full-field / lattice ratios of 1.03-2.0 at 4096^2); calls per candidate:
    6 scoring + 6 warm + 4 stage-0 (+ 6 stage-1 triplets if shortlisted, + 18 stage-2 triplets if selected; assumed for
    every candidate, an upper bound); ROM calls: A's measured smoke time x max(1, (r/384)^2); per triplet two A calls and
    three 0.1 s burns; compile 3 s per PS program and the smoke's logged FD/map compile per mesh; bank build and table set-up
    from the smoke's elapsed time (x number of ROM settings / 2). x 1.5 headroom applied by stage.py."""
    tA = float(np.median([x['tA1'] for x in smkc2['stage2']]))
    burn = 3 * float(cfg.get('burn', 0.1))
    tot, det = 0., {}
    for k in cfg['foms']:
        c = match['admissibility'][k]['cost']
        t = max(2 * c, c + 0.02)
        sec = 16 * t + 24 * (t + 2 * tA + burn) + (3. if k.startswith('ps') else 2.)
        det[k] = sec
        tot += sec
    for r in cfg['roms'] + [cfg['A']]:
        t = tA * max(1., (int(r['Rp']) / 384) ** 2)
        sec = 12 * t + 18 * (t + 2 * tA + burn) + 10.
        det[r['label']] = sec
        tot += sec
    tot += float(smkc2['elapsed_seconds']) * max(1., len(cfg['roms']) / 2)
    tot += 6 * 3 * (3 * tA + burn) + 20 * len(cfg.get('pilot_foms', []))     # controls and pilots
    return tot, det


def main_c2():
    ap = argparse.ArgumentParser()
    ap.add_argument('--c2', action='store_true')
    ap.add_argument('--smkc2', required=True)
    ap.add_argument('--match', required=True)
    ap.add_argument('--config', required=True)
    a = ap.parse_args()
    sm = json.loads((Path(a.smkc2) / 'result.json').read_text())
    assert sm.get('complete') is True and sm['gates']['G-RR'] and sm['gates']['G-FR'] and sm.get('pilots_ok'), 'C2 smoke not usable'
    M = json.loads((HERE / a.match).read_text())
    cfgp = HERE / a.config
    cfg = json.loads(cfgp.read_text())
    tot, det = project_c2(cfg, sm, M, None)
    out = dict(attempt=cfg['attempt'], smoke=str(a.smkc2), smoke_job=sm.get('job_id'), gpu=sm.get('gpu'),
               projection_seconds=tot, projection_hours=tot / 3600, required_limit_hours=1.5 * tot / 3600,
               configs={a.config: hashlib.sha256(cfgp.read_bytes()).hexdigest()}, runs=[f'c2time:{a.config}'], detail=det,
               note='estimates with headroom (x1.5), not demonstrated upper bounds')
    (HERE / 'checks' / f"resources-{cfg['attempt']}.json").write_text(json.dumps(out, indent=1) + '\n')
    print(cfg['attempt'], 'projection h', round(tot / 3600, 3), 'required limit h', round(1.5 * tot / 3600, 3))


if __name__ == '__main__' and '--c2' in __import__('sys').argv:
    main_c2()
