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
from pathlib import Path

import fomrun2 as FR
import refcore as RC

HERE = Path(__file__).resolve().parent
CPU_PER_RUN = 0.5


def fd_rate(pil, fam, L):
    """Seconds per stage: the slowest of the four tolerance levels' warm dt = 0.005 pilots; compile allowance: the largest."""
    ks = [f'fd|{fam}|L{L}|0.005|t{lev}' for lev in '0123']
    return max(pil[k]['seconds_per_unit'] for k in ks), max(pil[k]['compile_seconds'] for k in ks)


def ps_rate(pil, N):
    k = f'ps|N{N}|0.005'
    return pil[k]['seconds_per_unit'], pil[k]['compile_seconds']


def project(cfg, pil):
    tot, detail = 0., {}
    seen = set()
    for fam, L, dt in cfg.get('fd', []):
        s, comp = fd_rate(pil, fam, L)
        st = RC.n_stages(FR.canon_dt(dt), FR.FAM[fam][1])
        runs = 4 * 4 + 34 + 38 + 2 * 4
        nf = max(1., min(3., FR.canon_dt(dt) / 0.005))          # more Newton iterations per stage at larger steps
        sec = runs * (st * s * nf + CPU_PER_RUN) + (comp if (fam, L) not in seen else 0.)
        seen.add((fam, L))
        detail[f'fd|{fam}|L{L}|{dt}'] = sec
        tot += sec
    for N, dt in cfg.get('ps', []):
        s, comp = ps_rate(pil, N)
        steps = int(round(.25 / FR.canon_dt(dt)))
        sec = (38 + 8) * (steps * s + CPU_PER_RUN) + comp
        detail[f'ps|N{N}|{dt}'] = sec
        tot += sec
    return tot, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', required=True)
    ap.add_argument('--config', required=True)
    a = ap.parse_args()
    res = json.loads((Path(a.smoke) / 'result.json').read_text())
    assert res.get('complete') is True
    cfgp = HERE / a.config
    cfg = json.loads(cfgp.read_text())
    tot, detail = project(cfg, res['pilots'])
    out = dict(attempt=cfg['attempt'], smoke=str(a.smoke), smoke_job=res.get('job_id'), gpu=res.get('gpu'),
               projection_seconds=tot, projection_hours=tot / 3600, required_limit_hours=1.5 * tot / 3600,
               configs={a.config: hashlib.sha256(cfgp.read_bytes()).hexdigest()}, runs=[f'ladder:{a.config}'],
               detail=detail, note='estimates with headroom (x1.5), not demonstrated upper bounds')
    (HERE / 'checks' / f"resources-{cfg['attempt']}.json").write_text(json.dumps(out, indent=1) + '\n')
    print(cfg['attempt'], 'projection h', round(tot / 3600, 3), 'required limit h', round(1.5 * tot / 3600, 3))


if __name__ == '__main__':
    main()
