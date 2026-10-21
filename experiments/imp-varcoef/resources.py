"""Smoke-derived cost certificates and the budget plan (COMMON-BRIEF §7, DESIGN A1.20, A2.20, A3.20).

    python resources.py cert --smoke runs/<smoke job> --job <job> --run vref:configs/ref100.json [--run ...]
    python resources.py plan --smoke runs/<smoke job> --remaining '<json list of {job, runs}>'

Projection models (conservative; safety factor SAFETY on every projection):
  vref    max over smoke case-contrasts of the per-case solve seconds (all three levels, retries, compile) x cases,
          plus the max smoke cross-check seconds x cross-check cases;
  vladder per configuration: compile (max smoke compile excess of its family) + runs x per-run seconds, per-run seconds
          = (max over the family's smoke configurations of seconds / work) x work, work = res^2 x steps; runs = 8 dev +
          24 val (no pruning assumed) per contrast, x (1 + EXT_FRAC) for extensions; screening = SCREEN_FRAC of the
          configurations x (2 compiles + 8 full-field calls at MESH_FACTOR x the lattice per-run seconds); qualification =
          36 candidates x 3 variants x 32 runs per contrast;
  vtime   smoke elapsed x (production subjects x reps) / (smoke subjects x reps), plus the smoke bank build;
  vtest   like vladder for its keys on 16 cases.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SAFETY = 1.3
EXT_FRAC = 0.5
SCREEN_FRAC = 0.5
MESH_FACTOR = 3.0
QUAL = 36 * 3


def J(p):
    return json.loads(Path(p).read_text())


def work(key):
    import vkeys as VK
    sp = VK.parse(key)
    res = sp.get('N') or sp.get('L')
    return float(res) ** 2 * (0.25 / sp['dt'])


def fam(key):
    return key.split('|')[0]


def ladder_model(smoke):
    lad = J(Path(smoke) / 'output/vladder_smk-ladder/result.json')
    per, comp = {}, {}
    byk = {}
    for r in lad['runs']:
        if 'seconds' in r:
            byk.setdefault(r['key'], []).append(r['seconds'])
    for k, ts in byk.items():
        steady = min(ts)
        per.setdefault(fam(k), []).append(steady / work(k))
        comp.setdefault(fam(k), []).append(max(ts) - steady)
    return {f: max(v) for f, v in per.items()}, {f: max(comp[f]) for f in comp}


def proj_vref(smoke, cfg):
    ref = J(Path(smoke) / 'output/vref_smk-ref/result.json')
    per_case = [sum(l['seconds'] + l['compile_seconds'] for lv in c['levels'] for l in lv) for c in ref['cases'].values()]
    xc = [sum(s['seconds'] for s in x['stats'].values()) for x in ref['crosscheck'].values()] or [0.0]
    n = len(cfg['cases']) * len(cfg['contrasts'])
    nx = len(cfg.get('crosscheck', [])) * len(cfg['contrasts'])
    return (max(per_case) * n + max(xc) * nx) / 3600.0


def proj_vladder(smoke, cfg):
    per, comp = ladder_model(smoke)
    tot = 0.0
    nC = len(cfg['contrasts'])
    for k in cfg['keys']:
        f = fam(k)
        t = per[f] * work(k)
        tot += comp[f] + nC * (32 * t) * (1 + EXT_FRAC)
        tot += SCREEN_FRAC * nC * (2 * comp[f] + 8 * MESH_FACTOR * t)
    tq = np.mean([per[fam(k)] * work(k) for k in cfg['keys']])
    tot += nC * QUAL * 32 * tq
    return tot / 3600.0


def proj_vtime(smoke, cfg):
    tim = J(Path(smoke) / 'output/vtime_smk-time/result.json')
    sm = J(HERE / 'configs/smk-time.json')
    ns = len(sm['foms']) + len(sm['roms']) + 1
    npd = len(cfg['foms']) + len(cfg['roms']) + 1
    build = float(tim['memory']['bank'].get('build_seconds', 0.0))
    return (build + (tim['elapsed_seconds'] - build) * (npd * cfg['reps']) / (ns * sm['reps'])) / 3600.0


def proj_vtest(smoke, cfg):
    per, comp = ladder_model(smoke)
    return sum(comp[fam(k)] + 16 * per[fam(k)] * work(k.split('@')[0]) for k in cfg['keys']) / 3600.0


PROJ = dict(vref=proj_vref, vladder=proj_vladder, vtime=proj_vtime, vtest=proj_vtest)


def cert(smoke, job, runs):
    tot, parts, cfgs = 0.0, {}, {}
    for r in runs:
        drv, cf = r.split(':', 1)
        cfg = J(HERE / cf)
        h = PROJ[drv](smoke, cfg) * SAFETY
        parts[r] = h
        tot += h
        cfgs[cf] = hashlib.sha256((HERE / cf).read_bytes()).hexdigest()
    tim = J(Path(smoke) / 'output/vtime_smk-time/result.json')
    out = dict(job=job, runs=runs, configs=cfgs, parts_hours=parts, projection_hours=tot, safety=SAFETY,
               smoke=str(smoke), gpu=tim.get('gpu'), note='smoke-derived certificate (resources.py docstring)')
    (HERE / 'checks' / f'resources-{job}.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))
    return out


def plan(smoke, remaining):
    led = J(HERE / 'runs/GPU-LEDGER.json')
    spent = sum((j.get('gpu_hours') or 0.0) if j.get('final') else j['time_limit_hours'] for j in led['jobs'])
    items = []
    for it in remaining:
        if 'hours' in it:                       # reserve without a config yet (fallback, test-once)
            items.append(dict(job=it['job'], hours=float(it['hours']), basis=it.get('basis', 'reserve')))
            continue
        h = sum(PROJ[r.split(':', 1)[0]](smoke, J(HERE / r.split(':', 1)[1])) * SAFETY for r in it['runs'])
        items.append(dict(job=it['job'], hours=h, basis='certificate'))
    tot = spent + sum(i['hours'] for i in items)
    out = dict(spent_committed=spent, remaining=items, spent_plus_remaining=tot, phase_cap=8.0, fits=bool(tot <= 8.0))
    (HERE / 'checks/budget-plan.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['cert', 'plan'])
    ap.add_argument('--smoke', required=True)
    ap.add_argument('--job')
    ap.add_argument('--run', action='append', default=[])
    ap.add_argument('--remaining', default='[]')
    a = ap.parse_args()
    if a.cmd == 'cert':
        cert(a.smoke, a.job, a.run)
    else:
        plan(a.smoke, json.loads(a.remaining))
