"""jcp-speed3d resource certificates (DESIGN A1.12): project the wall time of a production config from the cluster smoke's
measured per-run times (same GPU type, H200) and write checks/resources-<attempt>.json, which cluster/stage.py verifies.

Grid (g1-g3): per setting and spec of the plan, per-case run time, certification time and compile estimate are taken from
the smoke grid rows of the same setting and spec when present; otherwise from the closest measured proxy, always an upper
bound: W256 <- M2_512, W512 / W768 <- W1024; a reduced rule <- the same setting's deployed rule; a missing step <- the
measured step of the same (kind, scheme) with the largest step count scaled by the step-count ratio. Timing panel:
subjects x cases x reps x (two A runs + one B run + three burn-ins), plus one warm call per subject and case. ADT tighter
(8 cases) and the replay (the FS-K2 CN 2.5 run cost per failing case, x 15 interventions) are added. FOM (f1): per
configuration the smoke's measured seconds per solve of the same family and mesh (scaled by stage count), times
(4 calibration cases x 3 levels + 64 cases + warm cases x reps + warm-up). Total x 1.5 must fit the allocation.

    python resources.py --smoke runs/smoke1/archive/output --config g1 --hours 24 [--gpu h200]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROXY = {'W256': 'M2_512', 'W512': 'W1024', 'W768': 'W1024', 'M2_512': 'M2_512', 'W1024': 'W1024'}


def steps_of(dt):
    return int(round(.25 / dt))


def grid_table(res):
    """{(setting, spec): dict(run, cert, compile)} from the smoke grid rows (deployed rules only)."""
    by = {}
    dep = {sn: next(iter(sv['rules'])) for sn, sv in res['settings'].items()}
    for r in res['rows']:
        if r['rule'] != dep.get(r['setting']) or r.get('path', 'mesh') != 'mesh':
            continue
        by.setdefault((r['setting'], r['spec']), []).append(r)
    out = {}
    for k, rs in by.items():
        t = [x['seconds_first'] for x in rs]
        out[k] = dict(run=float(np.median(t[1:])) if len(t) > 1 else t[0], compile=max(t[0] - (float(np.median(t[1:])) if len(t) > 1 else 0), 0.),
                      cert=float(np.median([x.get('cert_seconds', 0.) for x in rs])))
    return out


def lookup(tab, setting, spec):
    import s3common as S
    sp = S.parse_key(spec)
    prox = PROXY[setting]
    if (prox, spec) in tab:
        return tab[(prox, spec)], 'measured' if prox == setting else f'proxy {prox}'
    cand = [(k, v) for k, v in tab.items() if k[0] == prox and k[1].split('|')[0] == spec.split('|')[0]
            and k[1].split('|')[1] == spec.split('|')[1]]
    if not cand:
        cand = [(k, v) for k, v in tab.items() if k[0] == prox and k[1].split('|')[0] == spec.split('|')[0]]
    if not cand:
        cand = [(k, v) for k, v in tab.items() if k[0] == prox]
    k, v = max(cand, key=lambda kv: steps_of(S.parse_key(kv[0][1])[2]))
    f = max(steps_of(sp[2]) / steps_of(S.parse_key(k[1])[2]), 1.)
    return dict(run=v['run'] * f, cert=v['cert'] * f, compile=v['compile']), f'scaled from {k[0]}|{k[1]} x{f:.2f}'


def project_grid(cfg, smoke):
    import s3run as SR
    import s3common as S
    res = json.loads((smoke / 'smkgrid' / 'result.json').read_text())
    tab = grid_table(res)
    nc = len(cfg.get('case_subset') or range(cfg['cohort_count']))
    tc = cfg['timing']
    burn = tc['burn']
    total, detail = 0., {}
    cold = tab[('M2_512', 'fs|BE|0.01|K1')]['run']
    for s in cfg['settings']:
        sec, nsub, sub_t = 0., 0, 0.
        for rule in [s['rule']] + s.get('reduced', []):
            adt, sv = SR.grid_plan(s, rule, cfg)
            for sp in adt + sv:
                t, how = lookup(tab, s['name'], S.spec_key(sp))
                sec += t['compile'] + nc * (t['run'] + (t['cert'] if sp[0] != 'ad' else 0.))
                if sp[0] == 'ad' and rule == s['rule']:
                    sec += 8 * t['run']                              # tighter on cases 0-7
                if sp[0] != 'ad' and len(sp) <= 4:
                    nsub += 1
                    sub_t += t['run']
                if S.spec_key(sp) == 'fs|CN|0.025|K2' and rule == s['rule']:
                    sec += 0.25 * nc * 15 * t['run']                 # replay: up to a quarter of the cases, 15 interventions
        panel = tc['cases'] * tc['reps'] * (nsub * 2 * cold + sub_t + 3 * nsub * burn) + tc['cases'] * (sub_t + cold)
        detail[s['name']] = dict(rollouts_hours=sec / 3600, panel_hours=panel / 3600)
        total += sec + panel
    return total / 3600, detail


def project_fom(cfg, smoke):
    import fomcore as FC
    import fomrun as FR
    res = json.loads((smoke / 'smkfom' / 'result.json').read_text())
    per = {}
    for k, v in res['fom']['fd'].items():
        fam, L, dt = k.split('|')[0], int(k.split('|')[1][1:]), float(k.split('|')[2])
        s = [c['seconds_first'] for c in v.get('cases', {}).values()][1:]
        per[(fam, L)] = (float(np.median(s)) if s else None, FC.n_stages(dt, FR.FAM[fam][1]))
    ps = {}
    for k, v in res['fom']['ps'].items():
        s = [c['seconds_first'] for c in v['cases'].values()][1:]
        ps[int(k.split('|')[1][1:])] = (float(np.median(s)), steps_of(float(k.split('|')[2])))
    b = cfg['fom']
    tc = cfg['timing']
    nrun = len(b['calib_cases']) * 3 + cfg['cohort_count'] + tc['cases'] * (tc['reps'] + 1)
    total, detail = 0., {}
    for fam, L, dts in b['fd']:
        base = per.get((fam, L)) or per.get(('UB', L)) or max(((v, kk) for kk, v in per.items() if kk[1] <= L),
                                                             key=lambda z: z[1][1])[0]
        if base[0] is None:
            base = max(v for v in per.values() if v[0] is not None)
        scale_L = (L / max(kk[1] for kk in per if per[kk][0] is not None)) ** 3 if (fam, L) not in per else 1.
        for dt in dts:
            t = base[0] * FC.n_stages(dt, FR.FAM[fam][1]) / base[1] * scale_L
            total += nrun * t
            detail[f'{fam}|L{L}|{dt:.8g}'] = nrun * t / 3600
    Nm = max(ps)
    for N in b['ps']['Ns']:
        for dt in b['ps']['dts']:
            t = ps[Nm][0] * (N / Nm) ** 3 * steps_of(dt) / ps[Nm][1] * max(1., 1.)
            total += (cfg['cohort_count'] + tc['cases'] * (tc['reps'] + 1)) * t
            detail[f'ps|N{N}|{dt:.8g}'] = (cfg['cohort_count'] + tc['cases'] * (tc['reps'] + 1)) * t / 3600
    return total / 3600, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--hours', type=int, required=True)
    ap.add_argument('--gpu', default='h200')
    a = ap.parse_args()
    smoke = Path(a.smoke)
    cfgp = HERE / 'configs' / f'{a.config}.json'
    cfg = json.loads(cfgp.read_text())
    acc = json.loads((smoke / 'ACCEPTANCE.json').read_text())
    assert acc['verdict'] == 'PASS', 'the smoke must be accepted before any certificate'
    if cfg['mode'] == 'grid':
        h, det = project_grid(cfg, smoke)
    elif cfg['mode'] == 'fom':
        h, det = project_fom(cfg, smoke)
    else:
        raise SystemExit('certificates for prof / mesh / rep are written when their configs exist')
    setup = 0.5
    out = dict(attempt=a.config, config_sha256=hashlib.sha256(cfgp.read_bytes()).hexdigest(), gpu=a.gpu,
               hours_allocated=a.hours, projected_hours=h + setup, **{'projected_hours_x1.5': 1.5 * (h + setup)},
               detail=det, smoke=str(smoke), smoke_acceptance=acc['verdict'],
               smoke_job=json.loads((smoke / 'smkgrid' / 'result.json').read_text()).get('job_id'))
    (HERE / 'checks' / f'resources-{a.config}.json').write_text(json.dumps(out, indent=1) + '\n')
    print(a.config, f"projected {h + setup:.2f} h, x1.5 = {1.5 * (h + setup):.2f} h of {a.hours}")


if __name__ == '__main__':
    main()
