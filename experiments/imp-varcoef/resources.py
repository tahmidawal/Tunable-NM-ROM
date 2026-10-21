"""Smoke-derived cost certificates and the budget plan (COMMON-BRIEF §7, DESIGN A1.20, A2.20, A3.20, A7).

    python resources.py cert --smoke runs/<smoke job> --job <job> --run vref:configs/ref100.json [--run ...]
    python resources.py plan --smoke runs/<smoke job> --remaining '<json list>'

Upper-bound models (every per-run cost is the largest observed compile-inclusive time, scaled by work; SAFETY on top):
  vref    per level: (max over smoke cases of seconds / work at that level size) x work of the requested level, work =
          L^2 x stages; the smoke's largest level rate is used for any larger level; x (1 + retry) with retry = max(observed
          retry fraction, 0.5); plus cross-checks at the max smoke cross-check seconds;
  vladder per configuration and contrast: every case (8 dev + 24 val, no pruning), extensions (x (1 + EXT_FRAC)), screening of
          every configuration (2 compiles + 8 full-field calls at MESH_FACTOR), qualification of every configuration at most
          once per contrast (candidate + tight + tighter/3N variants, 32 cases, VARIANT_FACTOR); per-run cost = family rate x
          work, family rate = max over smoke runs of seconds / work (compile-inclusive);
  vtime   per subject: (6 scoring + 6 warm-up + 2 stage-0 calls) x its call cost + stage-1 triplets (6) if shortlisted +
          stage-2 triplets (6 x reps) if selected, triplet = 3 burns + 2 A + B, x RETRY (drift retries), plus the bank build;
          FOM call cost = its ladder screening cost (full-field, from candidates.json); ROM call cost = inherited C2 seconds;
  vtest   16 cases x family rate x work per key.
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
MESH_FACTOR = 3.0
VARIANT_FACTOR = 3.0 * 3          # three variants (candidate, tight, tighter or 3N), each up to 3x the candidate's cost
QUAL_FRAC = 0.5                   # bound on the fraction of configurations tried in qualification (A7.2 cap: <= 8 per group)
RETRY = 1.5
CATEGORIES = ('ref100', 'ref10', 'fallback-reserve', 'ladder', 'timing', 'timing-retry-reserve', 'test-refs', 'test-run')


def J(p):
    return json.loads(Path(p).read_text())


def nstages_fd(dt):
    import vkeys  # noqa: F401  (puts experiments/jcp-fom2d on sys.path)
    import refcore as FC
    return FC.n_stages(dt, 'cnr')


def work(key):
    import vkeys as VK
    sp = VK.parse(key)
    res = sp.get('N') or sp.get('L')
    return float(res) ** 2 * (0.25 / sp['dt'])


def fam(key):
    return key.split('@')[0].split('|')[-1] if key.startswith(('KERNEL|', 'SETUP|')) else key.split('|')[0]


def steps_res(key):
    import vkeys as VK
    sp = VK.parse(key)
    return 0.25 / sp['dt'], float(sp.get('N') or sp.get('L'))


def family_rates(smoke):
    """Per family: the observed per-step times tau_k = steady_k / steps_k (steady = the largest non-first-case time of a
    configuration; variants priced separately) and the largest compile excess c_f. A configuration's per-run cost is
    steps x (the largest per-step time observed at a resolution >= its own; beyond the largest observed resolution,
    quadratic extrapolation from it). Assumption: per-step time is non-decreasing in resolution (GPU latency floor)."""
    lad = J(Path(smoke) / 'output/vladder_smk-ladder/result.json')
    byk = {}
    for r in lad['runs']:
        if r.get('seconds') is not None:
            byk.setdefault(r['key'], []).append(r['seconds'])
    variant = lambda k: any(t in k for t in ('|1e-10', '|1e-12', '|tE', '|tF', '|Q'))   # priced by VARIANT_FACTOR
    out = {}
    for f in ('psi', 'etd', 'fdmg', 'ccv'):
        ks = [k for k in byk if k.split('|')[0] == f and len(byk[k]) >= 2 and not variant(k)]
        assert ks, ('no smoke timing for family', f)
        obs = [(max(byk[k][1:]) / steps_res(k)[0], steps_res(k)[1]) for k in ks]
        c_f = max(max(byk[k][0] - max(byk[k][1:]), 0.0) for k in ks)
        out[f] = (obs, c_f)
    return out


def run_cost(rate, key):
    obs, _ = rate[key.split('|')[0]]
    n, res = steps_res(key)
    above = [t for t, r in obs if r >= res]
    if above:                                   # per-step time is non-decreasing in resolution: bound by observations above
        return n * max(above)
    tmax, rmax = max(((t, r) for t, r in obs), key=lambda x: (x[1], x[0]))
    return n * tmax * (res / rmax) ** 2         # beyond the largest observed resolution: quadratic in res


def proj_vref(smoke, cfg):
    ref = J(Path(smoke) / 'output/vref_smk-ref/result.json')
    rates, retries, n = {}, 0, 0
    Cs = {f'C{float(C):g}' for C in cfg['contrasts']}
    for kc, c in ref['cases'].items():
        if kc.split('|')[0] not in Cs:                 # rates from the same contrast only
            continue
        for lv in c['levels']:
            n += 1
            retries += len(lv) > 1
            L, dt = lv[0]['L'], lv[0]['dt']
            secs = sum(a['seconds'] for a in lv)                    # compile-inclusive (vref.level)
            rates[L] = max(rates.get(L, 0.0), secs / (L * L * nstages_fd(dt)))
    top = rates[max(rates)]
    per_case = 0.0
    for L, dt in cfg['levels']:
        r = rates.get(L, top) if L <= max(rates) else top
        per_case += r * L * L * nstages_fd(dt)
    retry = max(retries / max(n, 1), 0.1)          # observed retry fraction (fallback reruns are reserved separately)
    xc = [sum(s['seconds'] for s in x['stats'].values()) for x in ref['crosscheck'].values()]
    assert xc, 'smoke has no cross-check timing'
    nc = len(cfg['cases']) * len(cfg['contrasts'])
    nx = len(cfg.get('crosscheck', [])) * len(cfg['contrasts'])
    return (per_case * (1 + retry) * nc + max(xc) * nx) / 3600.0


def structural(key):
    """Compiled-program identity (A7.1): the key without the traced knob (tol / kbar mode / CCv level)."""
    p = key.split('|')
    f = p[0]
    if f in ('psi', 'fdmg'):
        return '|'.join(p[:4] if f == 'psi' else p[:4])
    if f == 'etd':
        return '|'.join(p[:3])
    return '|'.join(p[:3])


def proj_vladder(smoke, cfg):
    rate = family_rates(smoke)
    nC = len(cfg['contrasts'])
    tot = 0.0
    progs = {structural(k): k.split('|')[0] for k in cfg['keys']}
    for s_, f in progs.items():                       # compiles: lattice + full-field + (spectral) Q = 3N, x extensions
        tot += (1 + EXT_FRAC) * rate[f][1] * (2 + (1 if f in ('psi', 'etd') else 0))
    dev_only = cfg.get('dev_only', {})
    nfull = sum(1 for C in cfg['contrasts'] if not dev_only.get(f'{float(C):g}'))
    ndev = nC - nfull
    for k in cfg['keys']:
        t = run_cost(rate, k)
        # full contrast: 32 runs (no pruning) with extensions; screening = 8 full-field calls; qualification at most once
        # per configuration (three variants x 32 runs, each up to 3x) for a QUAL_FRAC share; dev-only contrast: 8 runs
        tot += (1 + EXT_FRAC) * nfull * (32 * t + 8 * MESH_FACTOR * t) + QUAL_FRAC * nfull * 32 * t * VARIANT_FACTOR
        tot += (1 + EXT_FRAC) * ndev * 8 * t + nC * t        # + the T-kappa / temporal-check runs (one per key bound)
    return tot / 3600.0


def proj_vtime(smoke, cfg, cands):
    tim = J(Path(smoke) / 'output/vtime_smk-time/result.json')
    burn = float(tim['config']['burn'])
    tA = float(np.median([.5 * (t['tA1'] + t['tA2']) for t in tim['stage2']]))
    build = float(tim['memory']['bank'].get('build_seconds', 0.0))
    reps = int(cfg['reps'])
    roms = {r['label']: r for r in cfg['roms']}
    rb = {r['label']: r for r in J(HERE / 'checks/rom-box.json')['members']}
    shortlisted = {k for g in cfg['groups'] for k in g['candidates']}
    tot = build
    for lb in list(roms) + list(cfg['foms']):
        if lb in roms:
            c = rb[lb]['seconds']
        else:
            base = lb.split('|', 1)[1] if lb.startswith(('KERNEL|', 'SETUP|')) else lb
            base = base.replace('@C1', '@C100') if base.endswith('@C1') else base
            c = cands['candidates'][base]['screening'] * MESH_FACTOR
        trip = 3 * burn + 2 * tA + c
        tot += 14 * (c + burn) + (6 * trip if lb in shortlisted else 0) + 6 * reps * trip
    return tot * RETRY / 3600.0


def proj_vtest(smoke, cfg):
    rate = family_rates(smoke)
    return sum(rate[k.split('@')[0].split('|')[0]][1] + 16 * run_cost(rate, k.split('@')[0]) for k in cfg['keys']) / 3600.0


def project(smoke, r):
    drv, cf = r.split(':', 1)
    cfg = J(HERE / cf)
    if drv == 'vtime':
        return proj_vtime(smoke, cfg, J(HERE / cfg['bindings']['candidates']))
    return dict(vref=proj_vref, vladder=proj_vladder, vtest=proj_vtest)[drv](smoke, cfg)


def cert(smoke, job, runs):
    parts = {r: project(smoke, r) * SAFETY for r in runs}
    cfgs = {r.split(':', 1)[1]: hashlib.sha256((HERE / r.split(':', 1)[1]).read_bytes()).hexdigest() for r in runs}
    tim = J(Path(smoke) / 'output/vtime_smk-time/result.json')
    out = dict(job=job, runs=runs, configs=cfgs, parts_hours=parts, projection_hours=sum(parts.values()), safety=SAFETY,
               smoke=str(smoke), gpu=tim.get('gpu'), note='smoke-derived upper-bound certificate (resources.py docstring)')
    (HERE / 'checks' / f'resources-{job}.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))
    return out


def plan(smoke, remaining):
    """remaining: list of {category, job, runs | hours (+ basis) | deferred (+ reason)}; every CATEGORY must appear."""
    led = J(HERE / 'runs/GPU-LEDGER.json')
    spent = sum((j.get('gpu_hours') or 0.0) if j.get('final') else j['time_limit_hours'] for j in led['jobs'])
    seen = {it['category'] for it in remaining}
    missing = sorted(set(CATEGORIES) - seen)
    assert not missing, ('budget plan lacks categories', missing)
    items = []
    for it in remaining:
        if it.get('deferred'):
            items.append(dict(it, hours=0.0))
        elif 'hours' in it:
            items.append(dict(it, hours=float(it['hours'])))
        else:
            items.append(dict(it, hours=sum(project(smoke, r) * SAFETY for r in it['runs']), basis='certificate'))
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
