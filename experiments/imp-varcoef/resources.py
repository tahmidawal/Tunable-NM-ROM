"""Smoke-derived cost certificates and the budget plan (COMMON-BRIEF §7, DESIGN A1.20, A2.20, A3.20, A7).

    python resources.py cert --smoke runs/<smoke job> --job <job> --run vref:configs/ref100.json [--run ...]
    python resources.py plan --smoke runs/<smoke job> --remaining '<json list>'

Upper-bound models (every per-run cost is the largest observed compile-inclusive time, scaled by work; SAFETY on top):
  vref    per level: (max over smoke cases of seconds / work at that level size) x work of the requested level, work =
          L^2 x stages at the same contrast; the largest observed rate for unobserved levels; x (1 + retry) with retry =
          max(observed retry fraction, 0.1) (fallback reruns are a separate reserve); plus cross-checks at the max smoke
          cross-check seconds;
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
    top = max(rates.values())                         # the largest observed rate, for unobserved levels
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


def possible_extensions(keys):
    """Every configuration the A2.21/A3.10/A4.10/A5.10/A6.1 extension rules could add (superset): dt/2 and dt/4 of every
    row's finest step, and the cheap-edge step of every configuration on an edge."""
    import vkeys as VK
    sps = [VK.parse(k) for k in keys]
    new = set()
    rows = {}
    for sp in sps:
        r = dict(sp)
        r.pop('dt')
        rows.setdefault(json.dumps(r, sort_keys=True), []).append(sp)
    for r, v in rows.items():
        sp = min(v, key=lambda x: x['dt'])
        for d in (2, 4):
            new.add(VK.key(dict(sp, dt=sp['dt'] / d)))
    for f in {sp['f'] for sp in sps}:
        fs = [sp for sp in sps if sp['f'] == f]
        rk = 'N' if 'N' in fs[0] else 'L'
        for sp in fs:
            if sp[rk] == min(x[rk] for x in fs):
                new.add(VK.key(dict(sp, **{rk: 12 if rk == 'N' else 32})))
            if sp['dt'] == max(x['dt'] for x in fs):
                new.add(VK.key(dict(sp, dt=0.05)))
            if 'tol' in sp and sp['tol'] == max(x['tol'] for x in fs):
                new.add(VK.key(dict(sp, tol=1e-2)))
            if 'level' in sp and sp['level'] == max((x['level'] for x in fs), key=int):
                new.add(VK.key(dict(sp, level='4')))
    return sorted(new - set(keys))


def proj_vladder_split(smoke, cfg):
    """A8.1 two-job ladder. L1 (stop_after_val): lattice programs + dev runs (base + every possible extension) at every
    contrast + val runs of every configuration at full contrasts. L2 (resume_from): from the L1 runs, the configurations
    matching some target (valid on all pooled cases, pooled eligible worst <= max tau): full-field compile + 8 screening
    calls + burns each; qualification per (family, tau) of the 8 MOST expensive matching configurations (an upper bound on
    the 8 attempts), three variants x 32 runs at up to 3x, + Q = 3N compiles; + reruns of the temporal/T-kappa checks."""
    rate = family_rates(smoke)
    full = [C for C in cfg['contrasts'] if not cfg.get('dev_only', {}).get(f'{float(C):g}')]
    nC = len(cfg['contrasts'])
    allk = list(cfg['keys']) + possible_extensions(cfg['keys'])
    if cfg.get('stop_after_val'):
        tot = sum(rate[f][1] for f in {structural(k): k.split('|')[0] for k in allk}.values())
        tot += sum(run_cost(rate, k) * (8 * nC + 24 * len(full) + nC) for k in allk)
        return tot / 3600.0
    prev = J(HERE / cfg['resume_from_local'])
    burn = float(cfg['screen']['burn']) * int(cfg['screen']['reps']) * 2
    tot = 0.0
    for C in full:
        runs = [r for r in prev['runs'] if float(r['C']) == float(C)]
        byk = {}
        for r in runs:
            byk.setdefault(r['key'], []).append(r)
        match = {}
        for k, rr in byk.items():
            if len(rr) == 32 and all(r['valid'] for r in rr):
                el = [r['e'] for r in rr if r['eligible']]
                if el and max(el) <= max(cfg['taus']):
                    match[k] = max(el)
        # per-run cost from the L1 production measurements themselves; compiles are priced separately (rate[f][1])
        meas = {k: (sorted(r.get('seconds') or 0.0 for r in rr)[-2] if len(rr) > 1 else (rr[0].get('seconds') or 0.0))
                for k, rr in byk.items()}           # second-largest measured run time (the largest may hold a compile)
        cost_l1 = lambda k: meas.get(k, 0.0)
        mprog = {structural(k): k.split('|')[0] for k in match}
        tot += sum(rate[f][1] for f in mprog.values())
        tot += sum(8 * MESH_FACTOR * cost_l1(k) + burn for k in match)
        qual = []                                   # per-group bounds summed WITHOUT de-duplication (audit codex-a7-2 item 8)
        for f in ('psi', 'etd', 'fdmg', 'ccv'):
            for tau in cfg['taus']:
                ks = sorted((k for k in match if k.split('|')[0] == f and match[k] <= tau), key=lambda k: -cost_l1(k))
                qual += ks[:int(cfg.get('max_tries', 8))]
        tot += sum(32 * cost_l1(k) * VARIANT_FACTOR for k in qual)
        tot += sum(rate[k.split('|')[0]][1] for k in qual if k.split('|')[0] in ('psi', 'etd'))   # Q = 3N programs
        top = max(cost_l1(k) for k in byk)
        tot += 4 * 4 * top + 8 * top                                                              # temporal / T-kappa reruns
    return tot / 3600.0


def proj_vladder(smoke, cfg):
    if cfg.get('stop_after_val') or cfg.get('resume_from'):
        return proj_vladder_split(smoke, cfg)
    rate = family_rates(smoke)
    nC = len(cfg['contrasts'])
    tot = 0.0
    burn = float(cfg.get('screen', {}).get('burn', 0.1)) * int(cfg.get('screen', {}).get('reps', 3)) * 2
    allk = list(cfg['keys']) + possible_extensions(cfg['keys'])
    progs = {structural(k): k.split('|')[0] for k in allk}
    for s_, f in progs.items():                       # compiles: lattice + full-field + (spectral) Q = 3N, x extensions
        tot += rate[f][1] * (2 + (1 if f in ('psi', 'etd') else 0))
    dev_only = cfg.get('dev_only', {})
    nfull = sum(1 for C in cfg['contrasts'] if not dev_only.get(f'{float(C):g}'))
    ndev = nC - nfull
    for k in allk:
        t = run_cost(rate, k)
        # full contrast: 32 runs (no pruning); screening = 8 full-field calls + burns; dev-only contrast: 8 runs;
        # + T-kappa / temporal-check runs (one per configuration bound)
        tot += nfull * (32 * t + 8 * MESH_FACTOR * t + burn) + ndev * 8 * t + nC * t
    # qualification: each configuration at most once per contrast; per family at most 8 attempts x 3 targets, bounded by
    # the 24 most expensive configurations of the family (three variants x 32 runs, each up to 3x)
    for f in ('psi', 'etd', 'fdmg', 'ccv'):
        ts = sorted((run_cost(rate, k) for k in allk if k.split('|')[0] == f), reverse=True)[:24]
        tot += nfull * sum(32 * t * VARIANT_FACTOR for t in ts)
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
