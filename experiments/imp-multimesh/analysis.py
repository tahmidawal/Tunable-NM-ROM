"""Production input adapter (DESIGN 5, A2.2, A2.3, A3.1, A3.4, A4.1, A5.2, A7.1): builds the cost.py data model from
the pulled run JSONs, with the complete registered universe, the arm aliases (PM-s = CP-s at s; CPU-1024 at 256 =
CPR-1024 at 256), canonical artefact keys, the identity / timing acceptance gates, and the accounting variants.

    python analysis.py <runs dir> <out.json>
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np

import cost as C

OURS_X = [f'{R}|{sc}' for R in ('R128', 'R256', 'R384') for sc in ('BE', 'CNR')]
SRC_STEPS = {256: 'all', 1024: '12'}


def jl(p):
    return json.loads(Path(p).read_text())


def find(runs, rel):
    hits = sorted(Path(runs).glob(f'*/pulled/output/{rel}'))
    assert len(hits) == 1, (rel, [str(h) for h in hits])
    return hits[0]


def probe_numbers(runs):
    """J2 numbers from runs/j2/pulled/output/probe_numbers.json (written once from J2's logs by extract_probe.py,
    because J2 emits its training and head timers only in logs)."""
    d = jl(Path(runs) / 'j2' / 'pulled' / 'output' / 'probe_numbers.json')
    d['a100'] = jl(Path(runs).parent / 'inputs' / 'historical.json')['records']
    return d


def probe_from_logs(runs):
    """J2 (A2.2, A3.2, A4.2): phase-wise H200 bank fit, data generation, head (complete invocations)."""
    d = Path(runs) / 'j2' / 'pulled'
    log = (d / 'output' / 'r3a.log').read_text()
    job = next((d / 'logs').glob('*.out')).read_text()

    def step(n):
        m = re.search(rf'step +{n}/25000 .*\[(\d+)s\]', log)
        return float(m.group(1))
    gen = float(re.search(r'gen: trajectories 576/576 .*\[(\d+)s\]', log).group(1))
    t1, t5, t15, t20, t25 = (step(n) for n in (1, 5000, 15000, 20000, 25000))
    tdone = float(re.search(r'done: recon.*\[(\d+)s\]', log).group(1))
    rs, rf = (t15 - t5) / 10000, (t25 - t20) / 5000
    tau = (t20 - t15) - 5000 * rf
    fit = t1 + 289999 * rs + tau + 10000 * rf + (tdone - t25)
    ex = float(re.search(r'extract_wall=([0-9.]+)', job).group(1))
    hf = float(re.search(r'hfit_wall=([0-9.]+)', job).group(1))
    return dict(rho_sub=rs, rho_full=rf, tau=tau, t1=t1, t_done=tdone - t25, gen=gen, bankfit=fit, extract=ex, hfit=hf,
                head=ex + hf,
                sources=dict(h200=str(d / 'output' / 'r3a.log') + ' + ' + str(next((d / 'logs').glob('*.out')))))


def ours_inputs(runs, timing, probe):
    """ours[X]: status / errors per mesh (J5 oursq), query time / setup / compile per mesh (J6 A1+A2)."""
    ours = {X: dict(status={}, err={}, t={}, setup={}, compile={}, median={}) for X in OURS_X}
    eps = {}
    for L in C.MESHES:
        try:
            q = jl(find(runs, f'ours{L}/oursq.json'))
        except AssertionError:
            continue
        for X in OURS_X:
            c = q['configs'].get(X)
            if c is None:
                continue
            ours[X]['status'][L] = c['status']
            ours[X]['err'][L] = {f"{p['cohort']}|{p['case']}": p['e'] for p in c['per_case'] if p['eligible']}
            for p in c['per_case']:
                eps[f"{p['cohort']}|{p['case']}"] = p['eps_ref']
    for L in C.MESHES:
        M = timing['meshes'].get(str(L), {})
        att = [M[k] for k in sorted(M) if k.startswith('attempt') and M[k].get('valid')]
        if not att:
            for X in OURS_X:
                if ours[X]['status'].get(L) == 'QUALIFIED':
                    ours[X]['status'][L] = 'UNRESOLVED'          # no valid timing panel -> cost unresolved
            continue
        a = att[-1]
        for X in OURS_X:
            r1, r2 = a['A1'].get(X), a['A2'].get(X)
            if not (r1 and r2 and r1['identity']['ok'] and r2['identity']['ok']):
                ours[X]['status'][L] = 'UNRESOLVED'
                continue
            ours[X]['t'][L] = float(np.mean(r1['obs'] + r2['obs']))
            ours[X]['median'][L] = float(np.median(r1['obs'] + r2['obs']))
            ours[X]['setup'][L] = a['A1']['bank_seconds'] + a['A1']['ops_seconds'] + r1['setup_seconds']
            ci = r1.get('compile_info') or {}
            ours[X]['compile'][L] = None if ci.get('cache_hit_suspected', True) else r1['compile_seconds']
    return ours, eps


def comp_inputs(runs, timing):
    """comp[arm][L][cfg] with states, errors, timing and the artefact keys each configuration needs."""
    src = jl(find(runs, 'sources.json'))
    tg = {L: jl(find(runs, f'target{L}.json')) for L in C.MESHES}
    art = {}
    for s in (256, 1024):
        I = src[f'src{s}']
        art[f'snap[{s}]'] = I['snap_seconds']
        art[f'pod[{s}]'] = I['pod_seconds']
        art[f'codes[{s}]'] = I['codes_seconds']
        art[f'states[{s}]'] = I['states_seconds']
        for r, e in I['ecsw'].items():
            if 'seconds' in e:
                art[f'ecsw[{s}@{s},r{r}]'] = e['seconds']
    for L, T in tg.items():
        if 'states' in T:
            art[f'states[{L}]'] = T['states']['seconds']
        for name, A in T['arms'].items():
            s = A['source']
            if s != L:                     # an artefact measured by two arms: the larger measurement (results audit 1)
                k1, k2 = f'resample[{s}->{L}]', f'trust[{s}->{L}]'
                art[k1] = max(art.get(k1, 0.), A['timers'].get('resample', float('nan')))
                art[k2] = max(art.get(k2, 0.), A['timers'].get('codes', 0.))
            for r, e in A.get('ecsw', {}).items():
                if 'contrib_seconds' in e:
                    art[f'ecsw[{s}@{L},r{r}]'] = e['contrib_seconds'] + e['nnls_seconds']
                elif 'map_seconds' in e:
                    art[f'map[{s}->{L},r{r}]'] = e['map_seconds']

    def needs(kind, s, L, r):
        if s == L:
            return [f'snap[{s}]', f'pod[{s}]', f'states[{s}]', f'ecsw[{s}@{s},r{r}]']
        base = [f'snap[{s}]', f'pod[{s}]', f'codes[{s}]', f'resample[{s}->{L}]', f'trust[{s}->{L}]']
        if kind == 'reuse' and L > s:
            return base + [f'states[{s}]', f'ecsw[{s}@{s},r{r}]', f'map[{s}->{L},r{r}]']
        return base + [f'states[{L}]', f'ecsw[{s}@{L},r{r}]']

    # physical records: (target mesh, arm name) -> configs
    phys = {(L, n): A for L, T in tg.items() for n, A in T['arms'].items()}
    alias = {}
    for arm in C.CP_ARMS:
        kind = 'refit' if arm.startswith('CPR') else 'reuse'
        s = int(arm.split('-')[1])
        for L in C.MESHES:
            if L == s:
                alias[(arm, L)] = (f'PM-{s}', 'pm')
            elif kind == 'reuse' and L < s:
                alias[(arm, L)] = (f'CPR-{s}', 'refit')
            else:
                alias[(arm, L)] = (arm, kind)
    comp = {}
    for (arm, L), (pname, kind) in alias.items():
        s = int(arm.split('-')[1])
        A = phys.get((L, pname))
        Tm = timing['meshes'].get(str(L), {})
        att = [Tm[k] for k in sorted(Tm) if k.startswith('attempt') and Tm[k].get('valid')]
        tarm = att[-1]['arms'].get(pname) if att else None
        out = {}
        for cid in C.REG_CFGS:
            r = int(cid.split('|')[0][1:])
            rec = dict(state='UNRESOLVED', err={}, t=float('nan'), compile=None, armdata=0., needs=[], rollout_ok=None)
            c = None if A is None else A['configs'].get(cid)
            if c is not None and c['status'] in ('MEASURED-Q1', 'MEASURED-FAIL', 'NOT-CONSTRUCTIBLE'):
                rec['state'] = c['status']
                if c['status'] != 'NOT-CONSTRUCTIBLE':
                    rec['err'] = {f"{x['cohort']}|{x['case']}": x['e'] for x in c['rows'] if x['eligible']}
                    rec['rollout_ok'] = (c.get('rollout') or {}).get('ok')
                    rec['needs'] = needs(kind, s, L, r)
                    rec['run_seconds'] = c.get('run_seconds')
                    tc = None if tarm is None else tarm['configs'].get(cid)
                    if c['status'] == 'MEASURED-Q1':
                        if tc is None or not tc['identity']['ok']:
                            rec['state'] = 'UNRESOLVED'               # no valid timing -> cost unresolved
                        else:
                            rec['t'] = tc['mean']
                            rec['t_median'] = tc['median']
                            rec['armdata'] = tc['armdata_seconds']
                            ci = tc.get('compile_info') or {}
                            rec['compile'] = None if ci.get('cache_hit_suspected', True) else tc['compile_seconds']
            out[cid] = rec
        comp.setdefault(arm, {})[L] = out
    return comp, art, src, tg


def compile_max(timing, comp, ours, fresh):
    vals = [v for X in ours.values() for v in X['compile'].values() if v is not None]
    vals += [r['compile'] for a in comp.values() for L in a.values() for r in L.values() if r.get('compile') is not None]
    vals += [f['seconds'] for f in fresh]
    return float(max(vals)) if vals else float('nan')


def build(runs):
    timing = dict(meshes={})
    for tp in sorted(Path(runs).glob('*/pulled/output/time.json')):     # later jobs supply meshes the first lacked
        for L_, m in jl(tp)['meshes'].items():
            if any(v.get('valid') for k, v in m.items() if k.startswith('attempt')) or L_ not in timing['meshes']:
                timing['meshes'][L_] = m
    probe = probe_numbers(runs)
    ours, eps = ours_inputs(runs, timing, probe)
    comp, art, src, tg = comp_inputs(runs, timing)
    fresh = [jl(p) for p in sorted(Path(runs).glob('*/pulled/output/compile/*.json'))]
    offline = dict(A_dep=probe['gen'] + probe['bankfit'] + probe['head'],
                   A_nohead=probe['gen'] + probe['bankfit'],
                   A_100=probe['a100']['gen'] + probe['a100']['bankfit'] + probe['a100']['head'])
    D = dict(ours=ours, comp=comp, art=art, eps=eps, ours_offline=offline)
    D['compile_max'] = compile_max(timing, comp, ours, fresh)
    C.complete_universe(D)
    return D, dict(probe=probe, fresh=fresh, src=src, targets=tg, timing=timing)


def main():
    runs, outp = sys.argv[1], sys.argv[2]
    D, aux = build(runs)
    res = dict(inputs=dict(offline=D['ours_offline'], art=D['art'], compile_max=D['compile_max'], probe=aux['probe']),
               verdicts={})
    for verdict in C.VERDICTS:
        for var in ('A_dep', 'A_100', 'A_nohead'):
            res['verdicts'][f'{verdict}|{var}'] = C.stop_rule(D, verdict, var)
    res['breakeven'] = {}
    for k in (1, 2, 3):
        import itertools
        for ms in itertools.combinations(C.MESHES, k):
            for X in D['ours']:
                if all(C.status(D, X, L) == 'QUALIFIED' for L in ms):
                    b = C.breakeven(D, X, ms)
                    res['breakeven'][f"{'+'.join(map(str, ms))}|{X}"] = dict(crossings=b['crossings'],
                                                                           sign_at_1=b['diff'][0], sign_at_end=b['diff'][-1])
    # A_sel (A5.2, descriptive): every competitor target-job configuration run, our J5 accuracy job
    sel = sum((x.get('run_seconds') or 0.) + (x.get('armdata_seconds') or 0.) for T in aux['targets'].values()
              for A in T['arms'].values() for x in A['configs'].values())
    res['A_sel'] = dict(competitor_runs_seconds=sel, note='descriptive; mixed with historical A100 head arms in the report')
    C.dump(outp, dict(res, D=D))
    print('stop rule (worst, A_dep):', res['verdicts']['worst|A_dep']['outcome'])


if __name__ == '__main__':
    main()
