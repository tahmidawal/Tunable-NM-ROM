"""jcp-heads analysis (DESIGN 5 as amended; CPU only) -> runs/EXPORT.json. Reads only collected archives.

    python analyze.py rules       rule1024 -> runs/RULES.json (frozen rule per arm group; A1.4, A2.5)
    python analyze.py all         everything available -> runs/EXPORT.json

Names: heads H<K>s<seed> (deployed M = 4K), H<K>s<seed>m128 (fixed-M panel), the incumbent Hinc; linear L<r>, L<r>m128.
Every cost number carries a `contract` field. Every label is exploratory validation evidence (A4.1).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

import hscore as HSC
import hstats as HS
import nphead as NH
import p1stats as PS

HERE = Path(__file__).resolve().parent
RUNS = HERE / 'runs'
KS, SEEDS, LIN = (4, 8, 16, 32), (0, 1, 2), (4, 8, 16, 32, 64, 128, 256, 384)
SCHEMES = ('BE', 'CNR')
INC = HERE.parent / 'separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'
CONTRACT = ('online query, full-field contract: supplied field sampled at the initial-fit points, initial fit, evolution, '
            'full-field decode of the six outputs; A-normalised (A = linear r = 384, 1024^2, LSPG-CN-R, dt0, gauss128)')


def out_dir(att, drv, stem=None):
    return RUNS / att / 'archive' / 'output' / f'{drv}_{stem or att}'


def jload(p):
    return json.loads(Path(p).read_text())


def head_of(name):
    m = re.match(r'^H(\d+)s(\d)', name)
    if name.startswith('Hinc'):
        return NH.Head(INC)
    return NH.Head(HERE / f'heads/head_mid_k{m.group(1)}_s{m.group(2)}.npz')


# ------------------------------------------------------------------ training acceptance ----

INC_RECON, INC_ORACLE = 0.007507219122474576, 0.005200570538097191      # hfit_dn256b_full.json arms.mid


def accept_train(tr):
    """DESIGN 2.3 / A1.7 / A2.2 / A5 gates of the training job, recomputed from its result."""
    want = ['identity'] + [f'mid_k{K}_s{s}' for K in KS for s in SEEDS] + ['probe_k16_s0_2000']
    a = tr['arms']
    idn = a.get('identity', {})
    out = dict(complete=bool(tr.get('complete')), arms_present=sorted(a) == sorted(want),
               whitening=bool(tr['gates']['whitening_round_trip'] < 1e-10 and tr['gates']['q_equals_LT_h'] < 1e-10),
               disjointness=bool(all(tr['gates']['disjointness'][c]['min_maxabs_distance'] > 1e-9 for c in ('dev6', 'val32'))),
               identity_recon_rel=abs(idn['recon_train']['mean'] - INC_RECON) / INC_RECON if idn else None,
               identity_oracle_rel=abs(idn['oracle_test']['mean'] - INC_ORACLE) / INC_ORACLE if idn else None,
               compact_exact=bool(all(v.get('compact_vs_emitted') == 0. for k, v in a.items() if k != 'identity')),
               steps_done=bool(all(v['train']['steps_done'] == (2000 if k.startswith('probe') else 200000)
                                   for k, v in a.items() if k != 'identity')),
               finite=bool(all(np.isfinite([v['recon_train']['mean'], v['oracle_test']['mean']]).all() for v in a.values())))
    out['identity'] = bool(idn and out['identity_recon_rel'] <= 1e-9 and out['identity_oracle_rel'] <= 1e-3)
    out['accepted'] = bool(all(out[k] for k in ('complete', 'arms_present', 'whitening', 'disjointness', 'identity',
                                                 'compact_exact', 'steps_done', 'finite')))
    return out


# ------------------------------------------------------------------ rules ----

def rules():
    r = jload(out_dir('rule1024', 'hrule') / 'result.json')
    assert r['complete']
    out = dict(source=dict(job=r['job_id'], commit=r['commit'], gpu=r['gpu']), groups={})
    for g, v in r['groups'].items():
        out['groups'][g] = dict(status=v['status'], chosen=v['chosen'], trail=[dict(rule=t['rule'], n_pass=t['n_pass'],
                                                                                    min_pass=t['min_pass']) for t in v['trail']])
    p = RUNS / 'RULES.json'
    p.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1)[:3000])


# ------------------------------------------------------------------ accuracy ----

def acc_records(att):
    d = out_dir(att, 'hacc')
    r = jload(d / 'result.json')
    assert r['complete'], att
    recs = {}
    for name, a in r['arms'].items():
        spec = a['spec']
        for scheme in spec.get('schemes', ['BE']):
            R = dict(np.load(d / f'R_{name}_{scheme}.npz'))
            rows = [x for x in r['rows'] if x['arm'] == name and x['scheme'] == scheme]
            head = head_of(name) if spec['kind'] == 'head' else None
            q = HSC.qualify(R, rows, head=head, qp_rule=spec.get('qp'), qpp=bool(spec.get('qpp')))
            sc = HSC.score(R)
            ids = [str(x) for x in R['case_ids']]
            per = {c: dict(e=sc[c]['ST']['e'], eps=sc[c]['ST']['eps_ref'], eligible=bool(sc[c]['ST']['eligible']),
                           e_S=sc[c]['S']['e'] if 'S' in sc[c] else None,
                           eligible_S=bool(sc[c]['S']['eligible']) if 'S' in sc[c] else None) for c in ids}
            el = [v['e'] for v in per.values() if v['eligible']]
            recs[(name, scheme)] = dict(name=name, scheme=scheme, kind=spec['kind'], rule=spec['rule'], M=int(R['M']),
                                        Q1=q['Q1'], Q2=q['Q2'], Q3=q['Q3'], qualified=q['numerically_qualified'],
                                        disagreements=q['disagreements'][:20], per_case=per, case_qual=q['per_case'],
                                        median=float(np.median(el)) if el else None, worst=float(np.max(el)) if el else None,
                                        it_sum=[x['it_sum'] for x in rows], ic=[x.get('ic_reason') for x in rows],
                                        jacobian=a.get('jacobian'), describe={k: v for k, v in a.items() if k not in ('arms', 'meta')})
            print('ACC', att, name, scheme, recs[(name, scheme)]['qualified'], recs[(name, scheme)]['median'], flush=True)
    return r, recs


# ------------------------------------------------------------------ timing ----

def timing(att):
    d = out_dir(att, 'htime')
    r = jload(d / 'result.json')
    assert r['complete'], att
    subj = {}
    for lb, v in r['subjects'].items():
        subj[lb] = dict(spec=v['spec'], certification=v['certification'])
    fused, inv = {}, {}
    for x in r['fused']:
        fused.setdefault(x['B'], {}).setdefault(x['case'], []).append(x)
    for x in r['invocations']:
        inv.setdefault(x['B'], {}).setdefault(x['case'], []).append(x)
    cost = {}
    for lb in subj:
        cases = r['subjects'][lb]['timed_case_ids']
        per = {c: [x['ratio'] for x in fused.get(lb, {}).get(c, []) if x['valid']] for c in cases}
        pev = {c: [x['ratio'] for x in inv.get(lb, {}).get(c, []) if x['valid']] for c in cases}
        cost[lb] = dict(per_case={c: float(np.median(v)) if v else None for c, v in per.items()},
                        per_case_ev={c: float(np.median(v)) if v else None for c, v in pev.items()},
                        complete=bool(all(per[c] for c in cases)),
                        raw_ev={c: [(x['ratio'], x['valid']) for x in inv.get(lb, {}).get(c, [])] for c in cases},
                        abs_fused_median=float(np.median([x['t_fused'] for c in cases for x in fused.get(lb, {}).get(c, [])
                                                          if x['valid']] or [np.nan])),
                        contract=CONTRACT)
    return r, subj, cost


def tlabel(mesh, name, rule, scheme):
    return f'{mesh}|{name}|{rule}|{scheme}'


# ------------------------------------------------------------------ verdicts ----

def pools(recs, K, scheme, m128=False):
    """S_K: numerically qualified seeds of size K (A2.3)."""
    sfx = 'm128' if m128 else ''
    return [s for s in SEEDS if recs.get((f'H{K}s{s}{sfx}', scheme), {}).get('qualified')]


def seed_median(recs, names, scheme, cases):
    return np.array([np.median([recs[(n, scheme)]['per_case'][c]['e'] for n in names]) for c in cases])


def population(recs, keys):
    """E0-eligible (ST) cases (same for every configuration: eligibility is a property of the case-kind)."""
    first = recs[keys[0]]['per_case']
    return [c for c, v in first.items() if v['eligible']]


def h_acc_k(recs, scheme, m128=False):
    out = {}
    sfx = 'm128' if m128 else ''
    for lo, hi in zip(KS[:-1], KS[1:]):
        Sl, Sh = pools(recs, lo, scheme, m128), pools(recs, hi, scheme, m128)
        if len(Sl) < 2 or len(Sh) < 2:
            out[f'{lo}->{hi}'] = dict(outcome='unqualified size', S_lo=Sl, S_hi=Sh)
            continue
        nl, nh = [f'H{lo}s{s}{sfx}' for s in Sl], [f'H{hi}s{s}{sfx}' for s in Sh]
        cases = population(recs, [(nl[0], scheme)])
        el, eh = seed_median(recs, nl, scheme, cases), seed_median(recs, nh, scheme, cases)
        eps = np.array([recs[(nl[0], scheme)]['per_case'][c]['eps'] for c in cases])
        q = np.ones(len(cases), bool)
        out[f'{lo}->{hi}'] = dict(PS.h_acc_r_transition(eh - el, eps, q, q), S_lo=Sl, S_hi=Sh,
                                  median_lo=float(np.median(el)), median_hi=float(np.median(eh)))
    out['range'] = PS.h_acc_r_range([v['outcome'] for v in out.values() if 'outcome' in v])
    return out


def lh(recs, cost, scheme, mesh=1024, holm=None, per_seed=None):
    """Q-LH-dim and Q-LH-cost at 1024^2 for every K (A2.1-A2.4, A3.1-A3.3, A4.2). per_seed: restrict the head to one seed."""
    res = {}
    comps = {}
    for K in KS:
        S = [per_seed] if per_seed is not None else pools(recs, K, scheme)
        S = [s for s in S if recs.get((f'H{K}s{s}', scheme), {}).get('qualified')]
        hl = {s: tlabel(mesh, f'H{K}s{s}', recs[(f'H{K}s{s}', scheme)]['rule'], scheme) for s in S if (f'H{K}s{s}', scheme) in recs}
        Sp = [s for s in S if hl.get(s) in cost and cost[hl[s]]['complete']]                       # S'_K (A3.2)
        entry = dict(S=S, S_prime=Sp)
        if len(Sp) < (1 if per_seed is not None else 2):
            entry.update(status='unavailable (timing incomplete or unqualified)')
            res[K] = entry
            for r in LIN:
                comps[(K, r)] = 1.
            continue
        names = [f'H{K}s{s}' for s in Sp]
        cases = population(recs, [(names[0], scheme)])
        eH = seed_median(recs, names, scheme, cases)
        cH = np.array([np.median([cost[hl[s]]['per_case'][c] for s in Sp]) for c in cases])
        eps = np.array([recs[(names[0], scheme)]['per_case'][c]['eps'] for c in cases])
        rungs = {}
        for r in LIN:
            key = (f'L{r}', scheme)
            if key not in recs:
                continue
            lq = recs[key]['qualified']
            ll = tlabel(mesh, f'L{r}', recs[key]['rule'], scheme)
            eL = np.array([recs[key]['per_case'][c]['e'] for c in cases])
            complete = ll in cost and cost[ll]['complete']
            cL = np.array([cost[ll]['per_case'][c] for c in cases]) if complete else None
            err = HS.error_label(eH, eL, eps)
            cl = HS.cost_label(cL / cH, complete=complete) if complete else HS.cost_label(None, complete=False)
            rungs[r] = dict(qualified=lq, cost=cl, err=err, median_L=float(np.median(eL)), worst_L=float(np.max(eL)))
            comps[(K, r)] = err['p_x'] if (lq and complete) else 1.
        # Q-LH-dim: head (X) vs linear r = K, cost as head / linear
        dim = rungs.get(K)
        if dim is not None:
            cd = HS.cost_label(1. / (np.array([cost[tlabel(mesh, f'L{K}', recs[(f'L{K}', scheme)]['rule'], scheme)]['per_case'][c]
                                               for c in cases]) / cH)) if dim['cost']['label'] != 'unknown' else HS.cost_label(None, False)
            entry['lh_dim'] = dict(err=_slim(dim['err']), cost=cd, dominance=HS.dominance(dim['err'], cd, qualified=dim['qualified']))
        entry.update(population=len(cases), median_H=float(np.median(eH)), worst_H=float(np.max(eH)),
                     cost_H_median=float(np.median(cH)), rungs={r: dict(v, err=_slim(v['err'])) for r, v in rungs.items()},
                     _rungs_full=rungs)
        res[K] = entry
    return res, comps


def _slim(e):
    return {k: v for k, v in e.items() if k != 'tost'} | dict(tost_equivalent=e.get('tost', {}).get('equivalent'))


def finish_lh(res, comps, family):
    """Family-wise Holm over (K, r) components (A3.1), at-most-cost / equal-cost / bracket per K, F5."""
    fam = {f'{K}|{r}': comps.get((K, r), 1.) for K in family for r in LIN}
    rej = HS.holm(fam)
    per_k = {}
    for K in family:
        e = res[K]
        if 'rungs' not in e:
            per_k[K] = dict(positive_components=False, lh_dim='unavailable', qualified=False, resolved_negative=False)
            continue
        out = HS.lh_cost(dict(qualified=len(e['S_prime']) >= 2, complete=True), e['_rungs_full'],
                         rejected={r: rej[f'{K}|{r}'] for r in LIN if f'{K}|{r}' in rej})
        e['lh_cost'] = out
        e['holm'] = {r: dict(p=fam[f'{K}|{r}'], rejected=rej[f'{K}|{r}']) for r in LIN}
        per_k[K] = dict(positive_components=out['at_most_cost'] == 'head more accurate than every possibly-affordable rung',
                        lh_dim=e.get('lh_dim', {}).get('dominance', 'unavailable'), qualified=len(e['S_prime']) >= 2,
                        resolved_negative=str(out['at_most_cost']).startswith('linear at no more cost'))
    for K in res:
        res[K].pop('_rungs_full', None)
    return HS.f5(list(family), per_k), per_k


def h_cost_n(costs, recs_rule):
    """Per K, seed-0 head, BE: evolution-only cost across the timed meshes (P1 h_cost_n)."""
    out = {}
    meshes = [m for m in (256, 1024, 4096) if m in costs]
    for K in KS + ('inc',):
        name = 'Hinc' if K == 'inc' else f'H{K}s0'
        rule = recs_rule.get(name)
        if rule is None:
            continue
        lbs = {m: tlabel(m, name, rule, 'BE') for m in meshes}
        if not all(lb in costs[m] for m, lb in lbs.items()):
            out[str(K)] = dict(outcome='verdict not reached', reason='subject missing on a mesh')
            continue
        cases = sorted(set.intersection(*[set(costs[m][lbs[m]]['raw_ev']) for m in meshes]))
        cases = [c for c in cases if c.startswith('dev6')]
        nrep = 3
        ratio = np.full((len(cases), len(meshes), nrep), np.nan)
        valid = np.zeros_like(ratio, bool)
        for i, c in enumerate(cases):
            for j, m in enumerate(meshes):
                xs = costs[m][lbs[m]]['raw_ev'][c][:nrep]
                for k_, (rv, ok) in enumerate(xs):
                    ratio[i, j, k_], valid[i, j, k_] = rv, ok
        out[str(K)] = PS.h_cost_n(ratio, valid, meshes) | dict(meshes=meshes, contract='evolution only, A-normalised, dev6')
    return out


# ------------------------------------------------------------------ main ----

def main_all():
    exp = dict(note='exploratory validation evidence only (DESIGN A4.1); E0-conditional; eps_ref is a conditional Richardson '
                    'estimate', contract=CONTRACT)
    exp['rules'] = jload(RUNS / 'RULES.json') if (RUNS / 'RULES.json').exists() else None
    if (out_dir('train', 'htrain') / 'result.json').exists():
        tr = jload(out_dir('train', 'htrain') / 'result.json')
        exp['training'] = {k: dict(k=v['k'], recon=v['recon_train']['mean'], oracle=v['oracle_test']['mean'],
                                   oracle_over_floor=v['oracle_over_span_floor'], seconds=v.get('seconds'),
                                   code_gain=v['oracle_train_from_codes']['gain_vs_recon'])
                           for k, v in tr['arms'].items()}
        exp['training_meta'] = dict(job=tr['config']['slurm_job'], gpu=tr['config']['gpu'], gates=tr['gates'])
        exp['training_acceptance'] = accept_train(tr)
        tp, tf = tr['arms'].get('probe_k16_s0_2000'), tr['arms'].get('mid_k16_s0')
        if tp and tf:                                   # A1.7 sensitivity probe (reported, no required outcome)
            exp['training_probe'] = dict(oracle_probe=tp['oracle_test']['mean'], oracle_full=tf['oracle_test']['mean'],
                                         ratio=tp['oracle_test']['mean'] / tf['oracle_test']['mean'])
    recs = {}
    if (out_dir('acc1024', 'hacc') / 'result.json').exists():
        _, recs = acc_records('acc1024')
    exp['configs_1024'] = {f'{k[0]}|{k[1]}': {kk: vv for kk, vv in v.items() if kk not in ('per_case', 'case_qual')}
                           for k, v in recs.items()}
    exp['per_case_1024'] = {f'{k[0]}|{k[1]}': v['per_case'] for k, v in recs.items()}
    costs = {}
    for m in (256, 1024, 4096):
        if (out_dir(f'time{m}', 'htime') / 'result.json').exists():
            _, _, costs[m] = timing(f'time{m}')
    exp['cost'] = {str(m): {lb: {k: v for k, v in c.items() if k != 'raw_ev'} for lb, c in cm.items()} for m, cm in costs.items()}
    if recs:
        fam = jload(RUNS / 'F5-FAMILY.json')['family'] if (RUNS / 'F5-FAMILY.json').exists() else list(KS)
        exp['S_K'] = {sc: {K: pools(recs, K, sc) for K in KS} for sc in SCHEMES}
        exp['h_acc_k'] = {sc: h_acc_k(recs, sc) for sc in SCHEMES}
        exp['h_acc_k_m128'] = h_acc_k(recs, 'BE', m128=True)
        if 1024 in costs:
            exp['lh'] = {}
            for sc in SCHEMES:
                res, comps = lh(recs, costs[1024], sc)
                f5, per_k = finish_lh(res, comps, fam) if sc == 'BE' else (None, None)
                exp['lh'][sc] = dict(per_K=res, F5=f5, F5_inputs=per_k)
            exp['lh_per_seed'] = {}
            for s in SEEDS:
                res, comps = lh(recs, costs[1024], 'BE', per_seed=s)
                exp['lh_per_seed'][s] = res
            exp['F5_family'] = fam
        rr = {k[0]: v['rule'] for k, v in recs.items()}
        exp['h_cost_n'] = h_cost_n(costs, rr) if costs else None
        inc, k16 = recs.get(('Hinc', 'BE')), recs.get(('H16s0', 'BE'))
        if inc and k16 and 'training' in exp:
            cs = [c for c, v in inc['per_case'].items() if v['eligible']]
            ratio = float(np.median([k16['per_case'][c]['e'] for c in cs]) / np.median([inc['per_case'][c]['e'] for c in cs]))
            orat = exp['training']['mid_k16_s0']['oracle'] / exp['training']['identity']['oracle']
            exp['repro'] = dict(oracle_ratio=orat, rom_median_ratio=ratio,
                                label='reproduces' if abs(orat - 1) <= .15 and 1 / 1.25 <= ratio <= 1.25 else 'does not reproduce')
    (RUNS / 'EXPORT.json').write_text(json.dumps(exp, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)) + '\n')
    print('EXPORT written')


if __name__ == '__main__':
    {'rules': rules, 'all': main_all}[sys.argv[1]]()
