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
import npcheck as NC
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
    import smoke_accept as SA
    cfg = jload(HERE / 'configs' / 'rule1024.json')
    ladder = [f'gauss{p}' for p in cfg['ladder'] if p <= cfg['selectable_max']]
    pop = [f'dev6|{i}' for i in range(6)] + [f'val32|{i}' for i in range(10)]
    struct = {g['name']: SA.rule_trail_ok(g, r['groups'][g['name']], ladder, pop, pop[:6]) for g in cfg['groups']}
    ver = HSC.rule_verify(r, out_dir('rule1024', 'hrule'), cfg)
    bad = {k: v['problems'][:5] for k, v in ver.items() if not v['ok']}
    assert sorted(r['groups']) == sorted(g['name'] for g in cfg['groups']) and all(struct.values()) and not bad, (struct, bad)
    out = dict(source=dict(job=r['job_id'], commit=r['commit'], gpu=r['gpu']), groups={},
               verification=dict(structure=struct, recomputed={k: v['chosen_recomputed'] for k, v in ver.items()}))
    for g, v in r['groups'].items():
        out['groups'][g] = dict(status=v['status'], chosen=v['chosen'], trail=[dict(rule=t['rule'], n_pass=t['n_pass'],
                                                                                    min_pass=t['min_pass']) for t in v['trail']])
    p = RUNS / 'RULES.json'
    p.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1)[:3000])


# ------------------------------------------------------------------ accuracy ----

DERIVED = RUNS / 'derived'


def cached(name, fn):
    """Derived per-job records (COMMON-BRIEF 16: raw arrays are offloaded to the cluster). Computed from the raw archive
    when it is present locally and saved to runs/derived/<name>.pkl together with the archive's listing hash; reused
    when the raw arrays have been offloaded (the ARCHIVE-LOCATION listing hash must equal the cached one)."""
    import pickle
    import hashlib
    att = name.split('_')[0]
    arch = RUNS / att / 'archive'
    loc = arch / 'ARCHIVE-LOCATION.txt'
    DERIVED.mkdir(exist_ok=True)
    cp = DERIVED / f'{name}.pkl'
    if loc.exists():
        lh = [l_.split(': ')[1] for l_ in loc.read_text().splitlines() if l_.startswith('listing sha256')][0]
        c = pickle.load(open(cp, 'rb'))
        assert c['listing_sha256'] == lh, f'derived cache of {att} does not match the offloaded archive'
        return c['value']
    v = fn()
    lst = sorted(str(q.relative_to(arch)) for q in arch.rglob('*') if q.is_file())
    h = hashlib.sha256(''.join(hashlib.sha256((arch / f).read_bytes()).hexdigest() + '  ./' + f + '\n' for f in lst).encode()).hexdigest()
    pickle.dump(dict(listing_sha256=h, value=v), open(cp, 'wb'))
    return v


def have(att, drv):
    """A job's results are available: raw archive present, or offloaded with its derived cache."""
    return (out_dir(att, drv) / 'result.json').exists() or (RUNS / att / 'archive' / 'ARCHIVE-LOCATION.txt').exists()


def acc_records(att):
    return cached(f'{att}_acc', lambda: _acc_records(att))


def _acc_records(att):
    d = out_dir(att, 'hacc')
    r = jload(d / 'result.json')
    assert r['complete'], att
    man = jload(RUNS / 'MANIFEST-dev.json')                       # pinned E0 store / metric / model identities
    import p1score as PS1
    assert man['e0_store_version'] == PS1.STORE_VERSION and man['metric_id'] == PS1.METRIC_ID
    assert r['checkpoint_sha256'] == man['checkpoint_sha256'] and r['rotation_sha256'] == man['rotation_sha256']
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
                           ref_sha256=sc[c]['ST'].get('reference_sha256'), store_version=sc[c]['ST'].get('store_version'),
                           metric_id=sc[c]['ST'].get('metric_id'),
                           ref_sha256_S=sc[c]['S'].get('reference_sha256') if 'S' in sc[c] else None,
                           e_S=sc[c]['S']['e'] if 'S' in sc[c] else None,
                           eligible_S=bool(sc[c]['S']['eligible']) if 'S' in sc[c] else None) for c in ids}
            el = [v['e'] for v in per.values() if v['eligible']]
            recs[(name, scheme)] = dict(name=name, scheme=scheme, kind=spec['kind'], rule=spec['rule'], M=int(R['M']),
                                        Q1=q['Q1'], Q2=q['Q2'], Q3=q['Q3'],
                                        qualified=bool(q['numerically_qualified'] and (spec['kind'] != 'head' or (a.get('jacobian') or {}).get('passed'))),
                                        disagreements=q['disagreements'][:20], per_case=per, case_qual=q['per_case'],
                                        median=float(np.median(el)) if el else None, worst=float(np.max(el)) if el else None,
                                        it_sum=[x['it_sum'] for x in rows], ic=[x.get('ic_reason') for x in rows],
                                        jacobian=a.get('jacobian'), describe={k: v for k, v in a.items() if k not in ('arms', 'meta')})
            print('ACC', att, name, scheme, recs[(name, scheme)]['qualified'], recs[(name, scheme)]['median'], flush=True)
    return r, recs


# ------------------------------------------------------------------ timing ----

def timing(att, recs=None, acc_dir=None):
    return cached(f'{att}_time', lambda: _timing(att, recs, acc_dir))


def _timing(att, recs=None, acc_dir=None):
    """Per subject (DESIGN 6, A1.8, A2.4, A3.2; job audits): costs from VALID fused / split repeats whose hashes equal the
    subject's scoring trajectory (T_ident); timing-job Q1/Q2 labels recomputed from the saved arrays
    (hscore.qualify_timing); equivalence and doubled-work controls with coverage; subject-level drift; at 1024^2 the
    timing trajectories compared DIRECTLY with the accuracy job's (decoded scoring-lattice distance <= 2.5e-4 on every
    case, identical configuration identity). usable = all of these; otherwise the cost is 'unknown' downstream."""
    import timing as TM
    d = out_dir(att, 'htime')
    r = jload(d / 'result.json')
    assert r['complete'], att
    cfg = r['config']
    mesh = int(''.join(ch for ch in att[4:] if ch.isdigit()))
    fused, inv = {}, {}
    for x in r['fused']:
        fused.setdefault(x['B'], {}).setdefault(x['case'], []).append(x)
    for x in r['invocations']:
        inv.setdefault(x['B'], {}).setdefault(x['case'], []).append(x)
    eq = {}
    for e in r['equivalence']:
        eq.setdefault(e['subject'], []).append(e)
    dsp = cfg.get('doubled', [])
    dlab = {f"{x['mesh']}|{x['name']}|{x['rule']}|{x['scheme']}" for x in dsp}
    dbl = [x for x in r['doubled']]
    dbl_ok = bool(len(dbl) == 3 * len(dsp) and all(x['drift_ok'] and x['identity_ok'] and x['same_iterations']
                                                     and 1.8 <= x['ratio'] <= 2.2 for x in dbl)
                  and {x['subject'] for x in dbl} == dlab)
    panel = TM.panel_validity(r['invocations'])
    acc = None
    if mesh == 1024 and recs is not None:
        acc = jload(out_dir('acc1024', 'hacc') / 'result.json')
    cost = {}
    for lb, v in r['subjects'].items():
        sp = v['spec']
        name, sc = lb.split('|')[1], lb.split('|')[3]
        cases = v['timed_case_ids']
        Z = np.load(d / f"S_{lb.replace('|', '_')}.npz", allow_pickle=True)
        T_id = {str(c): str(h) for c, h in zip(Z['T_case_ids'], Z['T_ident'])}
        def hash_ok(x, c):        # explicit matching hashes, finite positive ratio (fail closed on anything missing)
            t = T_id.get(c)
            atts = x.get('attempts') or []
            hb = x.get('hash_B', atts[-1].get('hash_B') if atts else None)   # fused records keep the hash per attempt
            return bool(t and x['valid'] and hb == t and atts and all(a_.get('hash_B') == t for a_ in atts)
                        and np.isfinite(x['ratio']) and x['ratio'] > 0)
        per = {c: [x['ratio'] for x in fused.get(lb, {}).get(c, []) if hash_ok(x, c)] for c in cases}
        pev = {c: [x['ratio'] for x in inv.get(lb, {}).get(c, []) if hash_ok(x, c)] for c in cases}
        head = head_of(name) if sp['kind'] == 'head' else None
        tq = HSC.qualify_timing(Z, v['certification'], head=head, qp_rule=sp.get('qp'), cert_ids=[f'dev6|{i}' for i in range(6)])
        cons = None
        if acc is not None and name != 'A_R384':
            rec = recs.get((name, sc))
            if rec is None:
                cons = dict(consistent=False, reason='no accuracy record')
            else:
                R = np.load(out_dir('acc1024', 'hacc') / f'R_{name}_{sc}.npz')
                ida = [str(x) for x in R['case_ids']]
                coef, dim = HSC.coef_fn(dict(kind=sp['kind'], dim=int(Z['dim'])), head)
                same_id = bool(str(R['rule']) == str(Z['rule']) and int(R['M']) == int(Z['M']) and int(R['dim']) == int(Z['dim'])
                               and acc['arms'][name].get('head_sha256', '') == v['describe'].get('head_sha256', ''))
                dist, e_t = {}, {}
                for j, c in enumerate([str(x) for x in Z['T_case_ids']]):
                    if c not in ida:
                        dist[c] = float('inf')
                        continue
                    ft, fa = NC.decode257(coef(Z['T_W'][j]), dim), NC.decode257(coef(R['W'][ida.index(c)]), dim)
                    dist[c] = NC.fdist(ft, fa, float(Z['T_n0'][j]))
                dv = np.array(list(dist.values()), float)
                cons = dict(consistent=bool(same_id and sorted(dist) == sorted(ida) and np.isfinite(dv).all() and dv.max() <= 2.5e-4),
                            max_trajectory_distance=float(dv.max()) if dv.size else None, same_configuration=same_id)
        reps = cfg['reps']
        cover = all(sorted(x['rep'] for x in fused.get(lb, {}).get(c, [])) == list(range(int(reps[c.split('|')[0]])))
                    and sorted(x['rep'] for x in inv.get(lb, {}).get(c, [])) == list(range(int(reps[c.split('|')[0]])))
                    for c in cases) if name != 'A_R384' else True
        complete = bool(cases and cover and all(per[c] for c in cases))
        subj_drift = panel['subject_drift'].get(lb)
        drift_ok = subj_drift is not None and 1 / 1.1 <= subj_drift <= 1.1
        eq_ok = bool(eq.get(lb) and len(eq[lb]) == 1 and eq[lb][0]['passed'])
        usable = bool(complete and tq['label'].startswith('pass') and eq_ok and dbl_ok and drift_ok
                      and (cons is None or cons['consistent']))
        cost[lb] = dict(per_case={c: float(np.median(x)) if x else None for c, x in per.items()},
                        per_case_ev={c: float(np.median(x)) if x else None for c, x in pev.items()},
                        complete=complete, timing_job_label=tq['label'], timing_Q1=tq['Q1'], timing_Q2=tq['Q2'],
                        timing_disagreements=tq['disagreements'][:5], consistency=cons, equivalence=eq_ok,
                        subject_drift=subj_drift, usable=usable,
                        raw_ev={c: [(x['ratio'], hash_ok(x, c)) for x in inv.get(lb, {}).get(c, [])] for c in cases},
                        abs_fused_median=float(np.median([x['t_fused'] for c in cases for x in fused.get(lb, {}).get(c, [])
                                                          if hash_ok(x, c)] or [np.nan])),
                        contract=CONTRACT)
        print('TIME', att, lb, usable, tq['label'], cons and cons.get('max_trajectory_distance'), flush=True)
    meta = dict(job=r['job_id'], gpu=r['gpu'], doubled_ok=dbl_ok, split_panel=panel)
    return r, meta, cost


def tlabel(mesh, name, rule, scheme):
    return f'{mesh}|{name}|{rule}|{scheme}'


# ------------------------------------------------------------------ verdicts ----

def hname(K, s, m128=False):
    """Arm name of head (K, seed); in the fixed-M = 128 panel K = 32 IS the deployed configuration (A7)."""
    return f'H{K}s{s}' + ('m128' if m128 and 4 * K != 128 else '')


def pools(recs, K, scheme, m128=False):
    """S_K: numerically qualified seeds of size K (A2.3)."""
    return [s for s in SEEDS if recs.get((hname(K, s, m128), scheme), {}).get('qualified')]


def seed_median(recs, names, scheme, cases):
    return np.array([np.median([recs[(n, scheme)]['per_case'][c]['e'] for n in names]) for c in cases])


def population(recs, keys):
    """E0-eligible (ST) cases (same for every configuration: eligibility is a property of the case-kind)."""
    first = recs[keys[0]]['per_case']
    return [c for c, v in first.items() if v['eligible']]


def h_acc_k(recs, scheme, m128=False):
    out = {}
    for lo, hi in zip(KS[:-1], KS[1:]):
        Sl, Sh = pools(recs, lo, scheme, m128), pools(recs, hi, scheme, m128)
        if len(Sl) < 2 or len(Sh) < 2:
            out[f'{lo}->{hi}'] = dict(outcome='unqualified size', S_lo=Sl, S_hi=Sh)
            continue
        nl, nh = [hname(lo, s, m128) for s in Sl], [hname(hi, s, m128) for s in Sh]
        cases = population(recs, [(nl[0], scheme)])
        el, eh = seed_median(recs, nl, scheme, cases), seed_median(recs, nh, scheme, cases)
        eps = np.array([recs[(nl[0], scheme)]['per_case'][c]['eps'] for c in cases])
        q = np.ones(len(cases), bool)
        out[f'{lo}->{hi}'] = dict(PS.h_acc_r_transition(eh - el, eps, q, q), S_lo=Sl, S_hi=Sh,
                                  median_lo=float(np.median(el)), median_hi=float(np.median(eh)))
    out['range'] = PS.h_acc_r_range([v['outcome'] for v in out.values() if 'outcome' in v])
    return out


def h_acc_k_seed(recs, scheme, s):
    """Sensitivity: H-acc-K transitions of one seed's heads (unpooled)."""
    out = {}
    for lo, hi in zip(KS[:-1], KS[1:]):
        a, b = recs.get((f'H{lo}s{s}', scheme)), recs.get((f'H{hi}s{s}', scheme))
        if not (a and b and a['qualified'] and b['qualified']):
            out[f'{lo}->{hi}'] = dict(outcome='unqualified size')
            continue
        cases = [c for c, v in a['per_case'].items() if v['eligible']]
        el, eh = np.array([a['per_case'][c]['e'] for c in cases]), np.array([b['per_case'][c]['e'] for c in cases])
        eps = np.array([a['per_case'][c]['eps'] for c in cases])
        q = np.ones(len(cases), bool)
        out[f'{lo}->{hi}'] = PS.h_acc_r_transition(eh - el, eps, q, q)
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
        Sp = [s for s in S if hl.get(s) in cost and cost[hl[s]]['usable']]                         # S'_K (A3.2)
        entry = dict(S=S, S_prime=Sp)
        if len(Sp) < (1 if per_seed is not None else 2):
            entry.update(status='unavailable (timing incomplete or unqualified)')
            res[K] = entry
            for r in LIN:
                comps[(K, r)] = 1.
            continue
        names = [f'H{K}s{s}' for s in Sp]
        cases = population(recs, [(names[0], scheme)])
        ccases = [c for c in cases if all(cost[hl[s]]['per_case'].get(c) is not None for s in Sp)]   # cost population
        eH = seed_median(recs, names, scheme, cases)
        cH = np.array([np.median([cost[hl[s]]['per_case'][c] for s in Sp]) for c in ccases])
        eps = np.array([recs[(names[0], scheme)]['per_case'][c]['eps'] for c in cases])
        rungs = {}
        for r in LIN:
            key = (f'L{r}', scheme)
            if key not in recs:
                continue
            lq = recs[key]['qualified']
            ll = tlabel(mesh, f'L{r}', recs[key]['rule'], scheme)
            eL = np.array([recs[key]['per_case'][c]['e'] for c in cases])
            complete = ll in cost and cost[ll]['usable']
            cL = np.array([cost[ll]['per_case'].get(c, np.nan) for c in ccases]) if complete else None
            complete = bool(complete and cL is not None and np.isfinite(cL).all())
            err = HS.error_label(eH, eL, eps)
            cl = HS.cost_label(cL / cH, complete=complete) if complete else HS.cost_label(None, complete=False)
            rungs[r] = dict(qualified=lq, cost=cl, err=err, median_L=float(np.median(eL)), worst_L=float(np.max(eL)))
            comps[(K, r)] = err['p_x'] if (lq and complete) else 1.
        # Q-LH-dim: head (X) vs linear r = K, cost as head / linear
        dim = rungs.get(K)
        if dim is not None:
            cd = HS.cost_label(1. / (np.array([cost[tlabel(mesh, f'L{K}', recs[(f'L{K}', scheme)]['rule'], scheme)]['per_case'][c]
                                               for c in ccases]) / cH)) if dim['cost']['label'] != 'unknown' else HS.cost_label(None, False)
            entry['lh_dim'] = dict(err=_slim(dim['err']), cost=cd, dominance=HS.dominance(dim['err'], cd, qualified=dim['qualified']))
        entry.update(population=len(cases), cost_population=len(ccases),
                     scope='error on eligible all38, cost on ' + ('all38' if len(ccases) == len(cases) else f'{len(ccases)} timed cases'),
                     median_H=float(np.median(eH)), worst_H=float(np.max(eH)),
                     cost_H_median=float(np.median(cH)), rungs={r: dict(v, err=_slim(v['err'])) for r, v in rungs.items()},
                     _rungs_full=rungs)
        res[K] = entry
    return res, comps


def _slim(e):
    return {k: v for k, v in e.items() if k != 'tost'} | dict(tost_equivalent=e.get('tost', {}).get('equivalent'))


def finish_lh(res, comps, family, min_seeds=2):
    """Family-wise Holm over (K, r) components (A3.1), at-most-cost / equal-cost / bracket per K, F5."""
    fam = {f'{K}|{r}': comps.get((K, r), 1.) for K in family for r in LIN}
    rej = HS.holm(fam)
    per_k = {}
    for K in family:
        e = res[K]
        if 'rungs' not in e:
            per_k[K] = dict(positive_components=False, lh_dim='unavailable', qualified=False, resolved_negative=False)
            continue
        out = HS.lh_cost(dict(qualified=len(e['S_prime']) >= min_seeds, complete=True), e['_rungs_full'],
                         rejected={r: rej[f'{K}|{r}'] for r in LIN if f'{K}|{r}' in rej})
        e['lh_cost'] = out
        e['holm'] = {r: dict(p=fam[f'{K}|{r}'], rejected=rej[f'{K}|{r}']) for r in LIN}
        per_k[K] = dict(positive_components=out['at_most_cost'] == 'head more accurate than every possibly-affordable rung',
                        lh_dim=e.get('lh_dim', {}).get('dominance', 'unavailable'), qualified=len(e['S_prime']) >= min_seeds,
                        resolved_negative=str(out['at_most_cost']).startswith('linear at no more cost'))
    for K in res:
        res[K].pop('_rungs_full', None)
    return HS.f5(list(family), per_k), per_k


def cost_vs_k(timings, costs):
    """DESIGN 5.3 (descriptive): per mesh, per head (BE), median fused cost / A and evolution cost / A over dev6, LM
    iterations per step and evolution cost per LM iteration (A-normalised), from the hash-linked valid repeats; and per
    mesh the fit t(K) = a + b K^p of the per-case seed-median fused cost (p1stats.fit_power, case bootstrap 10,000) —
    over ALL trained seeds (labelled: qualification not required for a cost description)."""
    out = {}
    for m, (r, meta, cost) in timings.items():
        inv = {}
        for x in r['invocations']:
            inv.setdefault(x['B'], []).append(x)
        rows = {}
        for lb, c in cost.items():
            name = lb.split('|')[1]
            if not lb.endswith('|BE') or name == 'A_R384':
                continue
            xs = [x for x in inv.get(lb, []) if x['valid'] and x['case'].startswith('dev6')]
            its = [float(np.sum(x['it'])) for x in xs]
            steps = [len(x['it']) for x in xs]
            rows[name] = dict(fused=float(np.median([v for k_, v in c['per_case'].items() if k_.startswith('dev6') and v is not None]))
                              if any(v is not None for v in c['per_case'].values()) else None,
                              evolution=float(np.median([v for k_, v in c['per_case_ev'].items() if k_.startswith('dev6') and v is not None]))
                              if any(v is not None for v in c['per_case_ev'].values()) else None,
                              its_per_step=float(np.median([a / b for a, b in zip(its, steps)])) if xs else None,
                              ev_cost_per_iteration=float(np.median([x['ratio'] / max(1., np.sum(x['it'])) for x in xs])) if xs else None,
                              usable=c['usable'])
        fit = None
        seeds = [s for s in SEEDS if all(f'H{K}s{s}' in rows for K in KS)]          # 4096^2 panel: seed 0 only
        ks = list(KS) if seeds else []
        if len(ks) >= 3:
            dev = sorted({c_ for lb, c in cost.items() for c_ in c['per_case'] if c_.startswith('dev6')})
            lbs = {(K, s): next(lb for lb in cost if lb.split('|')[1] == f'H{K}s{s}' and lb.endswith('|BE')) for K in ks for s in seeds}
            T = np.array([[np.median([cost[lbs[(K, s)]]['per_case'][c_] for s in seeds]) for K in ks] for c_ in dev])
            pt = PS.fit_power(ks, T.mean(0))
            rng = np.random.default_rng(PS.SEED)
            ps = []
            for _ in range(PS.NBOOT):
                ft = PS.fit_power(ks, T[rng.integers(0, len(dev), len(dev))].mean(0))
                if ft.get('ok') and not ft.get('flat') and not ft.get('boundary'):
                    ps.append(ft['p'])
            fit = dict(point=pt, K=ks, n_ok_boot=len(ps),
                       ci=[float(np.percentile(ps, 2.5)), float(np.percentile(ps, 97.5))] if (pt.get('ok') and len(ps) > 100) else None,
                       identified=bool(pt.get('ok') and not pt.get('flat') and not pt.get('boundary')),
                       seeds=seeds, note='descriptive; seeds timed at this mesh; a fit failing the inherited acceptance (relative rms <= 0.2, '
                                         'interior exponent) leaves the exponent unidentified')
        out[str(m)] = dict(rows=rows, fit=fit, contract=CONTRACT + '; dev6')
    return out


def seed_spread(recs, scheme='BE'):
    """DESIGN 2.4: per K, the seeds' median E0 errors, their max/min, and qualification."""
    out = {}
    for K in KS:
        v = {s: recs.get((f'H{K}s{s}', scheme)) for s in SEEDS}
        med = {str(s): (x['median'] if x else None) for s, x in v.items()}
        vals = [x for x in med.values() if x]
        out[str(K)] = dict(medians=med, qualified={str(s): bool(x and x['qualified']) for s, x in v.items()},
                           max_over_min=max(vals) / min(vals) if vals else None)
    return out


def ic_summary(r):
    """A1.3 descriptive, from the accuracy job's production rows: initial-fit exit reasons, initial-fit LM iterations and
    stationarity ratio per head configuration (per-case values kept), and the rollout LM iterations."""
    out = {}
    for row in r['rows']:
        if row.get('ic_reason', -1) == -1:
            continue
        e = out.setdefault(f"{row['arm']}|{row['scheme']}", dict(per_case={}))
        e['per_case'][row['case']] = dict(reason=row['ic_reason'], it=row['ic_it'], gn=row['ic_gn'], rollout_it=row['it_sum'])
    for k, e in out.items():
        pc = e['per_case'].values()
        e.update(n=len(pc), stationary=sum(x['reason'] == 4 for x in pc), residual=sum(x['reason'] == 1 for x in pc),
                 other=sum(x['reason'] not in (1, 4) for x in pc), ic_it_median=float(np.median([x['it'] for x in pc])),
                 ic_gn_median=float(np.median([x['gn'] for x in pc])), ic_gn_max=float(np.max([x['gn'] for x in pc])),
                 rollout_it_median=float(np.median([x['rollout_it'] for x in pc])))
    return out


def h_cost_n_descriptive(costs):
    """Sensitivity (descriptive, no verdict): the same max/min statistic for EVERY head (all seeds, BE) from the hash-linked
    evolution timings, ignoring the qualification / control gates; labelled unqualified where a gate fails."""
    out = {}
    meshes = [m for m in (256, 1024, 4096) if m in costs]
    for m0 in meshes:
        for lb in costs[m0]:
            pass
    names = sorted({lb.split('|')[1] for m in meshes for lb in costs[m] if lb.endswith('|BE') and lb.split('|')[1].startswith('H')})
    for name in names:
        lbs = {m: next((lb for lb in costs[m] if lb.split('|')[1] == name and lb.endswith('|BE')), None) for m in meshes}
        if any(v is None for v in lbs.values()):
            continue
        med = {}
        for m, lb in lbs.items():
            v = [r_ for c, xs in costs[m][lb]['raw_ev'].items() if c.startswith('dev6') for r_, ok in xs if ok]
            med[str(m)] = float(np.median(v)) if v else None
        vals = [v for v in med.values() if v]
        out[name] = dict(medians=med, ratio=max(vals) / min(vals) if len(vals) == len(med) else None,
                         usable_all=all(costs[m][lb]['usable'] for m, lb in lbs.items()),
                         label='descriptive (gated verdict in h_cost_n)')
    return out


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
        if not all(costs[m][lb]['usable'] for m, lb in lbs.items()):
            out[str(K)] = dict(outcome='verdict not reached', reason='timing not usable on a mesh (qualification / controls)',
                               usable={str(m): costs[m][lb]['usable'] for m, lb in lbs.items()})
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
    if exp['rules']:
        exp.setdefault('job_gpus', {})['rule1024'] = dict(job=exp['rules']['source']['job'], gpu=exp['rules']['source']['gpu'])
    exp['thresholds'] = dict(Q1_distance=2.5e-4, tight_tighter=2.5e-5, Q2_distance=2.5e-4, rho=0.116, gamma=1e-3,
                             equal_cost_band=[1 / 1.1, 1.1], alpha=0.05, tost_margin='max(2 eps_ref, 0.1 min(e_head, e_lin))')
    if (out_dir('train', 'htrain') / 'result.json').exists():
        tr = jload(out_dir('train', 'htrain') / 'result.json')
        exp['training'] = {k: dict(k=v['k'], recon=v['recon_train']['mean'], oracle=v['oracle_test']['mean'],
                                   oracle_over_floor=v['oracle_over_span_floor'], seconds=v.get('seconds'),
                                   code_gain=v['oracle_train_from_codes']['gain_vs_recon'])
                           for k, v in tr['arms'].items()}
        exp['training_meta'] = dict(job=tr['config']['slurm_job'], gpu=tr['config']['gpu'], gates=tr['gates'],
                                    recipe=dict(hidden=512, layers=2, steps=200000, batch=4096, lr=1e-3, npz_sha256=tr['config']['npz_sha256']))
        exp.setdefault('job_gpus', {})['train'] = dict(job=tr['config']['slurm_job'], gpu=tr['config']['gpu'])
        exp['training_acceptance'] = accept_train(tr)
        tp, tf = tr['arms'].get('probe_k16_s0_2000'), tr['arms'].get('mid_k16_s0')
        if tp and tf:                                   # A1.7 sensitivity probe (reported, no required outcome)
            exp['training_probe'] = dict(oracle_probe=tp['oracle_test']['mean'], oracle_full=tf['oracle_test']['mean'],
                                         ratio=tp['oracle_test']['mean'] / tf['oracle_test']['mean'])
    recs = {}
    if have('acc1024', 'hacc'):
        acc_r, recs = acc_records('acc1024')
        exp.setdefault('job_gpus', {})['acc1024'] = dict(job=acc_r['job_id'], gpu=acc_r['gpu'])
    exp['configs_1024'] = {f'{k[0]}|{k[1]}': {kk: vv for kk, vv in v.items() if kk not in ('per_case', 'case_qual')}
                           for k, v in recs.items()}
    exp['per_case_1024'] = {f'{k[0]}|{k[1]}': v['per_case'] for k, v in recs.items()}
    costs, timings = {}, {}
    for m in (256, 1024, 4096):
        if have(f'time{m}', 'htime'):
            timings[m] = timing(f'time{m}', recs)
            _, exp.setdefault('timing_meta', {})[str(m)], costs[m] = timings[m]
            exp.setdefault('job_gpus', {})[f'time{m}'] = dict(job=timings[m][1]['job'], gpu=timings[m][1]['gpu'])
    if timings:
        exp['cost_vs_K'] = cost_vs_k(timings, costs)
    exp['cost'] = {str(m): {lb: {k: v for k, v in c.items() if k != 'raw_ev'} for lb, c in cm.items()} for m, cm in costs.items()}
    if recs:
        fam = jload(RUNS / 'F5-FAMILY.json')['family'] if (RUNS / 'F5-FAMILY.json').exists() else list(KS)
        exp['S_K'] = {sc: {K: pools(recs, K, sc) for K in KS} for sc in SCHEMES}
        exp['seed_spread'] = {sc: seed_spread(recs, sc) for sc in SCHEMES}
        exp['initial_fit'] = ic_summary(acc_r)
        exp['h_acc_k'] = {sc: h_acc_k(recs, sc) for sc in SCHEMES}
        exp['h_acc_k_m128'] = h_acc_k(recs, 'BE', m128=True)
        exp['h_acc_k_per_seed'] = {f'{sc}|{s}': h_acc_k_seed(recs, sc, s) for sc in SCHEMES for s in SEEDS}
        if 1024 in costs:
            exp['lh'] = {}
            for sc in SCHEMES:
                res, comps = lh(recs, costs[1024], sc)
                f5, per_k = finish_lh(res, comps, fam)          # CN-R: the same results; F5 is read from BE only
                exp['lh'][sc] = dict(per_K=res, F5=f5 if sc == 'BE' else None, F5_inputs=per_k if sc == 'BE' else None)
            exp['lh_per_seed'] = {}
            for sc in SCHEMES:
                for s in SEEDS:
                    res, comps = lh(recs, costs[1024], sc, per_seed=s)
                    finish_lh(res, comps, fam, min_seeds=1)
                    exp['lh_per_seed'][f'{sc}|{s}'] = res
            exp['F5_family'] = fam
        if have('acc256', 'hacc'):
            r256, recs256 = acc_records('acc256')
            exp.setdefault('job_gpus', {})['acc256'] = dict(job=r256['job_id'], gpu=r256['gpu'])
            exp['configs_256'] = {f'{k[0]}|{k[1]}': {kk: vv for kk, vv in v.items() if kk not in ('per_case', 'case_qual')}
                                  for k, v in recs256.items()}
            exp['per_case_256'] = {f'{k[0]}|{k[1]}': v['per_case'] for k, v in recs256.items()}
            exp['S_K_256'] = {sc: {K: pools(recs256, K, sc) for K in KS} for sc in SCHEMES}
            if 256 in costs:
                exp['lh_256'] = {}
                for sc in ('BE',):
                    res, comps = lh(recs256, costs[256], sc, mesh=256)
                    finish_lh(res, comps, fam)
                    exp['lh_256'][sc] = dict(per_K=res, scope='error on eligible all38 (acc256), cost on dev6 (time256)')
            cross = {}                                   # E3c descriptive: per-case error spread 256^2 vs 1024^2
            for key, v in recs256.items():
                w_ = recs.get(key)
                if w_ is None:
                    continue
                cs = [c for c in v['per_case'] if v['per_case'][c]['eligible']]
                dd = [abs(v['per_case'][c]['e'] - w_['per_case'][c]['e']) for c in cs]
                cross[f'{key[0]}|{key[1]}'] = dict(max_abs_change=float(max(dd)), median_abs_change=float(np.median(dd)),
                                                   worst_case=cs[int(np.argmax(dd))], both_qualified=bool(v['qualified'] and w_['qualified']))
            exp['cross_mesh_256_1024'] = cross
        rr = {k[0]: v['rule'] for k, v in recs.items()}
        exp['h_cost_n'] = h_cost_n(costs, rr) if costs else None
        exp['h_cost_n_descriptive'] = h_cost_n_descriptive(costs) if costs else None
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
