"""jcp-qualify analysis (DESIGN 2-7 with amendments A1-A4): every label and verdict recomputed on CPU from the collected
archives; writes checks/analysis.json (full) and runs/EXPORT.json (machine-readable labels, verdicts, fronts, key
numbers; a `contract` field on every cost number).

    JAX_PLATFORMS=cpu python analyze.py [--skip-3d] [--skip-time]
Archives: runs/<attempt>/archive/output/<driver>_<attempt>/. Missing archives -> "not reached" for what they support.
Inherited P1 configurations: labels from P1's checks/analysis.json (cited), per-case E0 scores recomputed from P1's
archives with the lane scorer (same metric).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import npcheck as NC
import p1stats as PST
import qscore as SC
import qstats as QS

HERE = Path(__file__).resolve().parent
RUNS = HERE / 'runs'
P1 = SC.P1
MESHES = (256, 1024, 4096)
RS = (128, 256, 384, 512)
SCHEMES = (('LSPG', 'BE'), ('LSPG', 'CNR'), ('GAL', 'CN'))
LARGE = {('LSPG', 'BE'): 2.5, ('LSPG', 'CNR'): 2.5, ('GAL', 'CN'): 5.}
RULES = {128: ('fib4181', 'fib6765', 'fib10946'), 256: ('gauss128', 'gauss160', 'gauss192'),
         384: ('gauss128', 'gauss160', 'gauss192'), 512: ('gauss192', 'gauss256', 'gauss320')}
DT0 = .005
SUB16 = set(SC.case_set('sub16'))
ALL38 = SC.case_set('all38')


def fam(form, sc, f):
    return f'{form}|{sc}|{f:g}'


def ckey(mesh, r, form, sc, f, fb=False):
    return f'{mesh}|R{r}|{form}|{sc}|{f:g}' + ('|fb' if fb else '')


def factors(form, sc):
    return (1., 2., LARGE[(form, sc)])


def out_dir(att, drv, stem=None):
    d = RUNS / att / 'archive' / 'output' / f'{drv}_{stem or att}'
    return d if (d / 'result.json').exists() else None


def inherited(r, form, sc, f, mesh):
    return form == 'LSPG' and f == 1 and (r in (128, 256, 384) or (r == 512 and mesh == 1024))


# ------------------------------------------------------------------------------------------- 2D configurations
def lane_config(mesh, d, res, cache, name, arm):
    form, sc, f = arm['form'], arm['scheme'], float(arm['factor'])
    ak = f'{form}_{sc}_{f:g}'
    R = dict(np.load(d / f'R_{name}_{ak}.npz'))
    r = int(R['Rp'])
    cases = SC.case_set(arm['cases']) if isinstance(arm['cases'], str) else [f'{a}|{b}' for a, b in arm['cases']]
    q2c = arm.get('q2_cases', 'sub16')
    q2 = [x for x in (SC.case_set(q2c) if isinstance(q2c, str) else [f'{a}|{b}' for a, b in q2c]) if x in cases]
    dev = [x for x in SC.case_set('dev6') if x in cases] if arm.get('tighter', True) else []
    expect = dict(cases=cases, dev=dev, q2=q2, qpp=True, form=form, steps=int(R['steps']))
    tabs = None
    if q2:
        rule, qp = str(R['rule']), res['settings'][name]['rules']['qp']
        k = (rule, qp, r, mesh)
        if k not in cache:
            cache[k] = (SC.rule_tables(rule, r, R['kx'], R['ky'], mesh), SC.rule_tables(qp, r, R['kx'], R['ky'], mesh))
        tabs = cache[k]
    rows = [x for x in res['rows'] if x['setting'] == name and x['arm'] == ak]
    q = SC.qualify(R, rows, expect, rho_tables=tabs)
    e0 = SC.score_config(R, DT0 * f, ('ST', 'S') if (form, sc, f) == ('LSPG', 'BE', 1.) else ('ST',))
    q3 = None
    if np.isfinite(R['q3_svals']).any():
        cj = [PST.conditioning(sv)['cond'] for sv in np.asarray(R['q3_svals']).reshape(-1, r) if np.isfinite(sv).all()]
        q3 = dict(J_cond_median=float(np.median(cj)), J_cond_max=float(np.max(cj)),
                  JTJ_cond_max=float(np.max(cj)) ** 2 if form == 'LSPG' else None,
                  A_cond=res['settings'][name]['A_cond'], A_rank=res['settings'][name]['A_rank'], jacobian=form)
    zu = SC.zero_update_distance(R, 0) if cases and cases[0] == 'dev6|0' else None
    return dict(R=R, q=q, e0=e0, q3=q3, zero_update=zu, rule=str(R['rule']), m=res['settings'][name]['m']['main'],
                source=f'lane {d.parent.parent.parent.name}', meta=res['settings'][name]['meta']['main'],
                trust=res['settings'][name]['trust'])


def p1_config(mesh, r, sc, p1an, ext=None):
    """Inherited P1 configuration: P1 labels (cited); E0 per case rescored from P1's archive (+ the lane's 22-case
    extension at 4096^2 for R128/R256: pooled Q1 over the union, A1 DESIGN 2.1)."""
    z = dict(np.load(P1 / 'runs' / f'acc{mesh}' / 'archive' / 'output' / f'p1acc_acc{mesh}' / f'R_R{r}_{sc}.npz'))
    z.setdefault('form', np.array('LSPG'))
    lab = p1an['configurations'][f'{mesh}|R{r}|{sc}']
    e0 = SC.score_config(z, DT0, ('ST', 'S') if sc == 'BE' else ('ST',))
    q = dict(Q1=dict(lab['Q1']), Q2=dict(lab['Q2']), numerically_qualified=lab['numerically_qualified'], source='P1 (cited)',
             per_case={})
    if ext is not None:
        qe = ext['q']
        g = np.concatenate([np.asarray(z['gamma'], float).ravel(), np.asarray(ext['R']['cert'], float).ravel()])
        pooled = int(np.sum(~(g <= SC.GAMMA)))
        other = lambda d_: all(v for k, v in d_.items() if k not in ('pass', 'certificate', 'stationarity'))
        mf = manifest_equal_p1(z, ext['R'])
        q['Q1'] = {'p1_non_certificate_predicates': other(lab['Q1']), 'ext22_non_certificate_predicates': other(qe['Q1']),
                   'pooled_noncertified': pooled, 'pooled_steps': int(g.size), 'numerical_manifest_equal': mf,
                   'pass': bool(other(lab['Q1']) and other(qe['Q1']) and pooled <= .01 * g.size and mf)}
        q['numerically_qualified'] = bool(q['Q1']['pass'] and q['Q2'].get('label') == 'pass')
        q['source'] = 'P1 (cited, 16 cases) + lane ext22 (pooled Q1)'
        e0.update(ext['e0'])
    return dict(R=z, q=q, e0=e0, q3=None, rule=RULES[r][0], source=q['source'])


def per_case(entry):
    return {c: v['ST']['e'] for c, v in entry['e0'].items()}


def eligible(entry):
    return {c: v['ST']['eligible'] for c, v in entry['e0'].items()}


def eps(entry):
    return {c: v['ST']['eps_ref'] for c, v in entry['e0'].items()}


def stats(entry):
    E, el = per_case(entry), eligible(entry)
    v = [E[c] for c in ALL38 if c in E and el[c]]
    ep = [eps(entry)[c] for c in ALL38 if c in E and el[c]]
    o = np.argsort(v) if v else []
    mid = [int(o[(len(v) - 1) // 2]), int(o[len(v) // 2])] if v else []
    return dict(n=len(v), median=float(np.median(v)) if v else None, worst=float(np.max(v)) if v else None,
                eps_median=float(np.median(ep)) if ep else None, complete38=len(v) == len(ALL38),
                eps_of_median_cases=float(max(ep[i] for i in mid)) if v else None,      # eps of the case(s) defining the
                eps_of_worst_case=float(ep[int(np.argmax(v))]) if v else None)          # median / the worst (roles)


def case_qualified(entry):
    q = entry['q']
    base = q['Q1']['pass'] and q['Q2'].get('label') == 'pass'
    pc = q.get('per_case') or {}
    return {c: bool(base and (c not in pc or (pc[c]['valid'] and pc[c]['tight_valid'] and pc[c]['dist_tight'] <= SC.BAR_Q1)))
            for c in entry['e0']}


def manifest_equal_p1(z, R):
    """Numerical-configuration equality of the lane's ext22 run with P1's archive (AM-1 rule; provenance not compared)."""
    keys = ('Rp', 'mesh', 'rule', 'scheme', 'dt', 'mode_hash', 'policy', 'level', 'budget', 'time_tag', 'metric_id_target')
    ok = all(str(z[k]) == str(R[k]) for k in keys) and np.array_equal(np.asarray(z['tol'], float), np.asarray(R['tol'], float)) \
        and str(R['form']) == 'LSPG' and np.array_equal(z['kx'], R['kx']) and np.array_equal(z['ky'], R['ky'])
    return bool(ok)


def manifest_equal_combined(R, rp):
    """Numerical-configuration equality with combined's c2grid configuration (mesh 1024, frozen = builder list at 1024^2,
    rule per setting, prod tolerances, budget 600, metric) before the reproduction gate is read (AM-1 / AM-7)."""
    dev = json.loads((HERE / 'runs' / 'MANIFEST-dev.json').read_text())
    comb = json.loads((Path(str(SC.COMB)).parents[3] / 'configs' / 'c2grid.json').read_text())
    rule = {s['Rp']: s['rule'] for s in comb['settings'] if s['name'] == f'R{rp}'}
    M = 4 * rp
    return bool(int(R['mesh']) == comb['mesh'] == 1024 and rule.get(rp) == str(R['rule']) and str(R['policy']) == 'frozen'
                and str(R['mode_hash']) == dev['mode_lists'][f'1024|frozen|{M}']['hash'] == dev['mode_lists'][f'1024|nominal|{M}']['hash']
                and str(R['level']) == 'prod' and int(R['budget']) == 600 and str(R['time_tag']) == SC.TIME_TAG
                and [str(R['form']), str(R['scheme'])] in comb['grid']['forms_schemes'])


def build_2d():
    p1an = json.loads((P1 / 'checks' / 'analysis.json').read_text())
    C, gates, cache = {}, {}, {}
    for mesh in MESHES:
        ext = {}
        atts = [f'acc{mesh}'] + sorted(p.name for p in RUNS.glob(f'fb{mesh}*'))
        for att in atts:
            d = out_dir(att, 'qacc')
            if d is None:
                continue
            res = json.loads((d / 'result.json').read_text())
            assert res['complete'], f'{att} incomplete'
            for s in res['config']['settings']:
                fb = s['name'].endswith('fb')
                for arm in s['arms']:
                    e = lane_config(mesh, d, res, cache, s['name'], arm)
                    if arm.get('ext22'):
                        ext[(s['Rp'], arm['scheme'])] = e
                        continue
                    k = ckey(mesh, s['Rp'], arm['form'], arm['scheme'], float(arm['factor']), fb)
                    assert k not in C, ('duplicate configuration', k)
                    C[k] = e
                    if mesh == 1024 and s['Rp'] in (128, 384, 512) and not fb:     # registered gate (DESIGN 7): blocks
                        g = SC.combined_reproduction(e['R'], s['Rp'], arm['form'], arm['scheme'], DT0 * float(arm['factor']))
                        g['manifest_equal'] = manifest_equal_combined(e['R'], s['Rp'])
                        g['passed'] = bool(g['passed'] and g['manifest_equal'])
                        gates[k] = g
                        e['q']['reproduction_gate'] = g['passed']
                        e['q']['numerically_qualified'] = bool(e['q']['numerically_qualified'] and g['passed'])
        for r in RS:
            for sc in ('BE', 'CNR'):
                if inherited(r, 'LSPG', sc, 1., mesh):
                    if mesh == 4096 and r in (128, 256) and (r, sc) not in ext:
                        continue                     # the 38-case population needs the lane's ext22 arm
                    C[ckey(mesh, r, 'LSPG', sc, 1.)] = p1_config(mesh, r, sc, p1an, ext.get((r, sc)))
    return C, gates


# ------------------------------------------------------------------------------------------------- B1
def h_acc_n(C, r, form, sc, f):
    e = {m: C.get(ckey(m, r, form, sc, f)) for m in MESHES}
    if e[256] is None or e[1024] is None:
        return dict(outcome='verdict not reached', reason='missing 256^2 or 1024^2 configuration')
    E = {m: per_case(e[m]) for m in MESHES if e[m]}
    el = {m: eligible(e[m]) for m in MESHES if e[m]}
    ep = eps(e[1024])
    pairs, pairs38 = {}, {}
    for m, tag in ((256, 'P256'), (4096, 'P4096')):
        if e[m] is None:
            continue
        for sub, dst in ((m != 4096, pairs), (False, pairs38)):
            cs = [c for c in E[m] if c in E[1024] and el[m][c] and el[1024][c] and (m != 4096 or c in SUB16 or dst is pairs38)]
            if m == 4096 and dst is pairs:
                cs = [c for c in cs if c in SUB16]
            dst[tag] = (np.array([E[m][c] for c in cs]), np.array([E[1024][c] for c in cs]), np.array([ep[c] for c in cs]))
    slopes = None
    if e[4096] is not None:
        cs = [c for c in E[4096] if c in SUB16 and all(c in E[m] and el[m][c] for m in MESHES)]
        slopes = [np.polyfit(np.log2(MESHES), [E[m][c] for m in MESHES], 1)[0] for c in cs]
    qual = all(e[m]['q']['numerically_qualified'] for m in MESHES if e[m] is not None)
    out = PST.h_acc_n(pairs, slopes, qualified=qual)
    out['sensitivity_P4096_38'] = PST.h_acc_n({'P256': pairs['P256'], 'P4096': pairs38['P4096']}, slopes, qualified=qual) \
        if 'P4096' in pairs38 else None
    out['qualified_configs'] = {m: e[m]['q']['numerically_qualified'] for m in MESHES if e[m] is not None}
    if e[4096] is None:
        out['label'] = 'two-mesh evidence, registered three-mesh verdict not reached'
    return out


def h_acc_r(C, mesh, form, sc, f):
    out = {}
    for lo, hi in ((128, 256), (256, 384), (384, 512)):
        a, b = C.get(ckey(mesh, lo, form, sc, f)), C.get(ckey(mesh, hi, form, sc, f))
        if a is None or b is None:
            out[f'{lo}->{hi}'] = dict(outcome='not reached')
            continue
        Ea, Eb, ela, elb, ep = per_case(a), per_case(b), eligible(a), eligible(b), eps(a)
        qa, qb = case_qualified(a), case_qualified(b)
        cs = [c for c in ALL38 if c in Ea and c in Eb and ela[c] and elb[c]]
        out[f'{lo}->{hi}'] = PST.h_acc_r_transition([Eb[c] - Ea[c] for c in cs], [ep[c] for c in cs],
                                                     [qa[c] for c in cs], [qb[c] for c in cs])
        out[f'{lo}->{hi}']['population'] = len(cs)
    out['range_128_384'] = PST.h_acc_r_range([out[k]['outcome'] for k in ('128->256', '256->384')])
    out['range_128_512'] = PST.h_acc_r_range([out[k]['outcome'] for k in ('128->256', '256->384', '384->512')])
    return out


# ------------------------------------------------------------------------------------------------- B4
def b4():
    d = out_dir('order', 'qacc')
    if d is None:
        return dict(status='not reached')
    out = b4_from(d, json.loads((d / 'result.json').read_text()))
    th = out.get('R384|GAL_TH06')
    sm = out_dir('smkA', 'qacc')
    th_s = b4_from(sm, json.loads((sm / 'result.json').read_text())).get('R384|GAL_TH06') if sm else None
    ok = bool(th is not None and th['label']['label'] == 'order 1' and th['label']['interpretable'] >= 5)
    agree = bool(th_s is not None and th is not None and th_s['label']['label'] == th['label']['label'])
    out['TH06_control'] = dict(label=th['label']['label'] if th else None, interpretable=th['label']['interpretable'] if th else 0,
                               smoke_label=th_s['label']['label'] if th_s else None, agrees_with_smoke=agree,
                               passed=bool(ok and agree))
    if not out['TH06_control']['passed']:                 # A1.4: a failed / unavailable control blocks every B4 label
        for k, v in out.items():
            if k != 'TH06_control' and isinstance(v, dict) and 'label' in v:
                v['label_unblocked'] = v['label']
                v['label'] = dict(v['label'], label='blocked (TH06 control failed or unavailable)')
    return out


def b4_from(d, res):
    out = {}
    for sname, blk in res.get('order', {}).items():
        r = int(sname[1:])
        for ak, v in blk.items():
            z = np.load(d / f'ORDER_{sname}_{ak}.npz')
            W = dict(zip([str(k) for k in z['keys']], z['W']))
            dec = lambda w: NC.decode257(w, r)
            fin, an, an2 = v['finest'], v['anchor'], v['anchor2']
            fit = [f_ for f_ in v['factors'] if f_ in (1., .5, .25)]
            cres = []
            for cid in v['cases']:
                n0 = v['n0'][cid]
                F = lambda role, f_, lev='tight': dec(W[f'{cid}|{role}|{f_:g}|{lev}'])
                Fa = F('main', an)
                e = {f_: NC.fdist(F('main', f_), Fa, n0) for f_ in v['factors'] if f_ not in (an, an2)}
                d_anchor = NC.fdist(Fa, F('main', an2), n0)
                d_solver = NC.fdist(F('main', fin), F('main', fin, 'tighter'), n0)
                Fq, Fqa = F('qp', fin), F('qp', an)
                Fm = F('main', fin)
                d_quad = max(float(np.linalg.norm((Fm[k] - Fa[k]) - (Fq[k] - Fqa[k]))) / n0 for k in range(1, 6))
                d_finer = NC.fdist(Fq, F('qpp', fin), n0)
                valid = all(ok for k, ok in v['valid'].items() if k.startswith(cid + '|'))
                c_ = QS.order_case(e, d_anchor, d_solver, d_quad, d_finer, valid, hs=tuple(fit))
                pw_all = sorted(e)[::-1]
                c_.update(case=cid, e={f'{k:g}': x for k, x in e.items()}, d_anchor=d_anchor, d_solver=d_solver, d_quad=d_quad,
                          d_finer=d_finer, pairwise_ladder=[float(np.log(e[pw_all[i]] / e[pw_all[i + 1]]) / np.log(pw_all[i] / pw_all[i + 1]))
                                                            if e[pw_all[i]] > 0 and e[pw_all[i + 1]] > 0 else None for i in range(len(pw_all) - 1)])
                cres.append(c_)
            out[f'R{r}|{ak}'] = dict(cases=cres, label=QS.order_label(cres))
    return out


# ------------------------------------------------------------------------------------------------- timing / B5
def timing(mesh, C):
    d = out_dir(f'time{mesh}', 'qtime')
    if d is None:
        return None
    res = json.loads((d / 'result.json').read_text())
    assert res['complete']
    subs = {}
    for lb, s in res['subjects'].items():
        sp = s['spec']
        if lb.startswith('1024|R384|gauss128|LSPG|CNR|1') and int(sp['mesh']) == 1024 and sp == res['config']['A']:
            pass
        inv = [x for x in res['invocations'] if x['B'] == lb]
        fus = [x for x in res['fused'] if x['B'] == lb]
        ok_inv = [x for x in inv if x['valid']]
        ok_fus = [x for x in fus if x['valid']]
        ident_fail = any(not x['all_hashes_ok'] for x in inv) or any(not x['hash_ok'] for x in fus)
        usable = bool(inv and len(ok_inv) >= (2 / 3) * len(inv) and fus and len(ok_fus) >= (2 / 3) * len(fus) and not ident_fail)
        # timing-job qualification on the six dev6 trajectories (A2.3 / A3.2 of P1)
        Z = dict(np.load(d / f"S_{lb.replace('|', '_')}.npz"))
        cid = [str(x) for x in Z['case_ids']]
        Z['valid'] = Z['valid']
        rows = []
        for r_ in s['certification']:
            coh, c = r_['case'].split('|')
            rows.append(dict(r_, cohort=coh, case=int(c)))
        expect = dict(cases=cid, dev=cid, q2=cid, qpp=True, form=sp['form'], steps=int(Z['steps']))
        tabs = SC.rule_tables(sp['rule'], int(sp['Rp']), Z['kx'], Z['ky'], int(sp['mesh'])), \
            SC.rule_tables(sp['qp'], int(sp['Rp']), Z['kx'], Z['ky'], int(sp['mesh']))
        Z['W_tighter'] = Z['W_tighter']
        Z['valid_tighter'] = Z['valid_tighter']
        Z['valid_qpp'] = Z['valid_qpp']
        Z['W_qpp'] = Z['W_qpp']
        Z.setdefault('w0', Z['W'][:, 0])
        q = SC.qualify(Z, rows, expect, rho_tables=tabs)
        subs[lb] = dict(spec=sp, triplets=len(inv), valid_triplets=len(ok_inv), fused=len(fus), valid_fused=len(ok_fus),
                        identity_failure=ident_fail, usable=usable,
                        c=float(np.median([x['ratio'] for x in ok_fus])) if ok_fus else None,
                        c_ev=float(np.median([x['ratio'] for x in ok_inv])) if ok_inv else None,
                        t_fused=float(np.median([x['t_fused'] for x in ok_fus])) if ok_fus else None,
                        t_evolve=float(np.median([x['t_evolve'] for x in ok_inv])) if ok_inv else None,
                        t_A=float(np.median([.5 * (x['tA1'] + x['tA2']) for x in ok_fus])) if ok_fus else None,
                        timing_label=dict(Q1=q['Q1'], Q2=q['Q2'], numerically_qualified=q['numerically_qualified']),
                        identity_hashes={r_['case']: r_['ident'] for r_ in s['certification']},
                        contract='full-field 2D query (input sampled at quadrature points; full-field decode), H200, '
                                 'ratio to P1 reference A in the same job')
    eq = res['equivalence']
    dbl = res['doubled']
    return dict(job=res['job_id'], gpu=res['gpu'], commit=res['commit'], subjects=subs,
                controls=dict(split_fused=all(x['passed'] for x in eq),
                              doubled=dict(n=len(dbl), median=float(np.median([x['ratio'] for x in dbl])) if dbl else None,
                                           passed=bool(dbl) and 1.8 <= float(np.median([x['ratio'] for x in dbl])) <= 2.2 and
                                           all(x['same_iterations'] and x['identity_ok'] for x in dbl))))


def is_candidate(rec):
    """DESIGN 6: both numerical labels, usable timing, the 38-case population."""
    return bool(rec['acc_label'] and rec['timing_label'] and rec['usable'] and rec['complete38'] and rec['c'] is not None)


def fronts_from_records(records):
    """Fronts over the candidate records only (unqualified records never enter, even if they would dominate)."""
    cand = {k: v for k, v in records.items() if is_candidate(v)}
    fr = {}
    for stat in ('median', 'worst'):
        for cost in ('c', 'c_ev'):
            fr[f'{stat}|{cost}'] = QS.front({k: dict(c=v[cost], e=v[stat], eps=v['eps']) for k, v in cand.items()})[0]
    return fr


def front2d(C, T):
    out = {}
    for mesh in MESHES:
        tm = T.get(mesh)
        if tm is None:
            out[mesh] = dict(status='timing not reached')
            continue
        cand, unq, unq_rec = {}, {}, {}
        for lb, s in tm['subjects'].items():
            sp = s['spec']
            k = ckey(mesh, int(sp['Rp']), sp['form'], sp['scheme'], float(sp['factor']))
            e = C.get(k)
            if e is None:
                unq[k] = 'no accuracy entry'
                continue
            st = stats(e)
            rec = dict(c=s['c'], c_ev=s['c_ev'], median=st['median'], worst=st['worst'], eps=st['eps_median'],
                       acc_label=e['q']['numerically_qualified'], timing_label=s['timing_label']['numerically_qualified'],
                       usable=s['usable'], complete38=st['complete38'])
            if is_candidate(rec):
                cand[k] = rec
            else:
                unq[k] = rec
                unq_rec[k] = rec
        fr = fronts_from_records({**cand, **unq_rec})
        out[mesh] = dict(candidates=cand, unqualified=unq, fronts=fr,
                         members=sorted(set(fr['median|c']) | set(fr['worst|c']), key=lambda k: cand[k]['c']))
    return out


def roles(C, T):
    out = {}
    for mesh in MESHES:
        tm = T.get(mesh)
        if tm is None:
            continue
        cost = {ckey(mesh, int(s['spec']['Rp']), s['spec']['form'], s['spec']['scheme'], float(s['spec']['factor'])): s['c']
                for s in tm['subjects'].values() if s['usable'] and s['timing_label']['numerically_qualified']}
        for k, c in cost.items():
            e = C.get(k)
            if e is None or not e['q']['numerically_qualified'] or c is None:
                continue
            r = int(k.split('|')[1][1:])
            b = ckey(mesh, r, 'LSPG', 'BE', 1.)
            if b not in C or cost.get(b) is None:
                continue
            if not C[b]['q']['numerically_qualified']:
                continue
            sx, sb = stats(e), stats(C[b])
            tie_m = 2 * max(sx['eps_of_median_cases'], sb['eps_of_median_cases'])
            tie_w = 2 * max(sx['eps_of_worst_case'], sb['eps_of_worst_case'])
            ratio = c / cost[b]
            fast = ratio <= .6 and sx['median'] <= sb['median'] + tie_m and sx['worst'] <= sb['worst'] + tie_w
            acc = ratio <= 1.1 and sx['median'] < sb['median'] - tie_m
            out[k] = dict(cost_vs_BE=ratio, role=('fast-equal-accuracy' if fast else '') + (' accurate-equal-cost' if acc else '') or 'no role')
    return out


# ------------------------------------------------------------------------------------------------- 3D
def q3d():
    import front3d as F3
    out = dict(rounds=[])
    tested = {}
    for att in sorted(p.name for p in RUNS.glob('q3d*')):
        d = out_dir(att, 'q3run')
        if d is None:
            continue
        res = json.loads((d / 'result.json').read_text())
        for key, m in res['members'].items():
            tested[key] = label3d(m, att)
        if res.get('control'):
            out['control'] = dict(label3d(res['control'], att), expected='fail',
                                  passed=label3d(res['control'], att)['label'] == 'fail')
        out['rounds'].append(dict(attempt=att, job=res.get('job_id'), members=list(res['members'])))
    C = F3.candidates()
    passed = {k: v for k, v in C.items() if tested.get(k, {}).get('label') == 'pass'}
    fm, _ = QS.front(F3.points(passed, 'median')) if passed else ([], {})
    fw, _ = QS.front(F3.points(passed, 'worst')) if passed else ([], {})
    pm, pw = F3.points(C, 'median'), F3.points(C, 'worst')
    open_ = [k for k in C if k not in tested and not (any(QS.dominates(pm[x], pm[k]) for x in passed)
                                                      and any(QS.dominates(pw[x], pw[k]) for x in passed))]
    ctrl_ok = bool(out.get('control', {}).get('passed'))
    if not ctrl_ok:                                        # the 3D under-resolved control blocks Q2-3D and B5-3D (A1.8)
        fm, fw = [], []
    out.update(tested=tested, qualified_subset_front=dict(median=fm, worst=fw), open_candidates=open_, control_fired=ctrl_ok,
               label='qualified subset of the tested candidates (never a completeness claim, A2.1)',
               contract='speed3d screening contract: full-field 65^3 mesh-path rollout; ratios from separate jobs against C_old')
    return out


def label3d(m, att):
    """Q2-3D label (DESIGN 2.4, A2.1) + reproduction gate against speed3d's archive."""
    import front3d as F3
    dq = np.asarray(m['dist_qp'], float)
    rho = np.asarray(m['rho'], float).ravel()
    rho_ok = rho[~np.isnan(rho)]
    roll = bool(dq.size and np.isfinite(dq).all() and dq.max() <= SC.BAR_Q2)
    rbar = bool(rho_ok.size and np.isfinite(rho_ok).all() and rho_ok.max() <= SC.BAR_RHO)
    conv = None
    if m.get('qpp'):
        dc = np.asarray(m['dist_qp_qpp'], float)
        conv = bool(dc.size and np.isfinite(dc).all() and dc.max() <= SC.BAR_Q2)
    succ_valid = all(m['valid_qp']) and (not m.get('qpp') or all(m['valid_qpp']))
    if not succ_valid:
        lab = 'unverified (successor invalid)'
    elif not rho_ok.size:
        lab = 'unverified (no rho indicators)'
    elif not (roll and rbar):
        lab = 'fail'
    elif not m.get('qpp'):
        lab = 'unverified (no registered check rule)'
    elif not conv:
        lab = 'unverified (finer rule not converged)'
    else:
        lab = 'pass'
    pop_ok = m['cases'] == list(range(64)) and (not m.get('qpp') or m.get('qpp_cases') == list(range(8)))
    off = offline3d(m, att)
    rep = repro3d(m, att)
    if not all(m['valid']):
        lab = 'fail (own runs invalid)'
    elif lab == 'pass' and not pop_ok:
        lab = 'blocked (population)'
    elif lab == 'pass' and not off['agrees']:
        lab = 'blocked (offline recomputation missing or disagrees)'
    elif lab == 'pass' and not (rep and rep['passed']):
        lab = 'blocked (reproduction gate failed or missing)'
    return dict(label=lab, population_ok=pop_ok, offline=off, worst_dist_qp=float(dq.max()), rho_max=float(rho_ok.max()) if rho_ok.size else None,
                worst_dist_qp_qpp=float(np.max(m['dist_qp_qpp'])) if m.get('qpp') else None, rollout=roll, rho=rbar,
                convergence=conv, own_runs_valid=all(m['valid']), reproduction=rep, qp=m['qp'], qpp=m.get('qpp'), attempt=att)


def offline3d(m, att):
    """Agreement of the in-job Q2-3D numbers with q3offline.py (distances on every case, rho on cases 0-1)."""
    p = HERE / 'checks' / f'q3offline-{att}.json'
    if not p.exists():
        return dict(agrees=False, reason='missing')
    o = json.loads(p.read_text())
    key = m['key'] if m.get('label') != 'control' else 'control|' + m['key']
    o = o.get(key) or o.get('control|' + m['key'])
    if o is None:
        return dict(agrees=False, reason='member missing')
    ag = lambda a_, b_: all(abs(x - y) <= 1e-9 * abs(y) + 1e-13 for x, y in zip(a_, b_)) and len(a_) == len(b_)
    rj = np.asarray(m['rho'], float)[:2].ravel().tolist()
    ok = ag(o['dist_qp'], m['dist_qp']) and (not m.get('qpp') or ag(o['dist_qp_qpp'], m['dist_qp_qpp'])) and \
        all((np.isnan(x) and np.isnan(y)) or abs(x - y) <= 1e-9 * abs(y) + 1e-13 for x, y in zip(o['rho_cases01'], rj))
    return dict(agrees=bool(ok))


def repro3d(m, att):
    S3 = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/runs')
    key = f"{m['setting']}|{m['rule']}|{m['spec']}"
    d = out_dir(att, 'q3run')
    z = np.load(d / 'W_q2.npz')
    if f"{m['setting']}|{m['rule']}|{m['spec']}" not in z.files:
        return None
    W = z[key]
    for g in ('g1', 'g2', 'g3'):
        a = np.load(S3 / g / 'archive' / 'output' / 'W_grid.npz')
        if key in a.files:
            Wa = a[key]
            ca = list(a[f'cases|{key}']) if f'cases|{key}' in a.files else list(range(len(Wa)))
            rel = [float(np.max(np.linalg.norm(W[i] - Wa[ca.index(j)], axis=1)) / np.linalg.norm(Wa[ca.index(j)][0]))
                   for i, j in enumerate(m['cases']) if j in ca]
            return dict(archive=g, cases=len(rel), worst=max(rel), passed=bool(rel) and max(rel) <= 1e-10)
    return dict(archive=None, passed=False, note='configuration not found in speed3d archives')


# ------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-3d', action='store_true')
    ap.add_argument('--skip-time', action='store_true')
    a = ap.parse_args()
    C, gates = build_2d()
    cfgs = {}
    for k, e in C.items():
        cfgs[k] = dict(source=e['source'], rule=e['rule'], Q1=e['q']['Q1'], Q2=e['q']['Q2'],
                       numerically_qualified=e['q']['numerically_qualified'], stats=stats(e), q3=e.get('q3'),
                       zero_update=e.get('zero_update'),
                       per_case={c: dict(e=v['ST']['e'], eps_ref=v['ST']['eps_ref'], eligible=v['ST']['eligible'],
                                         reference_sha256=v['ST'].get('reference_sha256'),
                                         S=v['S']['e'] if 'S' in v else None) for c, v in e['e0'].items()},
                       disagreements=[str(x) for x in e['q'].get('disagreements', [])][:20])
    B1 = dict(h_acc_n={}, h_acc_r={})
    for form, sc in SCHEMES:
        for f in factors(form, sc):
            for r in RS:
                if not inherited(r, form, sc, f, 256) or r == 512:
                    B1['h_acc_n'][f'R{r}|{fam(form, sc, f)}'] = h_acc_n(C, r, form, sc, f)
            for mesh in MESHES:
                B1['h_acc_r'][f'{mesh}|{fam(form, sc, f)}'] = h_acc_r(C, mesh, form, sc, f)
    T = {} if a.skip_time else {m: timing(m, C) for m in MESHES}
    T = {m: v for m, v in T.items() if v is not None}
    F = front2d(C, T) if T else {}
    out = dict(configurations=cfgs, reproduction_combined=gates, B1=B1, B4=b4(),
               timing={m: {k: v for k, v in t.items()} for m, t in T.items()}, fronts=F, roles=roles(C, T) if T else {},
               threeD=None if a.skip_3d else q3d())
    p1an = json.loads((P1 / 'checks' / 'analysis.json').read_text())
    out['B1']['p1_cited'] = dict(h_acc_n=p1an['h_acc_n'], h_acc_r=p1an['h_acc_r'],
                                 note='P1 verdicts cited (A1.5); P1 4096^2 transitions are on its 16-case subset')
    (HERE / 'checks').mkdir(exist_ok=True)
    (HERE / 'checks' / 'analysis.json').write_text(json.dumps(out, indent=1, default=str) + '\n')
    exp = dict(lane='jcp-qualify', label='E0 R_ST: conditional Richardson estimate (E0 A10; combined A10, A11, retracted A15 carried)',
               configurations={k: dict(numerically_qualified=v['numerically_qualified'], Q1_pass=v['Q1'].get('pass'),
                                       Q2=v['Q2'].get('label'), median=v['stats']['median'], worst=v['stats']['worst'],
                                       n=v['stats']['n'], source=v['source']) for k, v in cfgs.items()},
               h_acc_n={k: v.get('outcome') for k, v in B1['h_acc_n'].items()},
               h_acc_r={k: {t: x.get('outcome') if isinstance(x, dict) else x for t, x in v.items()} for k, v in B1['h_acc_r'].items()},
               B4={k: (v['label'] if isinstance(v, dict) and 'label' in v else v) for k, v in out['B4'].items()},
               fronts={str(m): dict(members=f.get('members'), fronts=f.get('fronts'),
                                    cost_contract='full-field 2D query; median fused / reference A, one H200 job per mesh')
                       for m, f in F.items()},
               roles=out['roles'],
               threeD=None if out['threeD'] is None else dict(label=out['threeD']['label'], contract=out['threeD']['contract'],
                                                             tested={k: v['label'] for k, v in out['threeD']['tested'].items()},
                                                             front=out['threeD']['qualified_subset_front'],
                                                             open_candidates=len(out['threeD']['open_candidates'])))
    (RUNS / 'EXPORT.json').write_text(json.dumps(exp, indent=1, default=str) + '\n')
    print('configurations', len(cfgs), 'qualified', sum(v['numerically_qualified'] for v in cfgs.values()))


if __name__ == '__main__':
    main()
