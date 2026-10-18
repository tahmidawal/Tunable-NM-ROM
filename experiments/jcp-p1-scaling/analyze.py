"""jcp-p1-scaling analysis (DESIGN 4, 5, 2.6; amendments A1-A5): every verdict recomputed on CPU from the collected
archives; writes checks/analysis.json (full) and runs/EXPORT.json (machine-readable verdicts and key numbers).

    python analyze.py            (archives under runs/<attempt>/archive/output)
Attempts: acc256, acc1024, acc4096 (p1acc), time (p1time), kern2d (p1kern), kern3d / p1e3d (p1three); smoke
acceptance from checks/smoke-accept.json. A missing archive is reported as "not reached" for what it supports.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import npcheck as NC
import p1score as SC
import p1stats as S

HERE = Path(__file__).resolve().parent
RUNS = HERE / 'runs'
QS_HIST = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-01-quadrature-study/experiments/quadrature-study/runs')
HIST_JOBS = {256: 'dv256', 1024: 'dv1024', 4096: 'dv4096'}
HIST_ARM = {'H64': 'gauss64', 'H640': 'gref', 'HF6765': 'fib6765', 'H96': 'gauss96'}
HIST_CLAIMS = {'H1': dict(source='reports/2026-10-01-burgers2d-offmesh-quadrature.md:102-129',
                          claim='per-case three-mesh spread of the worst first-order ST error, dev6+val32',
                          values_pp={'H64': .014, 'H640': .012, 'HF6765': .012}),
               'H4': dict(source='worktrees/2026-10-06-jcp-mechanism/experiments/jcp-mechanism/report.md:94',
                          claim='per-case 256^2 vs 1024^2 spread, dev6+val32', values_pp={'H96': .008, 'H640': .008})}
MESHES = (256, 1024, 4096)


def out_dir(att, sub):
    d = RUNS / att / 'archive' / 'output' / sub
    return d if (d / 'result.json').exists() else None


def load_acc(mesh):
    d = out_dir(f'acc{mesh}', f'p1acc_acc{mesh}')
    if d is None:
        return None
    r = json.loads((d / 'result.json').read_text())
    assert r['complete'], f'acc{mesh} incomplete'
    return d, r


def qualify_all(mesh, d, r, cache):
    """Per (config, scheme): qualification, E0 scores, first-order scores; decoded fields kept for P1g distances."""
    res = {}
    ccfg = {c['name']: c for c in r['config']['configs']}
    for name, c in ccfg.items():
        for sc in c['schemes']:
            R = dict(np.load(d / f'R_{name}_{sc}.npz'))
            cases = SC.case_set(c['cases'])
            expect = dict(cases=cases, dev=[x for x in SC.case_set('dev6') if x in cases],
                          q2=[x for x in SC.case_set(c.get('q2_cases', 'sub16')) if x in cases] if c.get('qp') else [],
                          qpp=bool(c.get('qpp')))
            tabs = None
            if c.get('qp'):
                key = (c['rule'], c['qp'], c['Rp'], mesh, c.get('policy', 'frozen'))
                if key not in cache:
                    cache[key] = (SC.rule_tables(c['rule'], c['Rp'], R['kx'], R['ky'], mesh),
                                  SC.rule_tables(c['qp'], c['Rp'], R['kx'], R['ky'], mesh))
                tabs = cache[key]
            q = SC.qualify(R, [x for x in r['rows'] if x['config'] == name and x['scheme'] == sc], c['Rp'], expect,
                           rho_tables=tabs, rho_cases=expect['dev'])
            sc_ = SC.score_config(R, ('ST', 'S') if sc == 'BE' else ('ST',))
            fo = {cid: SC.first_order_st(R['f257'][j], *cid.split('|')) for j, cid in enumerate([str(x) for x in R['case_ids']])}
            q3 = None
            if 'q3_svals' in R:
                cj = [S.conditioning(sv)['cond'] for sv in np.asarray(R['q3_svals']).reshape(-1, c['Rp'])]
                q3 = dict(J_cond_median=float(np.median(cj)), J_cond_max=float(np.max(cj)),
                          JTJ_cond_max=float(np.max(cj)) ** 2, A_cond=r['configs'][name]['A_cond'], A_rank=r['configs'][name]['A_rank'])
            res[(name, sc)] = dict(config=c, R=R, q=q, e0=sc_, fo=fo, q3=q3)
    return res


def per_case_e(entry):
    return {cid: v['ST']['e'] for cid, v in entry['e0'].items()}


def eligible(entry):
    return {cid: v['ST']['eligible'] for cid, v in entry['e0'].items()}


def eps(entry):
    return {cid: v['ST']['eps_ref'] for cid, v in entry['e0'].items()}


def case_qualified(entry):
    q = entry['q']
    base = q['Q1']['pass'] and q['Q2'].get('label') == 'pass'
    return {cid: bool(base and pc['valid'] and pc['tight_valid'] and pc['dist_tight'] <= SC.BAR_Q1) for cid, pc in q['per_case'].items()}


def h_acc_n(A, r, sc, name1024='R{r}', name_other='R{r}'):
    e = {m: A[m].get((name_other.format(r=r) if m != 1024 else name1024.format(r=r), sc)) if A.get(m) else None for m in MESHES}
    if e[256] is None or e[1024] is None:
        return dict(outcome='verdict not reached', reason='missing 256^2 or 1024^2 configuration')
    E = {m: per_case_e(e[m]) for m in MESHES if e[m] is not None}
    el = {m: eligible(e[m]) for m in MESHES if e[m] is not None}
    ep = eps(e[1024])
    pairs = {}
    sub16 = set(SC.case_set('sub16'))
    for m, tag in ((256, 'P256'), (4096, 'P4096')):
        if e[m] is None:
            continue
        cs = [c for c in E[m] if c in E[1024] and el[m][c] and el[1024][c] and (m != 4096 or c in sub16)]
        pairs[tag] = (np.array([E[m][c] for c in cs]), np.array([E[1024][c] for c in cs]), np.array([ep[c] for c in cs]))
    slopes = None
    if e[4096] is not None:
        cs = [c for c in E[4096] if c in sub16 and all(c in E[m] and el[m][c] for m in MESHES)]
        slopes = [np.polyfit(np.log2(MESHES), [E[m][c] for m in MESHES], 1)[0] for c in cs]
    qual = all(e[m]['q']['numerically_qualified'] for m in MESHES if e[m] is not None)
    out = S.h_acc_n(pairs, slopes, qualified=qual)
    out['qualified_configs'] = {m: e[m]['q']['numerically_qualified'] for m in MESHES if e[m] is not None}
    out['abs_error_medians'] = {m: float(np.median(list(E[m].values()))) for m in E}
    out['abs_error_worst'] = {m: float(np.max(list(E[m].values()))) for m in E}
    out['dropped_not_eligible'] = {tag: sum(1 for c in E[1024] if not el[1024][c]) for tag in pairs}
    return out


def h_acc_r(A, mesh, sc):
    trans = [(128, 256), (256, 384)] + ([(384, 512)] if mesh == 1024 else [])
    out = {}
    for lo, hi in trans:
        a, b = A[mesh].get((f'R{lo}', sc)), A[mesh].get((f'R{hi}', sc))
        if a is None or b is None:
            out[f'{lo}->{hi}'] = dict(outcome='not reached')
            continue
        Ea, Eb, ela, elb, ep = per_case_e(a), per_case_e(b), eligible(a), eligible(b), eps(a)
        qa, qb = case_qualified(a), case_qualified(b)
        cs = [c for c in Ea if c in Eb and ela[c] and elb[c]]
        out[f'{lo}->{hi}'] = S.h_acc_r_transition([Eb[c] - Ea[c] for c in cs], [ep[c] for c in cs],
                                                  [qa[c] for c in cs], [qb[c] for c in cs])
    prim = [v['outcome'] for k, v in out.items() if k != '384->512']
    out['range_128_384'] = S.h_acc_r_range(prim)
    if mesh == 1024:
        out['range_128_512'] = S.h_acc_r_range([v['outcome'] for v in out.values() if isinstance(v, dict) and 'outcome' in v])
    return out


def timing(t_dir, A):
    t = json.loads((t_dir / 'result.json').read_text())
    assert t['complete']
    veto = S.identity_veto(t['invocations'], t['fused'], t['doubled'])
    lab = lambda m, r_, sc, pol='frozen': f"{m}|R{r_}|{({128: 'fib4181', 256: 'gauss128', 384: 'gauss128', 512: 'gauss192'})[r_]}|{pol}|{sc}"
    cases = SC.case_set(t['config']['cases'])
    reps = int(t['config']['reps'])

    def arr(lbs, key='ratio'):
        ratio = np.full((len(cases), len(lbs), reps), np.nan)
        valid = np.zeros_like(ratio, bool)
        for x in t['invocations']:
            if x['B'] in lbs:
                ratio[cases.index(x['case']), lbs.index(x['B']), x['rep']] = x[key] if key != 'ratio' else x['ratio']
                valid[cases.index(x['case']), lbs.index(x['B']), x['rep']] = x['valid']
        return ratio, valid

    tq = {}
    for lb, s_ in t['subjects'].items():                    # timing-job qualification (A2.3, A3.2)
        R = dict(np.load(t_dir / f"S_{lb.replace('|', '_')}.npz"))
        sp = s_['spec']
        rows = [dict(x, cohort=x['case'].split('|')[0], case=int(x['case'].split('|')[1])) for x in s_['certification']]
        tabs = (SC.rule_tables(sp['rule'], int(sp['Rp']), R['kx'], R['ky'], int(sp['mesh'])),
                SC.rule_tables(sp['qp'], int(sp['Rp']), R['kx'], R['ky'], int(sp['mesh']))) if sp.get('qp') else None
        q = SC.qualify(R, rows, int(sp['Rp']), dict(cases=cases, dev=cases, q2=cases if sp.get('qp') else [], qpp=bool(sp.get('qpp'))),
                       rho_tables=tabs, rho_cases=cases)
        acc = A.get(int(sp['mesh']), {}).get((('R%d' % int(sp['Rp'])) + ('nom' if sp.get('policy') == 'nominal' else ''), sp['scheme']))
        tq[lb] = dict(timing_job=q['numerically_qualified'], Q1=q['Q1']['pass'], Q2=q['Q2'].get('label'),
                      accuracy_job=acc['q']['numerically_qualified'] if acc else None,
                      both=bool(q['numerically_qualified'] and acc and acc['q']['numerically_qualified']))
    dbl = {}
    for sub in sorted({x['subject'] for x in t['doubled']}):
        xs = [x for x in t['doubled'] if x['subject'] == sub]
        ok = [x['ratio'] for x in xs if x['drift_ok']]
        med = float(np.median(ok)) if ok else float('nan')
        dbl[sub] = dict(median=med, passed=bool(ok and len(ok) >= len(xs) / 2 and 1.8 <= med <= 2.2 and
                                                all(x['same_iterations'] and x['identity_ok'] for x in xs)))
    controls_ok = bool(dbl and all(v['passed'] for v in dbl.values()) and all(x['passed'] for x in t['equivalence']))
    res = {}
    for r_ in (128, 256, 384):
        for sc in ('BE', 'CNR'):
            for pol in ('frozen', 'nominal'):
                if pol == 'nominal' and r_ != 384:
                    continue
                lbs = [lab(m, r_, sc, pol if m != 1024 else 'frozen') for m in MESHES]
                present = [l_ for l_ in lbs if any(x['B'] == l_ for x in t['invocations'])]
                if len(present) < len(lbs):
                    res[f'R{r_}|{sc}|{pol}'] = dict(outcome='verdict not reached', reason='missing mesh', present=present)
                    continue
                if any(veto.get(l_) for l_ in lbs):
                    res[f'R{r_}|{sc}|{pol}'] = dict(outcome='identity failure: timing claims blocked', veto={l_: veto.get(l_) for l_ in lbs})
                    continue
                ratio, valid = arr(lbs)
                o = S.h_cost_n(ratio, valid, list(MESHES))
                if not controls_ok:            # doubled-work or split-equivalence control misbehaved: claim blocked
                    o = dict(o, procedure_outcome=o['outcome'], outcome='blocked (timing control misbehaved)')
                desc = {}
                for k in ('t_init', 't_evolve', 't_decode'):
                    v_, _ = arr(lbs, k)
                    desc[k] = {str(m): float(np.nanmedian(v_[:, i])) for i, m in enumerate(MESHES)}
                fu = {str(m): float(np.median([x['ratio'] for x in t['fused'] if x['B'] == lbs[i] and x['valid']] or [np.nan]))
                      for i, m in enumerate(MESHES)}
                o.update(contract='full-field 2D query (evolution timed; init/decode/fused descriptive)', descriptive_seconds=desc,
                         fused_ratio_median=fu, qualification={l_: tq.get(l_) for l_ in lbs},
                         label='deployment policy (mode set varies with mesh)' if pol == 'nominal' else 'frozen mode list')
                res[f'R{r_}|{sc}|{pol}'] = o
    return dict(controls_ok=controls_ok, h_cost_n=res, doubled=dbl, veto=veto, timing_qualification=tq, components=t['components'],
                equivalence_all=all(x['passed'] for x in t['equivalence']), gpu=t['gpu'],
                memory={lb: dict(live_bytes=s_['live_bytes'], compiled=s_['compiled']) for lb, s_ in t['subjects'].items()},
                banks=t['banks'])


def kernels(k_dir, ranks, tag, timing_controls_ok=True):
    k = json.loads((k_dir / 'result.json').read_text())
    assert k['complete']
    rows = k['panels']['fixed']['rows']
    cases = sorted({r_['case'] for r_ in rows})
    blocked = not (all(v['passed'] for v in k['sync'].values()) and k['panels']['fixed']['drift_ok'] and
                   k['component_equivalence']['passed'] and timing_controls_ok)       # doubled work also governs H-cost-r
    out = {}
    for comp, p in (('N', 1.), ('J', 1.), ('JTJ', 2.), ('chol', 3.)):
        T = np.full((len(cases), 2, len(ranks)), np.nan)
        cnt = {}
        for r_ in rows:
            if r_['comp'] != comp:
                continue
            ci = cases.index(r_['case'])
            kk = cnt.get((ci, r_['r']), 0)
            T[ci, kk, ranks.index(r_['r'])] = r_['median']
            cnt[(ci, r_['r'])] = kk + 1
        m_, M_ = rows[0]['m'], rows[0]['M']
        import p1comp as PC
        fl = [PC.flops_bytes(comp, m_, M_, r_)[0] for r_ in ranks]
        by = [PC.flops_bytes(comp, m_, M_, r_)[1] for r_ in ranks]
        o = S.h_cost_r(ranks, T, p, flops=fl, bytes_=by)
        if blocked:
            o = dict(o, outcome='blocked (control misbehaved)', procedure_outcome=o.get('outcome'))
        o['label'] = 'component-kernel scaling at fixed (M, m) (A2.2, A3.3)'
        o['fixed_cost_check'] = dict(fitted_a=o['point'].get('a'), measured_smallest_rank=float(np.nanmedian(T[:, :, 0])))
        o['medians_by_rank'] = {str(r_): float(np.nanmedian(T[:, :, i])) for i, r_ in enumerate(ranks)}
        out[comp] = o
    desc = {}
    for panel in k['panels']:
        if panel == 'fixed':
            continue
        desc[panel] = {}
        for r_ in k['panels'][panel]['rows']:
            desc[panel].setdefault(r_['comp'], {}).setdefault(str(r_['r']), []).append(r_['median'])
        desc[panel] = {c: {rr: float(np.median(v)) for rr, v in d_.items()} for c, d_ in desc[panel].items()}
    return dict(verdicts=out, descriptive=desc, sync={c: v['ratio_median'] for c, v in k['sync'].items()},
                drift=k['drift'], equivalence=k['component_equivalence'], gpu=k['gpu'], tag=tag)


def p1g(A):
    """Frozen vs nominal for R384-gauss128 (both schemes) and the historical BE rules (A1.8, A2.5)."""
    out = {}
    pairs = [('R384', 'R384nom', 'BE'), ('R384', 'R384nom', 'CNR')] + [(h, h + 'nom', 'BE') for h in HIST_ARM]
    for fz, nm, sc in pairs:
        o = dict(meshes={})
        for m in (256, 4096):
            if A.get(m) is None or (fz, sc) not in A[m] or (nm, sc) not in A[m]:
                o['meshes'][m] = 'not reached'
                continue
            a, b = A[m][(fz, sc)], A[m][(nm, sc)]
            ids = [str(x) for x in a['R']['case_ids']]
            Rp = a['config']['Rp']
            dist, de0, dfo = {}, {}, {}
            jb = {str(x): j for j, x in enumerate(b['R']['case_ids'])}
            for j, cid in enumerate(ids):
                if cid not in jb:
                    continue
                k = jb[cid]
                dist[cid] = NC.fdist(NC.decode257(a['R']['W'][j], Rp), NC.decode257(b['R']['W'][k], Rp), float(a['R']['n0'][j]))
                de0[cid] = a['e0'][cid]['ST']['e'] - b['e0'][cid]['ST']['e']
                dfo[cid] = a['fo'][cid] - b['fo'][cid]
            o['meshes'][m] = dict(cases=len(dist), dist_max=SC.dmax(dist.values()), dist_median=float(np.median(list(dist.values()))),
                                  de0_max_abs_pp=100 * float(np.max(np.abs(list(de0.values())))),
                                  dfo_max_abs_pp=100 * float(np.max(np.abs(list(dfo.values())))),
                                  qualified=dict(frozen=a['q']['numerically_qualified'], nominal=b['q']['numerically_qualified']))
        for pol, key in (('frozen', fz), ('nominal', nm)):
            E = {m: (A[m][(key if m != 1024 else fz, sc)]['fo'], A[m][(key if m != 1024 else fz, sc)]['e0'])
                 for m in MESHES if A.get(m) and (key if m != 1024 else fz, sc) in A[m]}
            if len(E) == 3:
                cs = [c for c in E[4096][0] if c in E[256][0] and c in E[1024][0]]
                sp3 = [100 * (max(E[m][0][c] for m in MESHES) - min(E[m][0][c] for m in MESHES)) for c in cs]
                sp3e = [100 * (max(E[m][1][c]['ST']['e'] for m in MESHES) - min(E[m][1][c]['ST']['e'] for m in MESHES)) for c in cs]
                o[f'spread3_{pol}'] = dict(first_order_max_pp=max(sp3), e0_max_pp=max(sp3e), cases=len(cs))
            if 256 in E and 1024 in E:
                cs = [c for c in E[256][0] if c in E[1024][0]]
                sp2 = [100 * abs(E[256][0][c] - E[1024][0][c]) for c in cs]
                sp2e = [100 * abs(E[256][1][c]['ST']['e'] - E[1024][1][c]['ST']['e']) for c in cs]
                o[f'spread2_{pol}'] = dict(first_order_max_pp=max(sp2), e0_max_pp=max(sp2e), cases=len(cs))
        if fz in HIST_ARM:                                    # reproduction diagnostic vs the historical job rows
            diffs = {}
            for m, job in HIST_JOBS.items():
                key = (fz + 'nom' if m != 1024 else fz, sc)
                if A.get(m) is None or key not in A[m]:
                    continue
                f_ = QS_HIST / job / 'archive' / 'output' / 'result.json'
                if not f_.exists():
                    continue
                hr = {f"{x['cohort']}|{x['case']}": x['ref_ST_evolved'] for x in json.loads(f_.read_text())['rows']
                      if x['setting'] == 'acc' and x['arm'] == HIST_ARM[fz]}
                d_ = [abs(A[m][key]['fo'][c] - hr[c]) for c in A[m][key]['fo'] if c in hr]
                diffs[m] = dict(max_abs=max(d_) if d_ else None, cases=len(d_))
            o['historical_reproduction'] = diffs
        out[f'{fz}|{sc}'] = o
    return out


def p1e_judge(rows, rho, pairs, steps=None):
    """A1.12 / A7.2: per adjacent rule pair, the rollout bar (worst distance <= 2.5e-4 over cases) and the rho bar (worst
    own-state rho of the coarser rule vs the finer <= 0.116), with validity; descriptive (no qualification label)."""
    out = {}
    for pr in pairs:
        d = [r_['dist'][pr] for r_ in rows if pr in r_['dist']]
        raw = [x for v in (rho.get(pr) or {}).values() for x in (v if isinstance(v, list) else [v])]
        trivial = sum(1 for x in raw if x == 'nan')                       # both vectors zero: excluded, counted
        failed = sum(1 for x in raw if x == 'inf' or x is None)           # non-finite / unknown: fails the bar
        rh = [float(x) for x in raw if isinstance(x, (int, float)) and x is not None]
        a_, b_ = pr.split('|')
        ok = all(r_['valid'].get(a_, False) and r_['valid'].get(b_, False) for r_ in rows)
        out[pr] = dict(worst_distance=SC.dmax(d), rollout_bar=bool(d) and SC.dmax(d) <= SC.BAR_Q2,
                       worst_rho=(float('inf') if failed else (max(rh) if rh else None)), rho_trivial=trivial, rho_failed=failed,
                       rho_bar=bool(rh) and not failed and max(rh) <= SC.BAR_RHO, all_valid=ok)
    if steps:
        out['solver'] = dict(saturated=sum(v['saturated'] for v in steps.values()), reason3=sum(v['reason3'] for v in steps.values()),
                             nonstationary=sum(v['nonstationary'] for v in steps.values()), steps=sum(v['steps'] for v in steps.values()),
                             vendor_bridge_worst=max(v['vendor_bridge_rel'] for v in steps.values()))
    return out


def main():
    cache = {}
    A, accinfo = {}, {}
    for m in MESHES:
        ld = load_acc(m)
        if ld is None:
            continue
        d, r = ld
        A[m] = qualify_all(m, d, r, cache)
        accinfo[m] = r
        print('qualified', m, flush=True)
    res = dict(configurations={}, h_acc_n={}, h_acc_r={}, p1g={}, q3={}, reproduction={}, p1e={})
    for m in A:
        for (name, sc), v in A[m].items():
            res['configurations'][f'{m}|{name}|{sc}'] = dict(Q1=v['q']['Q1'], Q2={k: v['q']['Q2'].get(k) for k in ('label', 'worst_dist_qp', 'rho_max', 'population')},
                                                            numerically_qualified=v['q']['numerically_qualified'],
                                                            e0_median=float(np.median(list(per_case_e(v).values()))),
                                                            e0_worst=float(np.max(list(per_case_e(v).values()))),
                                                            e0_S_median_eligible=float(np.median([x['S']['e'] for x in v['e0'].values() if x['S']['eligible']])) if sc == 'BE' else None,
                                                            S_not_eligible=sorted(c for c, x in v['e0'].items() if sc == 'BE' and not x['S']['eligible']),
                                                            e0_per_case=per_case_e(v),
                                                            e0_median_sub16=float(np.median([e_ for c, e_ in per_case_e(v).items() if c in SC.case_set('sub16')])),
                                                            cases=len(v['e0']), role='role not assessed (P1)', q3=v['q3'],
                                                            budget_exit_steps=v['q']['budget_exit_steps'],
                                                            pooled_nonstationary=v['q']['pooled_nonstationary'])
    for r_ in (128, 256, 384):
        for sc in ('BE', 'CNR'):
            res['h_acc_n'][f'R{r_}|{sc}|frozen'] = h_acc_n(A, r_, sc)
    for sc in ('BE', 'CNR'):
        res['h_acc_n'][f'R384|{sc}|nominal'] = dict(h_acc_n(A, 384, sc, name1024='R{r}', name_other='R{r}nom'),
                                                    label='deployment policy (mode set varies with mesh)')
    for m in A:
        for sc in ('BE', 'CNR'):
            res['h_acc_r'][f'{m}|{sc}'] = h_acc_r(A, m, sc)
    if 1024 in A:
        for name in ('R128', 'R384'):
            for sc in ('BE', 'CNR'):
                res['reproduction'][f'{name}|{sc}'] = SC.reproduction(
                    A[1024][(name, sc)]['R'], name, sc, f'LSPG|{sc}|0.005|prod', accinfo[1024],
                    json.loads((RUNS / 'acc1024' / 'archive' / 'PROVENANCE.json').read_text()))
        res['p1e']['2d'] = accinfo[1024].get('p1e')
        res['order'] = accinfo[1024].get('order')
        res['vendor_bridge'] = accinfo[1024].get('vendor_bridge')
    res['p1g'] = p1g(A)
    td = out_dir('time', 'p1time_time')
    res['timing'] = timing(td, A) if td else 'not reached'
    kd = out_dir('kern2d', 'p1kern_kern2d')
    tok = res['timing']['controls_ok'] if isinstance(res['timing'], dict) else False
    res['kernels_2d'] = kernels(kd, [32, 64, 128, 256, 384, 512], '2D', tok) if kd else 'not reached'
    k3 = out_dir('kern3d', 'p1three_kern3d')
    res['kernels_3d'] = kernels(k3, [64, 128, 256, 512, 768, 1024], '3D', tok) if k3 else 'not reached (deferred or failed)'
    e3 = out_dir('p1e3d', 'p1three_p1e3d')
    res['p1e']['3d'] = json.loads((e3 / 'result.json').read_text())['p1e'] if e3 else 'not reached (deferred or failed)'
    if isinstance(res['p1e']['3d'], dict):
        e = res['p1e']['3d']
        res['p1e']['3d_judged'] = p1e_judge(e['rows'], e.get('rho', {}), list(e['rows'][0]['dist']), e.get('steps'))
    if res['p1e'].get('2d'):
        res['p1e']['2d_judged'] = {}
        for cfg_, v in res['p1e']['2d'].items():
            rows = [dict(r_, valid={k_.replace('main', 'deployed'): x for k_, x in r_['valid'].items()}) for r_ in v['rows']]
            res['p1e']['2d_judged'][cfg_] = p1e_judge(v['rows'], {pr: {str(i): r_['rho'][pr] for i, r_ in enumerate(v['rows'])}
                                                                   for pr in v['rows'][0]['rho']}, list(v['rows'][0]['dist']))
    res['memory'] = {m: dict(setup=accinfo[m]['setup'], memory=accinfo[m]['memory'],
                             configs={k: dict(table_bytes=v['table_bytes'], setting_bytes=v['setting_bytes'], setup_seconds=v['setup_seconds'])
                                      for k, v in accinfo[m]['configs'].items()}, elapsed=accinfo[m].get('elapsed')) for m in accinfo}
    pil = RUNS / 'pilot' / 'archive' / 'output' / 'pilot_pilot' / 'result.json'
    res['pilot'] = {k: v for k, v in json.loads(pil.read_text()).items() if k in ('peak_bytes', 'prefix_retaining_required', 'steps', 'gpu')} if pil.exists() else None
    sm = HERE / 'checks' / 'smoke-accept.json'
    res['smoke'] = json.loads(sm.read_text()) if sm.exists() else None
    res['ledger'] = json.loads((RUNS / 'GPU-LEDGER.json').read_text())
    (HERE / 'checks').mkdir(exist_ok=True)
    (HERE / 'checks' / 'analysis.json').write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
    export = dict(lane='jcp-p1-scaling', status='see report.md', references='E0 R_ST (conditional Richardson eps_ref)',
                  verdicts=dict(h_cost_n={k: v.get('outcome') for k, v in (res['timing']['h_cost_n'].items() if isinstance(res['timing'], dict) else [])},
                                h_cost_r_2d={k: v['outcome'] for k, v in res['kernels_2d']['verdicts'].items()} if isinstance(res['kernels_2d'], dict) else 'not reached',
                                h_cost_r_3d={k: v['outcome'] for k, v in res['kernels_3d']['verdicts'].items()} if isinstance(res['kernels_3d'], dict) else res['kernels_3d'],
                                h_acc_n={k: v['outcome'] for k, v in res['h_acc_n'].items()},
                                h_acc_r={k: {kk: (vv['outcome'] if isinstance(vv, dict) else vv) for kk, vv in v.items()} for k, v in res['h_acc_r'].items()}),
                  qualification={k: v['numerically_qualified'] for k, v in res['configurations'].items()},
                  key_numbers=dict(configurations={k: dict(e0_median=v['e0_median'], e0_worst=v['e0_worst']) for k, v in res['configurations'].items()},
                                   h_cost_n={k: dict(R=v.get('R'), ci=v.get('ci'), medians=v.get('medians')) for k, v in
                                             (res['timing']['h_cost_n'].items() if isinstance(res['timing'], dict) else [])}),
                  contracts=dict(h_cost_n='full-field 2D query, evolution timed', h_cost_r='component kernels at fixed (M, m)'),
                  plots=[])
    (RUNS / 'EXPORT.json').write_text(json.dumps(export, indent=1, default=str))
    print(json.dumps(export['verdicts'], indent=1, default=str))


if __name__ == '__main__':
    main()
