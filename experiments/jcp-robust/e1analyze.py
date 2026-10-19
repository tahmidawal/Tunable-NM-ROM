"""jcp-robust E1 offline analysis, 2D, one mesh (DESIGN 2; A1.2-A1.5, A2.1-A2.4, A3.1-A3.4, A4.2). CPU only.

From one r2acc output directory: every distance is recomputed from the saved coefficients with the independent NumPy
decode (npcheck.decode257) and compared with the in-job value; E0 ST errors are scored through refstore with identity
assertions (qscore.score); per-case states follow A3.2/A4.2; draw-level qualification, H-E1a/b/c and the indicator table
use e1stats. Integrity evidence (expected population, distance agreement, G-cert, completeness), controls and the
reproduction gate are recorded per arm; e1verdicts.py applies the A1.4/A2.4 precedence across meshes. Also provides the
generic reproduction check used by G-E2.3. Writes checks/<tag>.json.

    JAX_PLATFORMS=cpu python e1analyze.py <r2acc output dir> [--tag e1-2d-<mesh>] [--fetched <dir>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import e1stats as E
import npcheck as NC
import qscore as QSC

HERE = Path(__file__).resolve().parent
TT = 't0,t1,t2,t3,t4,t5'
P1RUNS = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-17-jcp-p1-scaling/experiments/jcp-p1-scaling/runs')
QLIST = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-qualify/experiments/jcp-qualify/runs')
DRAWS = [f'd{s}' for s in range(E.NDRAW)]


def fd(fa, fb, n0):
    return NC.fdist(fa, fb, n0)


def agree(off, job):
    job = E.num(job)
    if not (np.isfinite(off) and np.isfinite(job)):
        return (not np.isfinite(off)) and (not np.isfinite(job))
    return bool(abs(off - job) <= 1e-9 * abs(job) + 1e-13)


def e0(field, cid, scheme, form, dt):
    coh, c = cid.split('|')
    rec = dict(case_id=cid, time_tag=TT, scheme=scheme, form=form, dt=dt, dt_expected=dt, requested=(coh, int(c)))
    s = QSC.score(field, rec, 'ST')
    return dict(e=float(s['e']), eps=float(s['eps_ref']) if s['eps_ref'] is not None else float('nan'),
                eligible=bool(s['eligible']))


def expected_ids(cfg_cases):
    import p1score as PS
    return PS.case_set(cfg_cases if isinstance(cfg_cases, str) else [tuple(x) for x in cfg_cases])


def analyse_arm(out, res, g, arm, rp):
    """One (group, arm) of one mesh: target + certification, every rule's per-case quantities, integrity evidence."""
    gname = g['name']
    labels = [r['label'] for r in g['rules']]
    want_ids = expected_ids(res['config'].get('cases', 'all38'))
    want_t = [c for c in expected_ids(g.get('tighter_cases', 'dev6')) if c in want_ids]
    rows = [r for r in res['rows'] if r['group'] == gname and r['arm'] == arm]
    integ = dict(ids_ok=True, mismatches=[], gcert_fail=0, missing=[])
    has_t = bool(g.get('target'))
    z0 = np.load(out / f'R_{gname}_{labels[0]}_{arm}.npz')
    n0_of = {str(c): float(z0['n0'][j]) for j, c in enumerate(z0['case_ids'])}
    form, scheme, dt = str(z0['form']), str(z0['scheme']), float(z0['dt'])
    out_arm = dict(arm=arm, form=form, scheme=scheme, dt=dt, rules={}, diagnostic=(form == 'GAL'))
    FT, tvalid, econv, tc_rows, tgt_cert = None, None, [None] * len(want_ids), [], False
    if has_t:
        tz = np.load(out / f'T_{gname}_{arm}.npz')
        tcases = [str(x) for x in tz['case_ids']]
        if tcases != want_ids:
            integ['ids_ok'] = False
        trow = {r['case']: r for r in rows if r['kind'] == 'target'}
        FT = NC.decode257(np.asarray(tz['W']).reshape(-1, rp), rp).reshape(len(tcases), 6, 257, 257)
        tvalid = [bool(x) for x in tz['valid']]
        chk_ids = [str(x) for x in tz['check_case_ids']]
        for k, cid in enumerate(chk_ids):
            j = tcases.index(cid)
            dc = fd(FT[j], NC.decode257(np.asarray(tz['W_check'][k]), rp), n0_of[cid])
            tc_rows.append(dict(case=cid, check_valid=bool(trow[cid].get('check_valid')), dist=dc, dist_job=trow[cid].get('dist_check'),
                                agree=agree(dc, trow[cid].get('dist_check'))))
        want_chk = [c for c in expected_ids(g.get('check_cases', 'dev6')) if c in want_ids] if g.get('check') else []
        tgt_cert = bool(g.get('check') and [r['case'] for r in tc_rows] == want_chk and
                        all(r['check_valid'] and r['agree'] and np.isfinite(r['dist']) and r['dist'] <= E.BAR_TC for r in tc_rows))
        econv = [e0(FT[j], cid, scheme, form, dt)['e'] if (tvalid[j] and tgt_cert) else None for j, cid in enumerate(tcases)]
    # target state for draw qualification (A3.3, A4.2): None = not certified; False = certified but a target run failed
    tgt_ok = (None if not tgt_cert else bool(all(tvalid)))
    out_arm['target'] = dict(present=has_t, certified=tgt_cert, all_valid=(all(tvalid) if tvalid else None), check_rows=tc_rows,
                             label='checked on the registered subset (dev6)', econv=econv)
    for r in g['rules']:
        lab = r['label']
        z = np.load(out / f'R_{gname}_{lab}_{arm}.npz')
        ids = [str(x) for x in z['case_ids']]
        tids = [str(x) for x in z['tighter_case_ids']]
        if ids != want_ids or (r.get('level', 'full') == 'full' and tids != want_t):
            integ['ids_ok'] = False
        rr = {x['case']: x for x in rows if x['label'] == lab and x['kind'] == 'prod'}
        if sorted(rr) != sorted(ids):
            integ['missing'].append((lab, 'rows'))
        F = NC.decode257(np.asarray(z['W']).reshape(-1, rp), rp).reshape(len(ids), 6, 257, 257)
        full = str(z['level']) == 'full'
        q = {k: [] for k in ('tight', 'tt', 'succ', 'target', 'rho_succ', 'rho_tgt', 'e', 'eps', 'eligible', 'valid',
                             'valid_tight', 'valid_tighter', 'valid_succ')}
        for j, cid in enumerate(ids):
            n0 = float(z['n0'][j])
            assert np.isfinite(n0) and n0 > 0, 'normaliser'
            f = F[j]
            q['valid'].append(bool(z['valid'][j]))
            sc = e0(f, cid, scheme, form, dt)
            q['e'].append(sc['e'] if q['valid'][-1] else math.inf)
            q['eps'].append(sc['eps'])
            q['eligible'].append(sc['eligible'])
            if not rr.get(cid, {}).get('gcert_ok', False):
                integ['gcert_fail'] += 1
            if has_t:
                d_ = fd(f, FT[j], n0) if tvalid[j] else math.inf
                q['target'].append(d_)
                if tvalid[j] and not agree(d_, rr.get(cid, {}).get('dist_target')):
                    integ['mismatches'].append((lab, cid, 'target'))
                q['rho_tgt'].append(float(np.max(z['rho_tgt'][j])) if np.isfinite(z['rho_tgt'][j]).all() else math.inf)
            if full:
                ft = NC.decode257(np.asarray(z['W_tight'][j]), rp)
                d1 = fd(f, ft, n0)
                q['tight'].append(d1)
                q['valid_tight'].append(bool(z['valid_tight'][j]))
                if not agree(d1, rr.get(cid, {}).get('dist_tight')):
                    integ['mismatches'].append((lab, cid, 'tight'))
                if cid in tids:
                    k = tids.index(cid)
                    d2 = fd(ft, NC.decode257(np.asarray(z['W_tighter'][k]), rp), n0)
                    q['tt'].append(d2)
                    q['valid_tighter'].append(bool(z['valid_tighter'][k]))
                    if not agree(d2, rr.get(cid, {}).get('dist_tight_tighter')):
                        integ['mismatches'].append((lab, cid, 'tt'))
                if str(z['succ']):
                    d3 = fd(f, NC.decode257(np.asarray(z['W_succ'][j]), rp), n0)
                    q['succ'].append(d3)
                    q['valid_succ'].append(bool(z['valid_succ'][j]))
                    q['rho_succ'].append(float(np.max(z['rho_succ'][j])) if np.isfinite(z['rho_succ'][j]).all() else math.inf)
                    if not agree(d3, rr.get(cid, {}).get('dist_succ')):
                        integ['mismatches'].append((lab, cid, 'succ'))
        nonst = E.pooled_nonstationary([z['cert'][j] for j in range(len(ids))], form)
        if full and str(z['succ']):
            qual, own, status = E.draw_qualified(
                dict(valid=q['valid'], valid_tight=q['valid_tight'], valid_tighter=q['valid_tighter'], valid_succ=q['valid_succ']),
                dict(tight=q['tight'], tt=q['tt'], succ=q['succ'], rho_succ=q['rho_succ'],
                     target=q['target'] if has_t else None, rho_tgt=q['rho_tgt'] if has_t else None, nonst_frac=nonst),
                tgt_ok if has_t else None, len(want_ids), len(want_t))
        else:
            qual, own, status = None, None, 'diagnostic (production only)'
        prod_failed = not all(q['valid'])
        if not has_t or tgt_ok is not True:
            D = None                                                     # target-dependent quantity unavailable
        else:
            D = math.inf if prod_failed else max(q['target'])
        out_arm['rules'][lab] = dict(rule=str(z['rule']), succ=str(z['succ']), rule_sha256=str(z['rule_sha256']),
                                     level=str(z['level']), qualified=qual, own_family_qualified=own, status=status,
                                     nonst_frac=nonst, D=D, prod_failed=prod_failed,
                                     rho_tgt_max=(max(q['rho_tgt']) if has_t else None),
                                     rho_succ_max=(max(q['rho_succ']) if q['rho_succ'] else None),
                                     tight_max=(max(q['tight']) if q['tight'] else None),
                                     succ_max=(max(q['succ']) if q['succ'] else None),
                                     e=q['e'], eps=q['eps'], eligible=q['eligible'], valid=q['valid'],
                                     case_q=([bool(q['valid'][i] and q['valid_tight'][i] and q['valid_succ'][i]
                                                   and q['tight'][i] <= E.BAR_Q1 and q['succ'][i] <= E.BAR_Q2
                                                   and q['rho_succ'][i] <= E.BAR_RHO) for i in range(len(ids))]
                                             if (full and str(z['succ'])) else None),
                                     dist_target=q['target'] if has_t else None)
    out_arm['integrity'] = dict(integ, ok=bool(integ['ids_ok'] and not integ['mismatches'] and integ['gcert_fail'] == 0
                                              and not integ['missing']))
    return out_arm


def arm_verdicts(a):
    """H-E1a/b/c and the indicator table over the registered draws d0..d15 (A2.2, A3.2-A3.4, A4.2)."""
    if any(l not in a['rules'] for l in DRAWS):
        return dict(population='unavailable (registered draws d0..d15 not all present)')
    dr = [a['rules'][l] for l in DRAWS]
    n = len(dr[0]['e'])
    elig = [all(d['eligible'][i] for d in dr) for i in range(n)]
    ok_e = [i for i in range(n) if elig[i]]
    t = a['target']
    tgt_full = bool(t['present'] and t['certified'] and t['all_valid'])
    h_a = E.h_e1a([d['qualified'] for d in dr]) if tgt_full else \
        dict(E.h_e1a([d['own_family_qualified'] for d in dr]), basis='Q1 + own-family Q2 only (target not certified / failed)')
    if not ok_e:
        return dict(H_E1a=h_a, H_E1b=dict(label='unavailable'), H_E1c=dict(label='unavailable', n=0), indicator='unavailable',
                    eligible_cases=0)
    eps = np.array(dr[0]['eps'], float)
    delta = 2 * float(np.max(eps[ok_e]))                                  # A3.4: uniform bound
    agg = lambda fn, d: math.inf if any(not d['valid'][i] for i in ok_e) else float(fn([d['e'][i] for i in ok_e]))
    stats = {}
    if tgt_full:
        stats['D'] = E.rank_interval([d['D'] for d in dr], 0.)
    stats['median_e'] = E.rank_interval([agg(np.median, d) for d in dr], delta)
    stats['worst_e'] = E.rank_interval([agg(np.max, d) for d in dr], delta)
    failed = np.array([[not d['valid'][i] for i in range(n)] for d in dr])
    emat = np.array([d['e'] for d in dr], float)
    econv = t['econv'] if tgt_full else [None] * n
    h_c = E.h_e1c(emat, emat[0], failed, eps, econv, elig)
    if not tgt_full:
        h_c['margin'] = '2 eps_ref only (target-free fallback)'
    ind = 'unavailable (target not certified / failed)'
    if tgt_full:
        ind = dict(tp=0, fp=0, fn=0, tn=0)
        for d in dr:
            i_ok, r_ok = d['rho_tgt_max'] <= E.BAR_RHO, d['D'] <= E.BAR_Q2
            ind[('t' if i_ok == r_ok else 'f') + ('p' if i_ok else 'n')] += 1
        ind['draw0'] = dict(rho_ok=dr[0]['rho_tgt_max'] <= E.BAR_RHO, rollout_ok=dr[0]['D'] <= E.BAR_Q2)
    return dict(H_E1a=h_a, H_E1b=dict(label=E.h_e1b({k: v[2] for k, v in stats.items()}),
                                       stats={k: list(v) for k, v in stats.items()}, delta_error=delta),
                H_E1c=h_c, indicator=ind, eligible_cases=len(ok_e))


# ------------------------------------------------------------------ reproduction (G-E1.2, G-E2.3; A2.1, A3.6)

def pinned_sha(src):
    """sha256 pinned by the owning lane's own listing: ('p1', mesh, name) or ('qualify', job, name)."""
    lane, job, name = src
    if lane == 'p1':
        lst, key = P1RUNS / job / 'archive' / 'OUTPUTS.sha256', f'output/p1acc_{job}/{name}'
    else:
        lst, key = QLIST / job / 'archive.sha256-listing', f'./output/qacc_{job}/{name}'
    for line in lst.read_text().splitlines():
        h, f = line.split(None, 1)
        if f.strip() == key:
            return h
    raise KeyError(src)


def source_path(src, fetched):
    lane, job, name = src
    if lane == 'p1':
        return P1RUNS / job / 'archive' / 'output' / f'p1acc_{job}' / name
    return Path(fetched) / job / 'output' / f'qacc_{job}' / name if fetched else None


def e1_sources(mesh, arm):
    """A2.1 archive map for draw 0 of the R128 fib4181 arms."""
    if arm in ('LSPG_BE_1', 'LSPG_CNR_1'):
        nm = 'R_R128_BE.npz' if arm == 'LSPG_BE_1' else 'R_R128_CNR.npz'
        src = [('p1', f'acc{mesh}', nm)]
        if mesh == 4096:
            src.append(('qualify', 'acc4096', f'R_R128_{arm}.npz'))
        return src
    return [('qualify', f'acc{mesh}', f'R_R128_{arm}.npz')]


def reproduce(W_new, ids, sources, fetched, want_ids):
    """Per case relative max-norm of W_new vs the archived production W (<= 1e-10); every expected case present; each
    archive file's sha256 equals its owning lane's pin; finite, nonzero archives."""
    if list(ids) != list(want_ids):
        return dict(passed=False, reason='new output does not hold the registered population')
    got, files = {}, []
    for src in sources:
        fp = source_path(src, fetched)
        if fp is None or not fp.exists():
            return dict(passed=False, reason=f'archive file unavailable: {src}')
        h = hashlib.sha256(fp.read_bytes()).hexdigest()
        if h != pinned_sha(src):
            return dict(passed=False, reason=f'archive hash differs from the pin: {src}')
        files.append(dict(src=list(src), sha256=h))
        a = np.load(fp)
        for j, cid in enumerate(a['case_ids']):
            got.setdefault(str(cid), np.asarray(a['W'][j]))
    rows = []
    for j, cid in enumerate(ids):
        if cid not in got:
            return dict(passed=False, reason=f'case {cid} missing from the archives', files=files)
        Wa, Wn = got[cid], np.asarray(W_new[j])
        den = float(np.max(np.abs(Wa))) if np.isfinite(Wa).all() else 0.
        if den == 0 or Wa.shape != Wn.shape:
            return dict(passed=False, reason=f'zero, non-finite or mis-shaped archive at {cid}', files=files)
        rows.append(float(np.max(np.abs(Wn - Wa)) / den) if np.isfinite(Wn).all() else math.inf)
    return dict(passed=bool(rows and max(rows) <= 1e-10), worst=max(rows), n=len(rows), files=files)


def controls(res, g, mesh):
    import rules2d
    gname = g['name']
    c = res['groups'][gname].get('controls', {})
    meta = res['groups'][gname]['rules']
    r = {}
    if 'd0' in meta and 'd1' in meta:                              # G-E1.1
        X0, w0 = rules2d.points('fib4181')
        Xs, ws = rules2d.points('fib4181s0')
        X1, _ = rules2d.points('fib4181s1')
        sh = (X1 - X0) % 1.0
        dep = hashlib.sha256(np.ascontiguousarray(np.concatenate([X0.ravel(), w0])).tobytes()).hexdigest()
        r['G-E1.1'] = dict(s0_equals_deployed=bool(np.array_equal(X0, Xs) and np.array_equal(w0, ws)),
                           job_sha_equals_deployed=bool(meta['d0']['meta']['rule_sha256'] == dep),
                           s1_pure_shift=bool(np.ptp((sh - sh[0] + .5) % 1.0, axis=0).max() < 1e-12),
                           s1_differs=bool(not np.array_equal(X1, X0)))
        r['G-E1.1']['passed'] = all(r['G-E1.1'].values())
    if g.get('permute'):
        p = c.get('permute')
        r['G-E1.3'] = dict(p, passed=bool(p and p.get('passed'))) if p else dict(passed=False, reason='missing')
    if g.get('identity'):
        i_ = c.get('identity')
        r['C-E1.2'] = dict(i_, fired=bool(i_ and i_.get('raised'))) if i_ else dict(fired=False, reason='missing')
    if g.get('fixture'):
        ef = HERE / 'checks' / 'establish-e1.json'
        pans = [p for p in json.loads(ef.read_text())['panels'] if p['panel'].startswith('2d|')] if ef.exists() else []
        job = c.get('fixture', {}).get('bad', {}).get('rho_tgt')
        if not pans:
            r['C-E1.1'] = dict(admitted=False, fired=False, reason='establishment missing (control unavailable)')
        else:
            r['C-E1.1'] = E.fixture_fired(pans[0]['states'], job)
            r['C-E1.1'].update(cpu=[s_['rho_tgt'] for s_ in pans[0]['states']], job=job,
                               deployed_draw_rho_tgt=c.get('fixture', {}).get('d0', {}).get('rho_tgt'))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--fetched', default=None, help='directory holding fetched qualify archive files (<job>/output/...)')
    a = ap.parse_args()
    out = Path(a.out)
    res = json.loads((out / 'result.json').read_text())
    import e2bank
    e2bank.use(res['config'].get('bank'))
    mesh = int(res['mesh'])
    rep = dict(source=str(out), mesh=mesh, commit=res['commit'], job_id=res['job_id'], gpu=res['gpu'], backend=res['backend'],
               complete=bool(res.get('complete')) and (out / 'COMPLETE').exists(), groups={})
    for g in res['config']['groups']:
        rp = int(g['Rp'])
        grep = rep['groups'][g['name']] = dict(arms={}, controls=controls(res, g, mesh))
        for arm in g['arms']:
            ak = f"{arm['form']}_{arm['scheme']}_{float(arm['factor']):g}"
            av = analyse_arm(out, res, g, ak, rp)
            if not av['diagnostic'] and all(l in av['rules'] for l in DRAWS):
                av['verdicts'] = arm_verdicts(av)
            elif av['diagnostic']:
                av['verdicts'] = 'diagnostic (GAL-CN): no verdict'
            if 'd0' in av['rules'] and g['name'] == 'R128' and not res['config'].get('bank'):
                z = np.load(out / f"R_{g['name']}_d0_{ak}.npz")
                av['G-E1.2'] = reproduce(z['W'], [str(x) for x in z['case_ids']], e1_sources(mesh, ak), a.fetched,
                                         expected_ids(res['config'].get('cases', 'all38')))
            grep['arms'][ak] = av
            v = av.get('verdicts')
            print(g['name'], ak, json.dumps({k: (x['label'] if isinstance(x, dict) and 'label' in x else x)
                                             for k, x in v.items()} if isinstance(v, dict) else v, default=str),
                  'integrity', av['integrity']['ok'], flush=True)
    tag = a.tag or f'e1-2d-{mesh}'
    (HERE / 'checks').mkdir(exist_ok=True)
    (HERE / 'checks' / f'{tag}.json').write_text(json.dumps(rep, indent=1, default=float) + '\n')


if __name__ == '__main__':
    main()
