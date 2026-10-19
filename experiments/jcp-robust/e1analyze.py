"""jcp-robust E1 offline analysis, 2D (DESIGN 2; A1.2-A1.5, A2.1-A2.3, A3.1-A3.4, A4.2). CPU only (no JAX).

From one r2acc output directory (one mesh): every distance is recomputed from the saved coefficients with the independent
NumPy decode (npcheck.decode257) and compared with the in-job value; E0 ST errors are scored through refstore with
identity assertions (qscore.score); draw-level qualification, H-E1a, H-E1b, H-E1c, the indicator 2x2 table and the controls
are evaluated by e1stats. Writes checks/e1-2d-<mesh>.json.

    JAX_PLATFORMS=cpu python e1analyze.py <r2acc output dir> [--mesh N] [--repro]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import e1stats as E
import npcheck as NC
import qscore as QSC

HERE = Path(__file__).resolve().parent
TT = 't0,t1,t2,t3,t4,t5'


def fd(fa, fb, n0):
    return NC.fdist(fa, fb, n0)


def agree(off, job):
    if not (np.isfinite(off) and np.isfinite(job)):
        return bool(off == job) or (not np.isfinite(off) and not np.isfinite(job))
    return bool(abs(off - job) <= 1e-9 * abs(job) + 1e-13)


def e0(field, cid, scheme, form, dt):
    coh, c = cid.split('|')
    rec = dict(case_id=cid, time_tag=TT, scheme=scheme, form=form, dt=dt, dt_expected=dt, requested=(coh, int(c)))
    s = QSC.score(field, rec, 'ST')
    return dict(e=float(s['e']), eps=float(s['eps_ref']) if s['eps_ref'] is not None else float('nan'),
                eligible=bool(s['eligible']))


def analyse_arm(out, res, gname, arm, labels, rp):
    """One (group, arm): target certification, every draw's per-case quantities, and the arm-level E1 verdicts."""
    tz = np.load(out / f'T_{gname}_{arm}.npz')
    tcases = [str(x) for x in tz['case_ids']]
    rows = [r for r in res['rows'] if r['group'] == gname and r['arm'] == arm]
    trow = {r['case']: r for r in rows if r['kind'] == 'target'}
    form, scheme, dt = str(tz['form']), str(tz['scheme']), float(tz['dt'])
    FT = NC.decode257(np.asarray(tz['W']).reshape(-1, rp), rp).reshape(len(tcases), 6, 257, 257)
    # ---- target certification (A3.3): every check case valid, distance <= 2.5e-5 (recomputed), target valid per case
    chk_ids = [str(x) for x in tz['check_case_ids']]
    z0 = np.load(out / f'R_{gname}_{labels[0]}_{arm}.npz')
    n0_of = {str(c): float(z0['n0'][j]) for j, c in enumerate(z0['case_ids'])}
    tc_rows = []
    for k, cid in enumerate(chk_ids):
        j = tcases.index(cid)
        fc = NC.decode257(np.asarray(tz['W_check'][k]), rp)
        dc = fd(FT[j], fc, n0_of[cid])
        tc_rows.append(dict(case=cid, check_valid=bool(trow[cid].get('check_valid')), dist=dc, dist_job=trow[cid].get('dist_check'),
                            agree=agree(dc, trow[cid].get('dist_check', float('nan')))))
    econv = []
    for j, cid in enumerate(tcases):
        econv.append(e0(FT[j], cid, scheme, form, dt)['e'] if bool(tz['valid'][j]) else None)
    target_valid = {cid: bool(tz['valid'][j]) for j, cid in enumerate(tcases)}
    out_arm = dict(arm=arm, form=form, scheme=scheme, dt=dt, draws={}, target_valid=target_valid)
    # draws
    per_draw, e_mat, fail_mat, D, med, worst = {}, [], [], [], [], []
    n0s, ecs = {}, {}
    for lab in labels:
        z = np.load(out / f'R_{gname}_{lab}_{arm}.npz')
        ids = [str(x) for x in z['case_ids']]
        assert ids == tcases, 'case order differs from the target'
        rr = {r['case']: r for r in rows if r['label'] == lab and r['kind'] == 'prod'}
        F = NC.decode257(np.asarray(z['W']).reshape(-1, rp), rp).reshape(len(ids), 6, 257, 257)
        level = str(z['level'])
        q = dict(dist_tight=[], dist_tt=[], dist_succ=[], dist_target=[], rho_succ=[], rho_tgt=[], e=[], eps=[], eligible=[],
                 valid=[], valid_tight=[], valid_tighter=[], valid_succ=[], mismatch=[])
        tids = [str(x) for x in z['tighter_case_ids']]
        for j, cid in enumerate(ids):
            n0 = float(z['n0'][j])
            assert np.isfinite(n0) and n0 > 0
            n0s[cid] = n0
            f = F[j]
            sc = e0(f, cid, scheme, form, dt)
            q['e'].append(sc['e']); q['eps'].append(sc['eps']); q['eligible'].append(sc['eligible'])
            q['valid'].append(bool(z['valid'][j]))
            dt_ = fd(f, FT[j], n0)
            q['dist_target'].append(dt_)
            if not agree(dt_, rr[cid].get('dist_target', float('nan'))):
                q['mismatch'].append((cid, 'dist_target', dt_, rr[cid].get('dist_target')))
            q['rho_tgt'].append(float(np.max(z['rho_tgt'][j])))
            if level == 'full':
                ft = NC.decode257(np.asarray(z['W_tight'][j]), rp)
                d1 = fd(f, ft, n0)
                q['dist_tight'].append(d1); q['valid_tight'].append(bool(z['valid_tight'][j]))
                if not agree(d1, rr[cid]['dist_tight']):
                    q['mismatch'].append((cid, 'dist_tight', d1, rr[cid]['dist_tight']))
                if cid in tids:
                    k = tids.index(cid)
                    ftt = NC.decode257(np.asarray(z['W_tighter'][k]), rp)
                    d2 = fd(ft, ftt, n0)
                    q['dist_tt'].append(d2); q['valid_tighter'].append(bool(z['valid_tighter'][k]))
                    if not agree(d2, rr[cid]['dist_tight_tighter']):
                        q['mismatch'].append((cid, 'dist_tt', d2, rr[cid]['dist_tight_tighter']))
                if str(z['succ']):
                    fq = NC.decode257(np.asarray(z['W_succ'][j]), rp)
                    d3 = fd(f, fq, n0)
                    q['dist_succ'].append(d3); q['valid_succ'].append(bool(z['valid_succ'][j]))
                    q['rho_succ'].append(float(np.max(z['rho_succ'][j])))
                    if not agree(d3, rr[cid]['dist_succ']):
                        q['mismatch'].append((cid, 'dist_succ', d3, rr[cid]['dist_succ']))
        nonst = E.pooled_nonstationary([z['cert'][j] for j in range(len(ids))], form)
        tgt_cert = bool(tc_rows and all(r['check_valid'] and r['agree'] for r in tc_rows) and
                        all(np.isfinite(r['dist']) and r['dist'] <= E.BAR_TC for r in tc_rows))
        tgt_ok = (None if not tgt_cert else bool(all(target_valid.values())))
        if level == 'full':
            qual, own, status = E.draw_qualified(
                dict(valid=q['valid'], valid_tight=q['valid_tight'], valid_tighter=q['valid_tighter'], valid_succ=q['valid_succ']),
                dict(tight=q['dist_tight'], tt=q['dist_tt'], succ=q['dist_succ'], rho_succ=q['rho_succ'],
                     target=q['dist_target'], rho_tgt=q['rho_tgt'], nonst_frac=nonst), tgt_ok)
        else:
            qual, own, status = None, None, 'diagnostic (production only)'
        per_draw[lab] = dict(rule=str(z['rule']), rule_sha256=str(z['rule_sha256']), level=level, qualified=qual,
                             own_family_qualified=own, status=status, nonst_frac=nonst, mismatches=q['mismatch'],
                             D=(max(q['dist_target']) if all(np.isfinite(q['dist_target'])) else math.inf),
                             rho_tgt_max=max(q['rho_tgt']), rho_succ_max=max(q['rho_succ']) if q['rho_succ'] else None,
                             dist_tight_max=max(q['dist_tight']) if q['dist_tight'] else None,
                             dist_succ_max=max(q['dist_succ']) if q['dist_succ'] else None,
                             e=q['e'], eps=q['eps'], eligible=q['eligible'], valid=q['valid'], dist_target=q['dist_target'])
    out_arm['target_cert'] = dict(rows=tc_rows, certified=tgt_cert, label='checked on the registered subset (dev6)')
    out_arm['econv'] = econv
    out_arm['draws'] = per_draw
    return out_arm, tcases


def arm_verdicts(a, draw_labels):
    """H-E1a/b/c and the indicator table over the registered draws d0..d15 (A2.2, A3.2-A3.4, A4.2)."""
    dr = [a['draws'][l] for l in draw_labels]
    cases = len(dr[0]['e'])
    elig = [all(d['eligible'][i] for d in dr) for i in range(cases)]
    failed = np.array([[not d['valid'][i] for i in range(cases)] for d in dr])
    e = np.array([[d['e'][i] if d['valid'][i] else np.inf for i in range(cases)] for d in dr])
    eps = np.array(dr[0]['eps'])
    tgt_cert = a['target_cert']['certified']
    econv = [ (a['econv'][i] if (tgt_cert and a['target_valid'][list(a['target_valid'])[i]]) else None) for i in range(cases)]
    h_a = E.h_e1a([d['qualified'] for d in dr])
    h_a_own = E.h_e1a([d['own_family_qualified'] for d in dr])
    ok_e = [i for i in range(cases) if elig[i]]
    delta = 2 * max(eps[i] for i in ok_e) if ok_e else float('nan')
    def agg(fn, d, i_fail):
        xs = [d['e'][i] for i in ok_e]
        return math.inf if any(not d['valid'][i] for i in ok_e) else fn(xs)
    stats = {}
    if tgt_cert:
        stats['D'] = E.rank_interval([d['D'] for d in dr], 0.)
    stats['median_e'] = E.rank_interval([agg(np.median, d, 0) for d in dr], delta)
    stats['worst_e'] = E.rank_interval([agg(np.max, d, 0) for d in dr], delta)
    h_b = E.h_e1b({k: v[2] for k, v in stats.items()})
    h_c = E.h_e1c(e, e[0], failed, eps, econv, elig)
    ind = dict(tp=0, fp=0, fn=0, tn=0)
    for d in dr:
        if not tgt_cert:
            break
        i_ok = d['rho_tgt_max'] <= E.BAR_RHO
        r_ok = d['D'] <= E.BAR_Q2
        ind[('t' if i_ok == r_ok else 'f') + ('p' if i_ok else 'n')] += 1
    d0 = dr[0]
    return dict(H_E1a=h_a, H_E1a_own_family=h_a_own, H_E1b=dict(label=h_b, stats={k: list(v) for k, v in stats.items()},
                delta_error=delta), H_E1c=h_c, indicator=ind if tgt_cert else 'unavailable (target not certified)',
                indicator_draw0=(dict(rho_ok=d0['rho_tgt_max'] <= E.BAR_RHO, rollout_ok=d0['D'] <= E.BAR_Q2) if tgt_cert else None),
                eligible_cases=sum(elig))


P1RUNS = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-17-jcp-p1-scaling/experiments/jcp-p1-scaling/runs')
QLIST = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-qualify/experiments/jcp-qualify/runs')


def archive_sources(mesh, arm):
    """A2.1 archive map: [(path, expected sha256 or None, case ids it supplies)] for draw 0 of arm at mesh."""
    p1 = P1RUNS / f'acc{mesh}' / 'archive' / 'output' / f'p1acc_acc{mesh}'
    if arm in ('LSPG_BE_1', 'LSPG_CNR_1'):
        nm = 'R_R128_BE.npz' if arm == 'LSPG_BE_1' else 'R_R128_CNR.npz'
        src = [(p1 / nm, None)]
        if mesh == 4096:
            src.append((f'qualify:acc4096/output/qacc_acc4096/R_R128_{arm}.npz', None))
        return src
    return [(f'qualify:acc{mesh}/output/qacc_acc{mesh}/R_R128_{arm}.npz', None)]


def qualify_sha(rel):
    """sha256 recorded in the qualify lane's archive listing for a file relative to burgers-archive/jcp-qualify/."""
    job, rest = rel.split('/', 1)
    for line in (QLIST / job / 'archive.sha256-listing').read_text().splitlines():
        h, f = line.split(None, 1)
        if f.strip() == './' + rest:
            return h
    raise KeyError(rel)


def reproduction(out, gname, arm, fetched):
    """G-E1.2 (A2.1): draw 0 W vs the archived production W, per case relative max-norm <= 1e-10; every expected case
    present; file hashes checked against the owning lane's listing (qualify) or recorded (P1)."""
    import hashlib
    z = np.load(out / f'R_{gname}_d0_{arm}.npz')
    ids = [str(x) for x in z['case_ids']]
    got, files = {}, []
    for src, _ in archive_sources(int(z['mesh']), arm):
        if isinstance(src, str):
            rel = src.split(':', 1)[1]
            fp = Path(fetched) / rel
            if not fp.exists():
                return dict(passed=False, reason=f'archive file not fetched: {rel}')
            want = qualify_sha(rel)
        else:
            fp, want = src, None
        h = hashlib.sha256(fp.read_bytes()).hexdigest()
        if want is not None and h != want:
            return dict(passed=False, reason=f'archive hash mismatch {fp}')
        files.append(dict(file=str(fp), sha256=h))
        a = np.load(fp)
        for j, cid in enumerate(a['case_ids']):
            got.setdefault(str(cid), np.asarray(a['W'][j]))
    rows = []
    for j, cid in enumerate(ids):
        if cid not in got:
            return dict(passed=False, reason=f'case {cid} missing from the archives', files=files)
        Wa, Wn = got[cid], np.asarray(z['W'][j])
        den = float(np.max(np.abs(Wa)))
        if not (np.isfinite(Wa).all() and np.isfinite(Wn).all()) or den == 0:
            return dict(passed=False, reason=f'non-finite or zero archive at {cid}', files=files)
        rows.append(float(np.max(np.abs(Wn - Wa)) / den))
    return dict(passed=bool(len(rows) == len(ids) and max(rows) <= 1e-10), worst=max(rows), n=len(rows), files=files)


def controls(res, gname, out, mesh):
    import rules2d
    c = res['groups'][gname].get('controls', {})
    r = {}
    meta = res['groups'][gname]['rules']
    if 'd0' in meta:                                               # G-E1.1: s0 = deployed points, s1 a pure shift
        X0, w0 = rules2d.points('fib4181')
        Xs, ws = rules2d.points('fib4181s0')
        X1, _ = rules2d.points('fib4181s1')
        sh = (X1 - X0) % 1.0
        import hashlib
        r['G-E1.1'] = dict(s0_equals_deployed=bool(np.array_equal(X0, Xs) and np.array_equal(w0, ws)),
                           job_sha_equals_deployed=bool(meta['d0']['meta']['rule_sha256'] == hashlib.sha256(
                               np.ascontiguousarray(np.concatenate([X0.ravel(), w0])).tobytes()).hexdigest()),
                           s1_pure_shift=bool(np.ptp((sh - sh[0] + .5) % 1.0, axis=0).max() < 1e-12))
        r['G-E1.1']['passed'] = all(r['G-E1.1'].values())
    if 'permute' in c:
        r['G-E1.3'] = dict(c['permute'])
    if 'identity' in c:
        r['C-E1.2'] = dict(c['identity'])
    if 'fixture' in c and 'bad' in c['fixture']:
        est = json.loads((HERE / 'checks' / 'establish-e1.json').read_text())
        pan = [p for p in est['panels'] if p['panel'].startswith('2d|')][0]
        r['C-E1.1'] = E.fixture_fired(pan['states'], c['fixture']['bad']['rho_tgt'])
        r['C-E1.1']['cpu'] = [s_['rho_tgt'] for s_ in pan['states']]
        r['C-E1.1']['job'] = c['fixture']['bad']['rho_tgt']
        if 'd0' in c['fixture']:
            r['C-E1.1']['deployed_draw_rho_tgt'] = c['fixture']['d0']['rho_tgt']
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--fetched', default=None, help='directory with qualify archive files (relative layout)')
    a = ap.parse_args()
    out = Path(a.out)
    res = json.loads((out / 'result.json').read_text())
    assert res.get('complete'), 'job output incomplete'
    rep = dict(source=str(out), mesh=res['mesh'], commit=res['commit'], job_id=res['job_id'], gpu=res['gpu'],
               backend=res['backend'], groups={})
    for g in res['config']['groups']:
        gname, rp = g['name'], int(g['Rp'])
        labels = [r['label'] for r in g['rules']]
        draw_labels = [l for l in labels if l.startswith('d')]
        grep = rep['groups'][gname] = dict(arms={}, controls=controls(res, gname, out, res['mesh']))
        for arm in g['arms']:
            ak = f"{arm['form']}_{arm['scheme']}_{float(arm['factor']):g}"
            av, _ = analyse_arm(out, res, gname, ak, labels, rp)
            if draw_labels:
                av['verdicts'] = arm_verdicts(av, draw_labels)
            if 'd0' in labels:
                av['G-E1.2'] = reproduction(out, gname, ak, a.fetched) if a.fetched or ak in ('LSPG_BE_1', 'LSPG_CNR_1') \
                    else dict(passed=False, reason='qualify archives not fetched')
            grep['arms'][ak] = av
            print(gname, ak, json.dumps({k: (v['label'] if isinstance(v, dict) and 'label' in v else v)
                                         for k, v in av.get('verdicts', {}).items()}, default=str), flush=True)
    tag = a.tag or f"e1-2d-{res['mesh']}"
    (HERE / 'checks').mkdir(exist_ok=True)
    (HERE / 'checks' / f'{tag}.json').write_text(json.dumps(rep, indent=1, default=float) + '\n')


if __name__ == '__main__':
    main()
