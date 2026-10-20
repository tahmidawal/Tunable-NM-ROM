"""jcp-qualify offline scoring and qualification (DESIGN 2.5, 4, A1.3; CPU only, no JAX). Generalises P1's p1score.py
(copied unchanged; its E0 scoring, identity assertions, NumPy rule tables and reproduction helpers are reused) to any
step size and to the GAL form.

  * E0 scoring with per-record identity assertions, now including the step size (Delta-t swap control);
  * offline recomputation of every Q1 / Q2 distance with the independent NumPy decode (npcheck.decode257) and of rho
    with independent NumPy rule tables, compared with the in-job values (A1.3 tolerance);
  * configuration-level Q1 / Q2 labels (DESIGN 4.1 as amended by A1.3, 4.2);
  * reproduction gates: against P1's archives (Delta t0) and combined's c2grid archives (any Delta t, both forms).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import sys
import npcheck as NC
import p1score as PS
_E0 = str(PS.E0)                       # p1score puts the references lane first on sys.path; move it last so that its
if _E0 in sys.path:                    # make_configs / make_report / collect never shadow this lane's modules
    sys.path.remove(_E0)
    sys.path.append(_E0)
from p1score import RS, IdentityError, agree, case_set, dmax, finite, rule_tables  # noqa: F401

HERE = Path(__file__).resolve().parent
TIME_TAG = PS.TIME_TAG
BAR_Q1, BAR_TT, BAR_Q2, BAR_RHO, GAMMA, ETA = 2.5e-4, 2.5e-5, 2.5e-4, .116, 1e-3, 1e-8
DT0 = .005
P1 = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-17-jcp-p1-scaling/experiments/jcp-p1-scaling')
COMB = PS.COMB


def certified(cert, form):
    """Per-step certificate classification (A1.3): LSPG gamma <= 1e-3; GAL eta <= 1e-8; non-finite never certified."""
    c = np.asarray(cert, float)
    return np.isfinite(c) & (c <= (GAMMA if form == 'LSPG' else ETA))


def gcert_ok(R, form, tolf):
    """G-cert recomputed offline (A1.3, A2.5; independent of qcert): LSPG gamma vs recorded gamma; GAL eta vs
    tolratio * tolf for positive scale, raw residual vs tolratio * 1e-300 for zero scale."""
    c = np.asarray(R['cert'], float)
    if not c.size:
        return False
    if form == 'LSPG':
        r = np.asarray(R['rec'], float)
        return bool(np.all(np.isfinite(c) & np.isfinite(r) & (np.abs(c - r) <= 1e-7 * np.abs(r) + 1e-12)))
    tr, sc, raw = (np.asarray(R[k], float) for k in ('tolratio', 'scale', 'raw'))
    pos = sc > 0
    ok_pos = np.isfinite(c) & np.isfinite(tr) & (np.abs(c - tr * tolf) <= 1e-6 * np.abs(tr * tolf) + 1e-13)
    ok_zero = np.isfinite(tr) & np.isfinite(raw) & (np.abs(raw - tr * 1e-300) <= 1e-6 * np.abs(tr * 1e-300) + 1e-15)
    return bool(np.all(np.where(pos, ok_pos, ok_zero)))


def score(field, rec, kind='ST'):
    """P1 score() plus the Delta-t identity: rec['dt'] must equal the configuration's dt (rec['dt_expected'])."""
    if abs(float(rec['dt']) - float(rec['dt_expected'])) > 1e-15:
        raise IdentityError(f"Delta-t swap: record dt {rec['dt']} scored as dt {rec['dt_expected']}")
    if kind == 'S' and not (rec.get('form', 'LSPG') == 'LSPG' and rec['scheme'] == 'BE' and abs(rec['dt'] - DT0) < 1e-15):
        raise IdentityError('reference-kind swap: S is matched only to LSPG-BE at dt0 = 0.005')
    return PS.score(field, dict(rec, scheme=rec['scheme'] if rec.get('form', 'LSPG') == 'LSPG' else 'GAL' + rec['scheme']), kind)


def score_config(R, dt_expected, kinds=('ST',)):
    ids = [str(x) for x in R['case_ids']]
    scheme, form = str(R['scheme']), str(R['form']) if 'form' in R else 'LSPG'
    res = {}
    for j, cid in enumerate(ids):
        coh, c = cid.split('|')
        rec = dict(case_id=cid, time_tag=str(R['time_tag']), scheme=scheme, form=form, dt=float(R['dt']),
                   dt_expected=dt_expected, requested=(coh, int(c)))
        res[cid] = {k: score(R['f257'][j], rec, k) for k in kinds
                    if k == 'ST' or (form == 'LSPG' and scheme == 'BE' and abs(float(R['dt']) - DT0) < 1e-15)}
    return res


def n0_of(R, j):
    n0 = float(R['n0'][j])
    if not (np.isfinite(n0) and n0 > 0):
        raise ValueError('non-positive or non-finite normalisation')
    return n0


def qualify(R, rows, expect, rho_tables=None):
    """Configuration-level Q1 / Q2 (DESIGN 4) from the saved arrays and the in-job rows. expect: dict(cases, dev, q2,
    qpp, form, steps). rho_tables: (t_q, t_qp) NumPy tables for the independent rho on the dev cases."""
    form, S = expect['form'], int(expect['steps'])
    Rp = int(R['Rp'])
    ids = [str(x) for x in R['case_ids']]
    out = dict(per_case={}, disagreements=[])
    if ids != list(expect['cases']):
        out['disagreements'].append(('population', ids, list(expect['cases'])))
    n = len(ids)
    for k, shp in (('W', (n, 6, Rp)), ('W_tight', (n, 6, Rp)), ('states', (n, S + 1, Rp)), ('cert', (n, S)),
                   ('rec', (n, S)), ('reason', (n, S)), ('f257', (n, 6, 257, 257)), ('w0', (n, Rp))):
        if tuple(np.shape(R[k])) != shp:
            out['disagreements'].append(('shape', k, np.shape(R[k]), shp))
    if str(R['form']) != form or int(R['steps']) != S:
        out['disagreements'].append(('form/steps', str(R['form']), int(R['steps'])))
    dec = lambda W: NC.decode257(W, Rp)
    rowby = {f"{r_['cohort']}|{r_['case']}": r_ for r_ in rows}
    ix = {cid: j for j, cid in enumerate(ids)}

    def cmp(cid, key, off):
        r_ = rowby.get(cid)
        if r_ is None or r_.get(key) is None:
            out['disagreements'].append((cid, key, 'missing in-job value'))
        elif not agree(off, r_[key]):
            out['disagreements'].append((cid, key, off, r_[key]))

    cert = np.asarray(R['cert'], float)
    nonc = int(np.sum(~certified(cert, form)))
    out.update(pooled_noncertified=nonc, pooled_steps=int(cert.size), gcert_all=gcert_ok(R, form, float(np.asarray(R['tol'])[1])),
               budget_exit_steps=int(np.sum(np.asarray(R['reason']) == 0)), certificate=('gamma' if form == 'LSPG' else 'eta'))
    if form == 'GAL' and 'gamma_lit' in R:
        g = np.asarray(R['gamma_lit'], float)
        out['gal_literal_gamma'] = dict(median=float(np.nanmedian(g)), max=float(np.nanmax(g)), above_1e3=int(np.sum(~(g <= 1e-3))))
        jc = np.asarray(R['jcond'], float)
        out['gal_step_jacobian_cond_max'] = float(np.nanmax(jc))
    for j, cid in enumerate(ids):
        n0 = n0_of(R, j)
        fp = dec(R['W'][j])
        rec_err = float(np.max(np.abs(fp - R['f257'][j])) / np.max(np.abs(R['f257'][j])))
        d_t = NC.fdist(fp, dec(R['W_tight'][j]), n0)
        valid = bool(finite(R['W'][j], R['w0'][j], R['states'][j], R['f257'][j])
                     and not np.any(np.asarray(R['reason'][j]) == 3) and bool(R['valid'][j]))
        out['per_case'][cid] = dict(valid=valid, tight_valid=bool(R['valid_tight'][j] and finite(R['W_tight'][j])),
                                    dist_tight=d_t, decode_rel=rec_err)
        if not rec_err <= 1e-12:
            out['disagreements'].append((cid, 'decode', rec_err))
        cmp(cid, 'dist_tight', d_t)
    devc = [str(x) for x in R['dev_case_ids']]
    if devc != list(expect['dev']):
        out['disagreements'].append(('dev population', devc, list(expect['dev'])))
    tt = []
    for k, cid in enumerate(devc):
        j = ix[cid]
        d_tt = NC.fdist(dec(R['W_tight'][j]), dec(R['W_tighter'][k]), n0_of(R, j))
        out['per_case'][cid].update(dist_tight_tighter=d_tt, tighter_valid=bool(R['valid_tighter'][k] and finite(R['W_tighter'][k])))
        cmp(cid, 'dist_tight_tighter', d_tt)
        tt.append(d_tt)
    q2case = [str(x) for x in R['q2_case_ids']]
    if q2case != list(expect['q2']):
        out['disagreements'].append(('q2 population', q2case, list(expect['q2'])))
    q2 = None
    if expect['q2']:
        if rho_tables is None:
            raise ValueError('Q2 needs the independent rho recomputation (P1 code audit 3)')
        dq, valid_qp, conv, valid_qpp = [], [], [], []
        rho_job = np.asarray(R['rho'], float)
        for k, cid in enumerate(q2case):
            j = ix[cid]
            d = NC.fdist(dec(R['W'][j]), dec(R['W_qp'][k]), n0_of(R, j))
            dq.append(d)
            out['per_case'][cid]['dist_qp'] = d
            cmp(cid, 'dist_qp', d)
            valid_qp.append(bool(R['valid_qp'][k] and finite(R['W_qp'][k])))
            if cid in expect['dev']:
                for kk in range(6):
                    off = NC.rho(np.asarray(R['W'][j, kk]), *rho_tables)
                    if not (np.isnan(off) and np.isnan(rho_job[k, kk])) and not agree(off, rho_job[k, kk]):
                        out['disagreements'].append((cid, f'rho[{kk}]', off, float(rho_job[k, kk])))
        if expect.get('qpp'):
            for k, cid in enumerate(devc):
                kq, j = q2case.index(cid), ix[cid]
                c_ = NC.fdist(dec(R['W_qp'][kq]), dec(R['W_qpp'][k]), n0_of(R, j))
                conv.append(c_)
                out['per_case'][cid]['dist_qp_qpp'] = c_
                cmp(cid, 'dist_qp_qpp', c_)
                valid_qpp.append(bool(R['valid_qpp'][k] and finite(R['W_qpp'][k])))
        rho = rho_job[:len(q2case)].ravel()
        rho_ok = rho[~np.isnan(rho)]
        q2 = dict(population=len(q2case), rollout=bool(dq) and dmax(dq) <= BAR_Q2, worst_dist_qp=dmax(dq),
                  rho_max=dmax(rho_ok) if rho_ok.size else None, rho_trivial=int(np.sum(np.isnan(rho))),
                  rho_infinite=int(np.sum(np.isinf(rho))), rho=bool(rho_ok.size) and dmax(rho_ok) <= BAR_RHO,
                  successor_valid=bool(valid_qp) and all(valid_qp),
                  check_valid=(not expect.get('qpp')) or (bool(valid_qpp) and all(valid_qpp)),
                  convergence=(not expect.get('qpp')) or (bool(conv) and dmax(conv) <= BAR_Q2),
                  worst_dist_qp_qpp=dmax(conv) if conv else None)
    q1 = dict(all_valid=bool(out['per_case']) and all(v['valid'] and v['tight_valid'] for v in out['per_case'].values()),
              certificate=bool(cert.size == n * S) and nonc <= .01 * cert.size, gcert=out['gcert_all'],
              distance=bool(out['per_case']) and dmax(v['dist_tight'] for v in out['per_case'].values()) <= BAR_Q1,
              tight_tighter=(not expect['dev']) or (bool(tt) and len(tt) == len(expect['dev']) and dmax(tt) <= BAR_TT and
                                                    all(out['per_case'][c]['tighter_valid'] for c in devc)))
    q1['offline_agrees'] = not out['disagreements']
    q1['pass'] = all(q1.values())
    out['Q1'] = q1
    if q2 is None:
        out['Q2'] = dict(label='not run here')
    else:
        q2['label'] = q2_label(q2, bool(out['disagreements']))
        out['Q2'] = q2
    out['numerically_qualified'] = bool(out['Q1']['pass'] and out['Q2'].get('label') == 'pass')
    return out


def q2_label(q2, disagreement=False):
    """Q2 label precedence (DESIGN 4.2; P1 A1.13, A2.4). q2: dict(successor_valid, check_valid, rho_max, rollout, rho,
    convergence). Used by qualify and by the smoke's under-resolved control (the same classifier)."""
    if disagreement:
        return 'fail (offline disagreement)'
    if not (q2['successor_valid'] and q2['check_valid']):
        return 'unverified (successor invalid)'
    if q2['rho_max'] is None:
        return 'unverified (no rho indicators)'
    if not (q2['rollout'] and q2['rho']):
        return 'fail'
    if not q2['convergence']:
        return 'unverified (finer rule not converged)'
    return 'pass'


def q2_from_runs(W, W_succ, rho, n0, Rp, succ_valid):
    """The Q2 predicate on raw arrays (offline decode): used for the under-resolved control (no check rule there, so
    convergence is reported as not applicable and does not gate the fail outcome)."""
    d = [NC.fdist(NC.decode257(W[i], Rp), NC.decode257(W_succ[i], Rp), n0[i]) for i in range(len(W))]
    r = np.asarray(rho, float).ravel()
    r = r[~np.isnan(r)]
    q2 = dict(rollout=dmax(d) <= BAR_Q2, worst_dist_qp=dmax(d), rho_max=dmax(r) if r.size else None,
              rho=bool(r.size) and dmax(r) <= BAR_RHO, successor_valid=bool(succ_valid), check_valid=True, convergence=True)
    q2['label'] = q2_label(q2)
    return q2


def combined_reproduction(R, rp, form, scheme, dt):
    """DESIGN 7 'reproduction gate (Delta t, form)': W vs combined's c2grid archive run '<form>|<scheme>|<dt>|prod' at
    1024^2 (R128 fib4181, R384 gauss128, R512 gauss192)."""
    z = np.load(COMB / f'W_R{rp}.npz')
    runs, ac = [str(x) for x in z['runs']], [str(x) for x in z['cases']]
    key = f'{form}|{scheme}|{dt:.10g}|prod'
    if key not in runs:
        cand = [r_ for r_ in runs if r_.startswith(f'{form}|{scheme}|') and r_.endswith('|prod') and abs(float(r_.split('|')[2]) - dt) < 1e-9]
        assert len(cand) == 1, (key, cand)
        key = cand[0]
    ri = runs.index(key)
    rows = []
    for j, cid in enumerate([str(x) for x in R['case_ids']]):
        Wa = z['W'][ac.index(cid), ri]
        rows.append(dict(case=cid, rel=float(np.max(np.linalg.norm(R['W'][j] - Wa, axis=1)) / np.linalg.norm(Wa[0]))))
    worst = dmax(r_['rel'] for r_ in rows)
    return dict(archive_key=key, rows=rows, worst=worst, passed=bool(worst <= 1e-10))


def p1_reproduction(R, name, scheme, mesh):
    """Lane driver vs P1's own archive of the same configuration (Delta t0), per case present in both."""
    z = np.load(P1 / 'runs' / f'acc{mesh}' / 'archive' / 'output' / f'p1acc_acc{mesh}' / f'R_{name}_{scheme}.npz')
    pc = [str(x) for x in z['case_ids']]
    rows = []
    for j, cid in enumerate([str(x) for x in R['case_ids']]):
        if cid in pc:
            Wa = z['W'][pc.index(cid)]
            rows.append(dict(case=cid, rel=float(np.max(np.linalg.norm(R['W'][j] - Wa, axis=1)) / np.linalg.norm(Wa[0]))))
    worst = dmax(r_['rel'] for r_ in rows) if rows else float('inf')
    return dict(rows=rows, worst=worst, passed=bool(rows) and worst <= 1e-10)


def zero_update_distance(R, j):
    """Zero-update control (DESIGN 7, A1.3): the constant trajectory c_k = c_0 against the saved tight trajectory, and the
    tight trajectory's own motion ||u5 - u0|| / ||u0|| (independent establishment)."""
    Rp = int(R['Rp'])
    n0 = n0_of(R, j)
    ft = NC.decode257(R['W_tight'][j], Rp)
    fz = NC.decode257(np.repeat(R['W'][j][:1], 6, 0), Rp)
    return dict(dist=NC.fdist(fz, ft, n0), motion=float(np.linalg.norm(ft[5] - ft[0]) / n0))


def load_json(p):
    return json.loads(Path(p).read_text())
