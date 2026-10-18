"""jcp-heads offline scoring and qualification (DESIGN 3, 4, A1.3, A1.4; CPU only, no JAX).

  * E0 scoring through qualify's qscore.score_config (identity assertions incl. dt; S only for LSPG-BE at dt0);
  * configuration-level Q1 / Q2 from the saved arrays of an ACCURACY record (hacc R_*.npz; timing records S_*.npz carry
    the dev6 certification arrays in a different layout and are labelled by analyze.timing_labels), every
    distance recomputed with an independent NumPy decode (nphead for heads: h~(z) by hand; npcheck.decode257),
    every rho recomputed with independent NumPy rule tables (npcheck.tables, numpy Gauss-Legendre);
  * anchor verification and the head initial-fit veto (A1.3) recomputed from the saved per-step certificates, budget
    exits and initial-fit exit reasons (the certificates themselves are the in-job autodiff values; the saved internal
    trajectories allow their recomputation, which analyze.py spot-checks with nphead.gammas);
  * Q3 flags (rank-deficient / degenerate step Jacobian, A1.3, A2.6).
"""
from __future__ import annotations

import numpy as np

import npcheck as NC
import nphead as NH
import qscore as QS_
from p1score import agree, dmax, finite

BAR_Q1, BAR_TT, BAR_Q2, BAR_RHO, GAMMA = 2.5e-4, 2.5e-5, 2.5e-4, .116, 1e-3
ANCHOR_T, ANCHOR_TT = 1e-6, 1e-7
TABLES = {}


def modes(M):
    import p1common as P            # frozen 1024^2 list (the same builder the jobs use)
    kx, ky = P.mode_list('frozen', 1024)
    return kx[:M], ky[:M]


def tables(rule, dim, M, L):
    key = (rule, dim, M, L)
    if key not in TABLES:
        X, w = NC.gauss_rule(int(rule[5:]))
        kx, ky = modes(M)
        TABLES[key] = NC.tables(X, w, dim, kx, ky, L)
    return TABLES[key]


def q3_flag(s):
    """A1.3 / A2.6 classifier of one step Jacobian's singular values: 'degenerate' (all-zero or non-finite),
    'rank-deficient step Jacobian' (sigma_min / sigma_max < 1e-10), else None."""
    s = np.asarray(s, float)
    if not np.isfinite(s).all() or s.max() <= 0:
        return 'degenerate'
    if s.min() / s.max() < 1e-10:
        return 'rank-deficient step Jacobian'
    return None


def coef_fn(R, head=None):
    if str(R['kind']) == 'head':
        assert head is not None, 'a head record needs its head (nphead.Head) for the independent decode'
        return lambda W: np.array([head.c(w) for w in np.asarray(W)]), 512
    return lambda W: np.asarray(W), int(R['dim'])


def qualify(R, rows, head=None, qp_rule=None, qpp=True):
    """R: an accuracy record (hacc R_*.npz); rows: its in-job rows. Returns per-case values, Q1, Q2, Q3 and labels."""
    coef, dim = coef_fn(R, head)
    dec = lambda W: NC.decode257(coef(W), dim)
    ids = [str(x) for x in R['case_ids']]
    ix = {c: j for j, c in enumerate(ids)}
    rowby = {r_['case']: r_ for r_ in rows}
    out = dict(per_case={}, disagreements=[])
    is_head = str(R['kind']) == 'head'

    def cmp(cid, key, off):
        r_ = rowby.get(cid)
        if r_ is None or r_.get(key) is None:
            out['disagreements'].append((cid, key, 'missing in-job value'))
        elif not agree(off, r_[key]):
            out['disagreements'].append((cid, key, off, r_[key]))

    cert = np.asarray(R['cert'], float)
    rec = np.asarray(R['rec'], float)
    gcert = bool(cert.size and np.all(np.isfinite(cert) & np.isfinite(rec) & (np.abs(cert - rec) <= 1e-7 * np.abs(rec) + 1e-12)))
    nonc = int(np.sum(~(cert <= GAMMA)))
    out.update(pooled_noncertified=nonc, pooled_steps=int(cert.size), gcert_all=gcert,
               budget_exit_steps=int(np.sum(np.asarray(R['reason']) == 0)))
    for j, cid in enumerate(ids):
        n0 = float(R['n0'][j])
        assert np.isfinite(n0) and n0 > 0
        C = coef(R['W'][j])
        fp = NC.decode257(C, dim)
        dec_err = float(np.max(np.abs(fp - R['f257'][j])) / np.max(np.abs(R['f257'][j])))
        coef_err = float(np.max(np.abs(C - R['C'][j])) / np.max(np.abs(C)))
        d_t = NC.fdist(fp, dec(R['W_tight'][j]), n0)
        ic_ok = (not is_head) or int(R['ic'][j][0]) in (1, 4)
        valid = bool(finite(R['W'][j], R['w0'][j], R['states'][j], R['f257'][j]) and not np.any(np.asarray(R['reason'][j]) == 3)
                     and bool(R['valid'][j]) and ic_ok)
        ct = np.asarray(R['cert_tight'][j], float)
        ver_t = bool(R['valid_tight'][j] and finite(R['W_tight'][j]) and int(R['budget_exits_tight'][j]) == 0
                     and np.all(np.isfinite(ct) & (ct <= ANCHOR_T)) and ((not is_head) or int(R['ic_tight'][j][0]) == 4))
        if ver_t != bool(R['verified_tight'][j]):
            out['disagreements'].append((cid, 'verified_tight', ver_t, bool(R['verified_tight'][j])))
        out['per_case'][cid] = dict(valid=valid, ic_ok=ic_ok, tight_verified=ver_t, dist_tight=d_t, decode_rel=dec_err,
                                    coef_rel=coef_err)
        if not (dec_err <= 1e-9 and coef_err <= 1e-9):
            out['disagreements'].append((cid, 'decode', dec_err, coef_err))
        cmp(cid, 'dist_tight', d_t)
    devc = [str(x) for x in R['dev_case_ids']]
    tt = []
    for k, cid in enumerate(devc):
        j = ix[cid]
        d_tt = NC.fdist(dec(R['W_tight'][j]), dec(R['W_tighter'][k]), float(R['n0'][j]))
        ctt = np.asarray(R['cert_tighter'][k], float)
        be = int(R['budget_exits_tighter'][k]) if 'budget_exits_tighter' in R else 0
        ver_tt = bool(R['valid_tighter'][k] and finite(R['W_tighter'][k]) and be == 0 and np.all(np.isfinite(ctt) & (ctt <= ANCHOR_TT))
                      and ((not is_head) or int(R['ic_tighter'][k][0]) == 4))
        out['per_case'][cid].update(dist_tight_tighter=d_tt, tighter_verified=ver_tt)
        cmp(cid, 'dist_tight_tighter', d_tt)
        tt.append(d_tt)
    q2case = [str(x) for x in R['q2_case_ids']]
    q2 = None
    if q2case:
        rule = str(R['rule'])
        M = int(R['M'])
        L = int(R['mesh'])
        t_q, t_qp = tables(rule, dim, M, L), tables(qp_rule, dim, M, L)
        dq, vq, conv, vqq, rho_off = [], [], [], [], []
        rho_job = np.asarray(R['rho'], float)
        for k, cid in enumerate(q2case):
            j = ix[cid]
            d = NC.fdist(dec(R['W'][j]), dec(R['W_qp'][k]), float(R['n0'][j]))
            dq.append(d)
            out['per_case'][cid]['dist_qp'] = d
            cmp(cid, 'dist_qp', d)
            vq.append(bool(R['valid_qp'][k] and finite(R['W_qp'][k])))
            C = coef(R['W'][j])
            for kk in range(6):
                off = NC.rho(C[kk], t_q, t_qp)
                rho_off.append(off)
                if not (np.isnan(off) and np.isnan(rho_job[k, kk])) and not agree(off, rho_job[k, kk], rel=1e-6, ab=1e-12):
                    out['disagreements'].append((cid, f'rho[{kk}]', off, float(rho_job[k, kk])))
        if qpp:
            for k, cid in enumerate(devc):
                if cid not in q2case:
                    continue
                kq, j = q2case.index(cid), ix[cid]
                c_ = NC.fdist(dec(R['W_qp'][kq]), dec(R['W_qpp'][k]), float(R['n0'][j]))
                conv.append(c_)
                out['per_case'][cid]['dist_qp_qpp'] = c_
                cmp(cid, 'dist_qp_qpp', c_)
                vqq.append(bool(R['valid_qpp'][k] and finite(R['W_qpp'][k])))
        rho = np.asarray(rho_off, float)
        rok = rho[~np.isnan(rho)]
        q2 = dict(population=len(q2case), rollout=bool(dq) and dmax(dq) <= BAR_Q2, worst_dist_qp=dmax(dq),
                  rho_max=dmax(rok) if rok.size else None, rho=bool(rok.size) and dmax(rok) <= BAR_RHO,
                  successor_valid=bool(vq) and all(vq), check_valid=(not qpp) or (bool(vqq) and all(vqq)),
                  convergence=(not qpp) or (bool(conv) and dmax(conv) <= BAR_Q2), worst_dist_qp_qpp=dmax(conv) if conv else None)
    pc = out['per_case'].values()
    q1 = dict(all_valid=bool(pc) and all(v['valid'] for v in pc),
              anchors_verified=bool(pc) and all(v['tight_verified'] for v in pc),
              certificate=bool(cert.size) and nonc <= .01 * cert.size, gcert=gcert,
              distance=bool(pc) and dmax(v['dist_tight'] for v in pc) <= BAR_Q1,
              tight_tighter=bool(tt) and dmax(tt) <= BAR_TT and all(out['per_case'][c]['tighter_verified'] for c in devc))
    q1['offline_agrees'] = not out['disagreements']
    q1['pass'] = all(q1.values())
    out['Q1'] = q1
    if q2 is None:
        out['Q2'] = dict(label='not run here')
    else:
        q2['label'] = QS_.q2_label(q2, bool(out['disagreements']))
        out['Q2'] = q2
    out['numerically_qualified'] = bool(q1['pass'] and out['Q2'].get('label') == 'pass')
    # Q3 flags
    sv = np.asarray(R['q3_svals'], float)
    flags = [q3_flag(sv[k, t_]) for k in range(sv.shape[0]) for t_ in range(sv.shape[1])]
    flags = [f_ for f_ in flags if f_]
    finite_sv = sv[np.isfinite(sv).all(axis=2) & (sv.max(axis=2) > 0)]
    out['Q3'] = dict(cond_median=float(np.median(finite_sv[:, 0] / finite_sv[:, -1])) if finite_sv.size else None,
                     cond_max=float(np.max(finite_sv[:, 0] / finite_sv[:, -1])) if finite_sv.size else None,
                     flags=sorted(set(flags)))
    return out


def score(R, kinds=('ST', 'S')):
    """E0 per case (qualify's qscore.score_config: identity assertions; S only for LSPG-BE at dt0)."""
    return QS_.score_config(R, float(R['dt']), kinds)
