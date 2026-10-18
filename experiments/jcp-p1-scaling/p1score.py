"""jcp-p1-scaling offline scoring and qualification (DESIGN 3, 5, A1.3, A1.4, A2.4; CPU only, no JAX).

  * E0 scoring through refstore with per-record identity assertions (cohort, case, time tag, reference kind, store
    version, metric id); S only for LSPG-BE at dt0 (labelled); first-order refdv ST for P1g comparisons.
  * Offline recomputation of every Q1 / Q2 distance from the saved coefficients with the independent NumPy decode
    (npcheck.decode257), compared with the in-job values (A1.3 tolerance).
  * Configuration-level Q1 / Q2 labels (DESIGN 5.1, 5.2 as amended).
  * Reproduction gate against the combined archive (AM-1).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
E0 = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references')
sys.path.insert(0, str(E0))
import refstore as RS  # noqa: E402
import npcheck as NC  # noqa: E402

REFDV = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-01-quadrature-study/experiments/quadrature-study/runs/refdv/archive/output')
COMB = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/runs/c2grid/archive/output')
TIME_TAG = 't0,t1,t2,t3,t4,t5'
BAR_Q1, BAR_TT, BAR_Q2, BAR_RHO, GAMMA = 2.5e-4, 2.5e-5, 2.5e-4, .116, 1e-3


class IdentityError(ValueError):
    pass


def agree(off, job, rel=1e-9, ab=1e-13):
    return bool(abs(off - job) <= rel * abs(job) + ab)


def score(field, rec, kind='ST'):
    """rec: dict(case_id='dev6|3', time_tag=TIME_TAG, scheme='BE'|'CNR', dt=0.005, requested=(cohort, case))."""
    coh, c = rec['requested']
    if rec['case_id'] != f'{coh}|{c}':
        raise IdentityError(f"case swap: field {rec['case_id']} scored as {coh}|{c}")
    if rec['time_tag'] != TIME_TAG:
        raise IdentityError(f"time swap: field time tag {rec['time_tag']}")
    if kind == 'S' and not (rec['scheme'] == 'BE' and abs(rec['dt'] - .005) < 1e-15):
        raise IdentityError('reference-kind swap: S is matched only to LSPG-BE at dt0 = 0.005')
    s = RS.score(field, 2, coh, int(c), kind)
    if (s['cohort'], s['case'], s['kind']) != (coh, int(c), kind) or s['store_version'] != RS.store_version():
        raise IdentityError('score record identity differs from the request')
    return s


def first_order_st(field, coh, c):
    r = np.load(REFDV / f'ref_ST_{coh}_{int(c):03d}.npz')['f257']
    n0 = float(np.linalg.norm(r[0]))
    return max(float(np.linalg.norm(field[k] - r[k])) / n0 for k in range(1, 6))


def n0_of(R, j):
    return float(R['n0'][j])


def qualify(R, rows, Rp):
    """Configuration-level Q1 / Q2 of one (config, scheme) from its saved arrays (R = dict of the R_*.npz) and the
    in-job rows. Recomputes every distance with the NumPy decode and checks agreement with the job (A1.3)."""
    ids = [str(x) for x in R['case_ids']]
    n = len(ids)
    dec = lambda W: NC.decode257(W, Rp)
    rowby = {f"{r_['cohort']}|{r_['case']}": r_ for r_ in rows}
    out = dict(cases=n, per_case={}, disagreements=[])
    gam = np.asarray(R['gamma'])
    nonst = int(np.sum(~(gam <= GAMMA)))
    out['pooled_nonstationary'] = nonst
    out['pooled_steps'] = int(gam.size)
    gc = np.all(np.abs(gam - R['gn_rec']) <= 1e-7 * np.abs(R['gn_rec']) + 1e-12)
    out['gcert_all'] = bool(gc)
    reasons = np.asarray(R['reason'])
    out['budget_exit_steps'] = int(np.sum(reasons == 0))
    for j, cid in enumerate(ids):
        n0 = n0_of(R, j)
        fp = dec(R['W'][j])
        rec_err = float(np.max(np.abs(fp - R['f257'][j])) / np.max(np.abs(R['f257'][j])))
        ft = dec(R['W_tight'][j])
        d_t = NC.fdist(fp, ft, n0)
        valid = bool(np.isfinite(R['W'][j]).all() and not np.any(reasons[j] == 3))
        pc = dict(valid=valid and bool(R['valid'][j]), tight_valid=bool(R['valid_tight'][j]), dist_tight=d_t,
                  decode_rel=rec_err)
        r_ = rowby.get(cid, {})
        if r_ and not agree(d_t, r_['dist_tight']):
            out['disagreements'].append((cid, 'dist_tight', d_t, r_['dist_tight']))
        out['per_case'][cid] = pc
    ids_ix = {cid: j for j, cid in enumerate(ids)}
    devc = [str(x) for x in R['dev_case_ids']]
    tt = []
    for k, cid in enumerate(devc):
        j = ids_ix[cid]
        d_tt = NC.fdist(dec(R['W_tight'][j]), dec(R['W_tighter'][k]), n0_of(R, j))
        out['per_case'][cid]['dist_tight_tighter'] = d_tt
        out['per_case'][cid]['tighter_valid'] = bool(R['valid_tighter'][k])
        r_ = rowby.get(cid, {})
        if r_.get('dist_tight_tighter') is not None and not agree(d_tt, r_['dist_tight_tighter']):
            out['disagreements'].append((cid, 'dist_tight_tighter', d_tt, r_['dist_tight_tighter']))
        tt.append(d_tt)
    q1 = dict(all_valid=all(v['valid'] and v['tight_valid'] for v in out['per_case'].values()),
              stationarity=bool(nonst <= .01 * gam.size), gcert=out['gcert_all'],
              distance=all(v['dist_tight'] <= BAR_Q1 for v in out['per_case'].values()),
              tight_tighter=bool(tt) and all(x <= BAR_TT for x in tt) and all(
                  v.get('tighter_valid', True) for v in out['per_case'].values()),
              offline_agrees=not out['disagreements'])
    q1['pass'] = all(q1.values())
    out['Q1'] = q1
    if 'W_qp' in R:
        q2case = [str(x) for x in R['q2_case_ids']]
        dq, rho_all, valid_qp, conv = [], [], [], []
        for k, cid in enumerate(q2case):
            j = ids_ix[cid]
            fq = dec(R['W_qp'][k])
            d = NC.fdist(dec(R['W'][j]), fq, n0_of(R, j))
            dq.append(d)
            out['per_case'][cid]['dist_qp'] = d
            r_ = rowby.get(cid, {})
            if r_.get('dist_qp') is not None and not agree(d, r_['dist_qp']):
                out['disagreements'].append((cid, 'dist_qp', d, r_['dist_qp']))
            rho_all += [x for x in np.asarray(R['rho'][k]).tolist()]
            valid_qp.append(bool(R['valid_qp'][k]))
        for k, cid in enumerate(devc if 'W_qpp' in R else []):
            j = ids_ix[cid]
            kq = q2case.index(cid)
            conv.append(NC.fdist(dec(R['W_qp'][kq]), dec(R['W_qpp'][k]), n0_of(R, j)))
            out['per_case'][cid]['dist_qp_qpp'] = conv[-1]
        rho = np.array(rho_all, float)
        trivial = int(np.sum(np.isnan(rho)))
        rho_ok = rho[~np.isnan(rho)]
        q2 = dict(population=len(q2case), rollout=bool(dq) and all(x <= BAR_Q2 for x in dq),
                  rho_max=float(np.max(rho_ok)) if rho_ok.size else None, rho_trivial=trivial,
                  rho=bool(rho_ok.size) and bool(np.max(rho_ok) <= BAR_RHO), successor_valid=all(valid_qp),
                  convergence=bool(conv) and all(x <= BAR_Q2 for x in conv), worst_dist_qp=max(dq) if dq else None)
        if not q2['successor_valid']:
            q2['label'] = 'unverified (successor invalid)'
        elif not rho_ok.size:
            q2['label'] = 'unverified (no rho indicators)'
        elif not (q2['rollout'] and q2['rho']):
            q2['label'] = 'fail'
        elif not q2['convergence']:
            q2['label'] = 'unverified (finer rule not converged)'
        else:
            q2['label'] = 'pass'
        out['Q2'] = q2
    else:
        out['Q2'] = dict(label='not run (descriptive arm)')
    out['numerically_qualified'] = bool(out['Q1']['pass'] and out['Q2'].get('label') == 'pass')
    return out


def score_config(R, scheme, kinds=('ST',)):
    ids = [str(x) for x in R['case_ids']]
    res = {}
    for j, cid in enumerate(ids):
        coh, c = cid.split('|')
        rec = dict(case_id=cid, time_tag=TIME_TAG, scheme=scheme, dt=.005, requested=(coh, int(c)))
        res[cid] = {k: score(R['f257'][j], rec, k) for k in kinds if k == 'ST' or scheme == 'BE'}
    return res


def manifest_equal(job, name):
    """AM-1 / AM-7: numerical-configuration equality of a fresh 1024^2 configuration with the combined archive
    (provenance fields recorded, not required equal)."""
    arch = json.loads((HERE / 'checks' / 'combined-c2grid-manifest.json').read_text())
    a, c = arch['setup'][name], job['configs'][name]
    chk = dict(mesh=job['mesh'] == arch['mesh'], Rp=c['Rp'] == a['Rp'], M=c['M'] == a['M'], rule=c['rules']['main'] == a['rule'],
               m=c['m']['main'] == a['m'], trust=abs(c['trust'] - a['trust']) <= 1e-12 * a['trust'],
               A_cond=abs(c['A_cond'] - a['A_cond']) <= 1e-9 * a['A_cond'], policy=c['policy'] == 'frozen',
               checkpoint=job['checkpoint_sha256'] == arch['checkpoint_sha256'],
               cohorts=all(job['cohorts'][k] == v for k, v in arch['cohorts'].items()),
               edges=job['setup']['bank']['edges'] == arch['bank']['edges'])
    return dict(checks=chk, equal=all(chk.values()), provenance=dict(old=dict(commit=arch['commit'], gpu=arch['gpu'],
                jax=arch['jax_version']), new=dict(commit=job['commit'], gpu=job['gpu'], jax=job['jax_version'])))


def reproduction(R, name, scheme, run_key, job=None):
    z = np.load(COMB / f'W_{name}.npz')
    runs, ac = list(z['runs']), list(z['cases'])
    ri = runs.index(run_key)
    rows = []
    for j, cid in enumerate([str(x) for x in R['case_ids']]):
        Wa = z['W'][ac.index(cid), ri]
        rows.append(dict(case=cid, rel=float(np.max(np.linalg.norm(R['W'][j] - Wa, axis=1)) / np.linalg.norm(Wa[0]))))
    man = manifest_equal(job, name) if job is not None else None
    ok = bool(max(r_['rel'] for r_ in rows) <= 1e-10) and (man is None or man['equal'])
    return dict(rows=rows, worst=max(r_['rel'] for r_ in rows), manifest=man, passed=ok)
