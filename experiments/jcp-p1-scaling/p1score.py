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


STORE_VERSION = RS.store_version()
METRIC_ID = RS.V.METRIC_ID[2]
PINNED = json.loads((HERE / 'runs' / 'MANIFEST-dev.json').read_text()) if (HERE / 'runs' / 'MANIFEST-dev.json').exists() else None
if PINNED is not None:
    assert PINNED['e0_store_version'] == STORE_VERSION, 'E0 store differs from the development manifest'


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
    if (s['cohort'], s['case'], s['kind']) != (coh, int(c), kind) or s['store_version'] != STORE_VERSION or \
            s['metric_id'] != METRIC_ID:
        raise IdentityError('score record identity differs from the request or the pinned store / metric')
    s['reference_sha256'] = RS._manifest()[f'2d/{coh}/{kind}/c{int(c):03d}.npz']
    s['time_tag'] = rec['time_tag']
    return s


def first_order_st(field, coh, c):
    r = np.load(REFDV / f'ref_ST_{coh}_{int(c):03d}.npz')['f257']
    n0 = float(np.linalg.norm(r[0]))
    return max(float(np.linalg.norm(field[k] - r[k])) / n0 for k in range(1, 6))


def n0_of(R, j):
    n0 = float(R['n0'][j])
    if not (np.isfinite(n0) and n0 > 0):
        raise ValueError('non-positive or non-finite normalisation')
    return n0


def case_set(name):
    dev = [f'dev6|{c}' for c in range(6)]
    if name == 'all38':
        return dev + [f'val32|{c}' for c in range(32)]
    if name == 'sub16':
        return dev + [f'val32|{c}' for c in range(10)]
    if name == 'dev6':
        return dev
    if name == 'dev2':
        return dev[:2]
    return [f'{a}|{b}' for a, b in name]


def finite(*xs):
    return all(np.isfinite(np.asarray(x, float)).all() for x in xs)


def dmax(xs):
    """NaN-safe maximum: any non-finite entry -> inf (never silently ignored)."""
    xs = np.asarray(list(xs), float)
    return float('inf') if xs.size == 0 or not np.isfinite(xs).all() else float(xs.max())


def qualify(R, rows, Rp, expect, rho_tables=None, rho_cases=()):
    """Configuration-level Q1 / Q2 (DESIGN 5 as amended) from the saved arrays and the in-job rows.
    expect: dict(cases=[ids], dev=[ids], q2=[ids] or [], qpp=bool). rho_tables: (t_q, t_qp) NumPy tables for the
    independent rho recomputation on rho_cases (all six outputs). Every in-job value has an offline counterpart; a
    missing row, shape or population is a disagreement (fails the label)."""
    ids = [str(x) for x in R['case_ids']]
    out = dict(per_case={}, disagreements=[])
    S = 50
    if ids != list(expect['cases']):
        out['disagreements'].append(('population', ids, list(expect['cases'])))
    n = len(ids)
    for k, shp in (('W', (n, 6, Rp)), ('W_tight', (n, 6, Rp)), ('states', (n, S + 1, Rp)), ('gamma', (n, S)),
                   ('gn_rec', (n, S)), ('reason', (n, S)), ('f257', (n, 6, 257, 257)), ('w0', (n, Rp))):
        if tuple(np.shape(R[k])) != shp:
            out['disagreements'].append(('shape', k, np.shape(R[k]), shp))
    dec = lambda W: NC.decode257(W, Rp)
    rowby = {f"{r_['cohort']}|{r_['case']}": r_ for r_ in rows}
    ids_ix = {cid: j for j, cid in enumerate(ids)}

    def cmp(cid, key, off):
        r_ = rowby.get(cid)
        if r_ is None or r_.get(key) is None:
            out['disagreements'].append((cid, key, 'missing in-job value'))
        elif not agree(off, r_[key]):
            out['disagreements'].append((cid, key, off, r_[key]))

    gam = np.asarray(R['gamma'], float)
    nonst = int(np.sum(~(gam <= GAMMA)))
    out.update(pooled_nonstationary=nonst, pooled_steps=int(gam.size),
               gcert_all=bool(gam.size and np.all(np.abs(gam - R['gn_rec']) <= 1e-7 * np.abs(R['gn_rec']) + 1e-12)),
               budget_exit_steps=int(np.sum(np.asarray(R['reason']) == 0)))
    for j, cid in enumerate(ids):
        n0 = n0_of(R, j)
        fp = dec(R['W'][j])
        rec_err = float(np.max(np.abs(fp - R['f257'][j])) / np.max(np.abs(R['f257'][j])))
        ft = dec(R['W_tight'][j])
        d_t = NC.fdist(fp, ft, n0)
        valid = bool(finite(R['W'][j], R['w0'][j], R['states'][j], R['f257'][j], R['gn_rec'][j])
                     and not np.any(np.asarray(R['reason'][j]) == 3) and bool(R['valid'][j]))
        out['per_case'][cid] = dict(valid=valid, tight_valid=bool(R['valid_tight'][j] and finite(R['W_tight'][j])),
                                    dist_tight=d_t, decode_rel=rec_err)
        if not rec_err <= 1e-12:                    # the scored fields must be the decode of the saved coefficients
            out['disagreements'].append((cid, 'decode', rec_err))
        cmp(cid, 'dist_tight', d_t)
    devc = [str(x) for x in R['dev_case_ids']]
    if devc != list(expect['dev']):
        out['disagreements'].append(('dev population', devc, list(expect['dev'])))
    tt = []
    for k, cid in enumerate(devc):
        j = ids_ix[cid]
        d_tt = NC.fdist(dec(R['W_tight'][j]), dec(R['W_tighter'][k]), n0_of(R, j))
        out['per_case'][cid].update(dist_tight_tighter=d_tt, tighter_valid=bool(R['valid_tighter'][k] and finite(R['W_tighter'][k])))
        cmp(cid, 'dist_tight_tighter', d_tt)
        tt.append(d_tt)
    q2case = [str(x) for x in R['q2_case_ids']] if 'q2_case_ids' in R else []
    if 'W_qp' in R and q2case != list(expect['q2']):
        out['disagreements'].append(('q2 population', q2case, list(expect['q2'])))
    q2 = None
    if expect['q2']:
        if rho_tables is None or set(rho_cases) != set(expect['dev']):
            raise ValueError('Q2 needs the independent rho recomputation on the dev cases (code audit 3)')
        nq = len(expect['q2'])
        for k, shp in (('W_qp', (nq, 6, Rp)), ('rho', (nq, 6)), ('valid_qp', (nq,))) + \
                ((('W_qpp', (len(expect['dev']), 6, Rp)), ('valid_qpp', (len(expect['dev']),))) if expect.get('qpp') else ()):
            if tuple(np.shape(R[k])) != shp:
                out['disagreements'].append(('shape', k, np.shape(R[k]), shp))
        dq, valid_qp, conv, valid_qpp = [], [], [], []
        rho_job = np.asarray(R['rho'], float)
        for k, cid in enumerate(q2case):
            j = ids_ix[cid]
            fq = dec(R['W_qp'][k])
            d = NC.fdist(dec(R['W'][j]), fq, n0_of(R, j))
            dq.append(d)
            out['per_case'][cid]['dist_qp'] = d
            cmp(cid, 'dist_qp', d)
            valid_qp.append(bool(R['valid_qp'][k] and finite(R['W_qp'][k])))
            if cid in rho_cases and rho_tables is not None:      # independent rho (NumPy tables, A2.4 semantics)
                for kk in range(6):
                    off = NC.rho(np.asarray(R['W'][j, kk]), *rho_tables)
                    if not (np.isnan(off) and np.isnan(rho_job[k, kk])) and not agree(off, rho_job[k, kk]):
                        out['disagreements'].append((cid, f'rho[{kk}]', off, float(rho_job[k, kk])))
        if expect.get('qpp'):
            for k, cid in enumerate(devc):
                kq, j = q2case.index(cid), ids_ix[cid]
                c_ = NC.fdist(dec(R['W_qp'][kq]), dec(R['W_qpp'][k]), n0_of(R, j))
                conv.append(c_)
                out['per_case'][cid]['dist_qp_qpp'] = c_
                cmp(cid, 'dist_qp_qpp', c_)
                valid_qpp.append(bool(R['valid_qpp'][k] and finite(R['W_qpp'][k])))
        rho = rho_job[:len(q2case)].ravel()
        nonfin = int(np.sum(np.isinf(rho)))
        trivial = int(np.sum(np.isnan(rho)))
        rho_ok = rho[~np.isnan(rho)]
        q2 = dict(population=len(q2case), rollout=bool(dq) and dmax(dq) <= BAR_Q2, worst_dist_qp=dmax(dq),
                  rho_max=dmax(rho_ok) if rho_ok.size else None, rho_trivial=trivial, rho_infinite=nonfin,
                  rho=bool(rho_ok.size) and dmax(rho_ok) <= BAR_RHO, successor_valid=bool(valid_qp) and all(valid_qp),
                  check_valid=(not expect.get('qpp')) or (bool(valid_qpp) and all(valid_qpp)),
                  convergence=(not expect.get('qpp')) or (bool(conv) and dmax(conv) <= BAR_Q2))
    q1 = dict(all_valid=bool(out['per_case']) and all(v['valid'] and v['tight_valid'] for v in out['per_case'].values()),
              stationarity=bool(gam.size == n * S) and nonst <= .01 * gam.size, gcert=out['gcert_all'],
              distance=bool(out['per_case']) and dmax(v['dist_tight'] for v in out['per_case'].values()) <= BAR_Q1,
              tight_tighter=bool(tt) and len(tt) == len(expect['dev']) and dmax(tt) <= BAR_TT and
              all(out['per_case'][c]['tighter_valid'] for c in devc),
              offline_agrees=not out['disagreements'])
    q1['pass'] = all(q1.values())
    out['Q1'] = q1
    if q2 is None:
        out['Q2'] = dict(label='not run (descriptive arm)')
    else:
        if out['disagreements']:
            q2['label'] = 'fail (offline disagreement)'
        elif not (q2['successor_valid'] and q2['check_valid']):
            q2['label'] = 'unverified (successor invalid)'
        elif q2['rho_max'] is None:
            q2['label'] = 'unverified (no rho indicators)'
        elif not (q2['rollout'] and q2['rho']):
            q2['label'] = 'fail'
        elif not q2['convergence']:
            q2['label'] = 'unverified (finer rule not converged)'
        else:
            q2['label'] = 'pass'
        out['Q2'] = q2
    out['Q1']['offline_agrees'] = not out['disagreements']        # final, after every comparison
    out['Q1']['pass'] = all(v for k, v in out['Q1'].items() if k != 'pass')
    out['numerically_qualified'] = bool(out['Q1']['pass'] and out['Q2'].get('label') == 'pass')
    return out


def rule_tables(rule, Rp, kx, ky, L):
    """NumPy tables of a named rule for the independent rho (points from the vendored rule definitions)."""
    if rule.startswith('gauss'):
        X, w = NC.gauss_rule(int(rule[5:]))
    else:
        sys.path.insert(0, str(HERE / 'vendor' / 'quad2d' / 'vendor'))
        import hari_quadrature as HQ
        a, b, k = 1, 1, 2
        n = int(rule[3:])
        while b < n:
            a, b, k = b, a + b, k + 1
        X, w = HQ.fibonacci_lattice(k, seed=0)
        X, w = np.asarray(X, float), np.asarray(w, float)
    return NC.tables(X, w, Rp, kx, ky, L)


def score_config(R, kinds=('ST',)):
    ids = [str(x) for x in R['case_ids']]
    scheme = str(R['scheme'])
    res = {}
    for j, cid in enumerate(ids):
        coh, c = cid.split('|')
        rec = dict(case_id=cid, time_tag=str(R['time_tag']), scheme=scheme, dt=float(R['dt']), requested=(coh, int(c)))
        res[cid] = {k: score(R['f257'][j], rec, k) for k in kinds if k == 'ST' or scheme == 'BE'}
    return res


def manifest_equal(job, name, R):
    """AM-1 / AM-7: numerical-configuration equality of a fresh 1024^2 configuration with the combined archive.
    Compared: mesh, r, M, ordered mode identities (hash of the 1024^2 builder list = the archive's construction), rule,
    m and the rule-point hash, trust, A condition number, policy, bank edges, checkpoint and rotation sha256, cohorts,
    scheme / dt / form / tolerance level / budget / initialisation / weighting / output times (the archive config and
    the shared code text). Provenance (commit, GPU, jax) recorded, not compared."""
    arch = json.loads((HERE / 'checks' / 'combined-c2grid-manifest.json').read_text())
    dev = json.loads((HERE / 'runs' / 'MANIFEST-dev.json').read_text())
    a, c = arch['setup'][name], job['configs'][name]
    M = c['M']
    chk = dict(mesh=job['mesh'] == arch['mesh'] == 1024, Rp=c['Rp'] == a['Rp'], M=M == a['M'],
               modes=str(R['mode_hash']) == dev['mode_lists'][f'1024|frozen|{M}']['hash'] ==
               dev['mode_lists'][f'1024|nominal|{M}']['hash'],
               rule=c['rules']['main'] == a['rule'], m=c['m']['main'] == a['m'],
               rule_points=c['meta']['main']['rule_sha256'] == dev['rules'][a['rule']]['sha256'],
               trust=abs(c['trust'] - a['trust']) <= 1e-12 * a['trust'],
               A_cond=abs(c['A_cond'] - a['A_cond']) <= 1e-9 * a['A_cond'], policy=c['policy'] == 'frozen',
               checkpoint=job['checkpoint_sha256'] == arch['checkpoint_sha256'] == dev['checkpoint_sha256'],
               rotation=job['rotation_sha256'] == dev['rotation_sha256'],
               cohorts=all(job['cohorts'][k] == v for k, v in arch['cohorts'].items()),
               edges=job['setup']['bank']['edges'] == arch['bank']['edges'],
               scheme_dt=float(R['dt']) == .005 and str(R['scheme']) in ('BE', 'CNR'),
               archive_grid=['LSPG', str(R['scheme'])] in arch['config']['grid']['forms_schemes'] and 1 in arch['config']['grid']['dt_factors'],
               solver_text=dev['code']['tol_of_prod_LSPG'] == [1e-3, 1e-9] and dev['code']['budget'] == 600,
               run_form=str(R['form']) == 'LSPG', run_level=str(R['level']) == 'prod',
               run_tol=[float(x) for x in R['tol']] == [1e-3, 1e-9], run_budget=int(R['budget']) == 600,
               run_weighting=str(R['weighting']) == 'LSPG D = 1/(a0 + dt b0 nu lam)',
               run_time_tag=str(R['time_tag']) == TIME_TAG, run_metric=str(R['metric_id_target']) == METRIC_ID)
    return dict(checks=chk, equal=all(chk.values()), provenance=dict(old=dict(commit=arch['commit'], gpu=arch['gpu'],
                jax=arch['jax_version']), new=dict(commit=job['commit'], gpu=job['gpu'], jax=job['jax_version'])))


def reproduction(R, name, scheme, run_key, job):
    z = np.load(COMB / f'W_{name}.npz')
    runs, ac = list(z['runs']), list(z['cases'])
    ri = runs.index(run_key)
    rows = []
    for j, cid in enumerate([str(x) for x in R['case_ids']]):
        Wa = z['W'][ac.index(cid), ri]
        rows.append(dict(case=cid, rel=float(np.max(np.linalg.norm(R['W'][j] - Wa, axis=1)) / np.linalg.norm(Wa[0]))))
    man = manifest_equal(job, name, R)
    worst = dmax(r_['rel'] for r_ in rows)
    return dict(rows=rows, worst=worst, manifest=man, passed=bool(worst <= 1e-10 and man['equal']))
