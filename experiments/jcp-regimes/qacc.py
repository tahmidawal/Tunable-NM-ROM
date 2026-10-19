"""jcp-qualify accuracy / qualification job (DESIGN 2, 4, 5, 7): one mesh per job, one GPU. Generalises P1's p1acc.py to
any step size dt = factor * dt0 and to both forms (LSPG, GAL).

Config: {"mesh", "edges", "settings": [{"name", "Rp", "rule", "qp", "qpp", "arms": [{"form", "scheme", "factor",
"cases", "q2_cases"?, "tighter"?(default true), "q3"?(default true)}]}], "blocks": [...]}.
Per setting (Rp, rule) and arm (form, scheme, factor), per case of the arm's case set:
  prod      fused query, internal trajectory recorded; certified in-job (LSPG: p1core.make_certifier, autodiff gamma;
            GAL: qcert.make_gal_certifier, autodiff eta / gamma / cond) and compared with the solver's own record (G-cert)
  tight     same configuration at the tight tolerance (Q1 distance), every case
  tighter   dev6 cases (Q1 tight/tighter agreement)
  qp        successor-rule replay on the Q2 population (sub16 by default); rho of the arm's own six output states
  qpp       check-rule replay on dev6 (finer-rule convergence)
  Q3        singular values of the scheme's step Jacobian at output indices 1, 3, 5 of each dev6 trajectory
Blocks: 'controls' (smoke numerical controls), 'order' (B4), each on its own Setting.

    python qacc.py --config configs/<attempt>.json --out <dir> [--dry-run]
Outputs: result.json, R_<setting>_<form>_<scheme>_<factor>.npz per arm, extra npz per block, COMPLETE.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import p1common as P
from p1common import Q, QS, T2, DT0, sha
import qcert as QC
import jax
import jax.numpy as jnp

GAMMA = 1e-3
import guard as G                                        # jcp-regimes edit R4 (code audit 2): write boundary


def np_save(path, **arrays):
    G.check_write(path)
    np.savez_compressed(path, **arrays)

PROBE = np.sort(np.random.default_rng(20261019).choice(257 * 257, 4096, replace=False))   # jcp-regimes edit R2


def steps_of(dt):
    s, k = int(round(.25 / dt)), int(round(.05 / dt))
    assert abs(s * dt - .25) < 1e-12 and abs(k * dt - .05) < 1e-12, dt
    return s


def arm_key(form, scheme, factor):
    return f'{form}_{scheme}_{factor:g}'


def co_of(scheme):
    return np.array(T2.SCHEMES[scheme][:5], float)


class Queries:
    """Compiled split queries per (setting, form, nrec, budget, mutation), built lazily (p1core.make_split_query)."""

    def __init__(self):
        self.c = {}

    def get(self, st, form, nrec, budget=600, mutation=None, nstep=T2.NSTEP):
        k = (id(st), form, nrec, budget, mutation, nstep)
        if k not in self.c:
            self.c[k] = T2.make_split_query(Q, 'point', form, st.Rp, st.L, st.trust, nrec=nrec, budget=budget,
                                            mutation=mutation, nstep=nstep)
        return self.c[k]


QQ = Queries()


def run(bank, st, role, form, scheme, dt, level, u0, nu, record=False, budget=600, mutation=None):
    """End-to-end fused query, guarded by the table metadata check (wrong-mesh control)."""
    st.check(role, bank)
    S = steps_of(dt)
    nstep = max(T2.NSTEP, S + 1)
    q = QQ.get(st, form, S + 1 if record else 0, budget, mutation, nstep)
    g, t_ = P.tol_of(form, level)
    return q['fused'](u0, nu, st.data[(role, form)], st.cold, T2.sched(scheme, dt, g, t_))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    cases = P.case_list(cfg)
    if a.dry_run:
        inv = []
        for s in cfg['settings']:
            for arm in s['arms']:
                n = len(P.select(cases, arm['cases']))
                inv.append((s['name'], arm_key(arm['form'], arm['scheme'], arm['factor']), n, steps_of(DT0 * arm['factor'])))
        print(json.dumps(dict(arms=inv, rollouts_prod=sum(x[2] for x in inv)), indent=0))
        return
    out = Path(a.out)
    G.check_write(out / 'result.json')
    out.mkdir(parents=True, exist_ok=True)
    G.check_write(out / 'result.json')
    P.check_env(cfg)
    begin = time.perf_counter()
    el = lambda: round(time.perf_counter() - begin, 1)
    L = int(cfg['mesh'])
    rep = dict(config=cfg, commit=P.os.environ.get('SOURCE_COMMIT'), job_id=P.os.environ.get('SLURM_JOB_ID'),
               backend=jax.default_backend(), gpu=jax.devices()[0].device_kind, nvidia_smi=P.gpu_info(),
               jax_version=jax.__version__, mesh=L, checkpoint_sha256=QS.sha_file(T2.CKPT),
               rotation_sha256=QS.sha_file(Q.ROTATION), cohorts=P.cohort_hashes(cfg), setup={}, rows=[], settings={},
               memory={}, phases={}, elapsed={}, complete=False)
    save = lambda: json.dump(QS.clean(rep), open(G.check_write(out / 'result.json'), 'w'), indent=1)
    fz = cfg['frozen']                                   # jcp-regimes edit R3 (code audit 1): frozen identities enforced
    assert rep['checkpoint_sha256'] == fz['checkpoint_sha256'], rep['checkpoint_sha256']
    assert rep['rotation_sha256'] == fz['rotation_sha256'], rep['rotation_sha256']
    assert set(fz['settings']) == {s_['name'] for s_ in cfg['settings']}
    QS.burn(.05)
    rep['memory']['start'] = P.mem()
    bank = P.Bank(L, cfg['edges'], nrb=cfg.get('bank_blocks'))
    rep['setup']['bank'] = dict(edges=bank.edges, seconds=bank.build_seconds, bytes=bank.bytes, mem=P.mem())
    ops = bank.ops('frozen')
    rep['setup']['ops_frozen'] = dict(seconds=ops['seconds'], list_hash_2048=P.list_hash(ops['kx'], ops['ky']), mem=P.mem())
    s256 = L // 256
    U0 = {}
    for i, (coh, c, ph) in enumerate(cases):
        u0n, nu_ = P.ic_full(L, coh, ph)                  # jcp-regimes edit R1 (A1.6 iv)
        U0[i] = (jnp.asarray(u0n), nu_, float(np.linalg.norm(u0n[::s256, ::s256])))
    rep['ic_check'] = {}
    for coh in dict.fromkeys(c_[0] for c_ in cases):     # n0 equals the lattice-IC norm (A1.6 iv), one case per cohort
        i = next(k for k, c_ in enumerate(cases) if c_[0] == coh)
        lat = P.ic_full(256, coh, cases[i][2])[0]
        rep['ic_check'][coh] = dict(case=cases[i][1], n0=U0[i][2], n0_lattice=float(np.linalg.norm(lat)),
                                    equal=bool(np.array_equal(np.asarray(U0[i][0])[::s256, ::s256], lat)))
    rep['phases']['setup'] = el()
    print('SETUP', el(), P.mem(), flush=True)
    save()

    def go(st, role, form, scheme, dt, level, ci, record=False, budget=600, mutation=None):
        u0, nu, n0 = U0[ci]
        t1 = time.perf_counter()
        v = run(bank, st, role, form, scheme, dt, level, u0, nu, record=record, budget=budget, mutation=mutation)
        jax.block_until_ready(v)
        return v, P.restrict(v['fields'], L), time.perf_counter() - t1

    for sc_ in cfg['settings']:
        t_set = time.perf_counter()
        roles = {k: sc_[k] for k in ('qp', 'qpp') if sc_.get(k)}
        b0 = P.mem()['in_use']
        st = P.Setting(bank, sc_['name'], sc_['Rp'], sc_['rule'], 'frozen', roles=roles, forms=('LSPG', 'GAL'), nrec=0)
        want = fz['settings'][sc_['name']]               # R3: rule / mesh / mode-list identity equals FRONT-2D's manifest
        got = {k: st.meta['main'][k] for k in want}
        assert got == want, (sc_['name'], got, want)
        A = np.asarray(st.base['A'])
        sA = np.linalg.svd(A, compute_uv=False)
        srep = rep['settings'][sc_['name']] = dict(Rp=st.Rp, M=st.M, policy=st.policy, rules=st.rules, m=st.m, meta=st.meta,
                                                   trust=st.trust, table_bytes=st.table_bytes,
                                                   setting_bytes=P.mem()['in_use'] - b0, A_svals=sA.tolist(),
                                                   A_cond=float(sA[0] / sA[-1]), A_rank=int(np.sum(sA > sA[0] * 1e-13)),
                                                   arms={})
        nl = st.q['LSPG']['nl']
        certs = dict(LSPG=T2.make_certifier(nl, st.Rp), GAL=QC.make_gal_certifier(nl, st.Rp))
        jacs = dict(LSPG=T2.make_jac(nl), GAL=QC.make_gal_jac(nl))
        rep['elapsed'][f"{sc_['name']}|setup"] = time.perf_counter() - t_set
        for arm in sc_['arms']:
            t_arm = time.perf_counter()
            form, scheme, fac = arm['form'], arm['scheme'], float(arm['factor'])
            dt = DT0 * fac
            S = steps_of(dt)
            ak = arm_key(form, scheme, fac)
            ids = P.select(cases, arm['cases'])
            q2 = [i for i in P.select(cases, arm.get('q2_cases', 'all')) if i in ids] if st.rules.get('qp') else []
            dev = [i for i in P.select(cases, arm.get('tighter_cases', 'sub')) if i in ids] if arm.get('tighter', True) else []
            n = len(ids)
            g_prod, t_prod = P.tol_of(form, 'prod')
            sch = T2.sched(scheme, dt, g_prod, t_prod)
            R = dict(case_ids=np.array([f'{cases[i][0]}|{cases[i][1]}' for i in ids]),
                     W=np.full((n, 6, st.Rp), np.nan), w0=np.full((n, st.Rp), np.nan),
                     states=np.full((n, S + 1, st.Rp), np.nan), cert=np.full((n, S), np.nan), gamma_lit=np.full((n, S), np.nan),
                     jcond=np.full((n, S), np.nan), rec=np.full((n, S), np.nan), raw=np.full((n, S), np.nan),
                     scale=np.full((n, S), np.nan), tolratio=np.full((n, S), np.nan), it=np.full((n, S), -1),
                     reason=np.full((n, S), -1), f257=np.full((n, 6, 257, 257), np.nan), n0=np.full(n, np.nan),
                     W_tight=np.full((n, 6, st.Rp), np.nan), valid=np.zeros(n, bool), valid_tight=np.zeros(n, bool),
                     secs=np.full(n, np.nan), secs_tight=np.full(n, np.nan),
                     dev_case_ids=np.array([f'{cases[i][0]}|{cases[i][1]}' for i in dev]),
                     W_tighter=np.full((len(dev), 6, st.Rp), np.nan), valid_tighter=np.zeros(len(dev), bool),
                     q2_case_ids=np.array([f'{cases[i][0]}|{cases[i][1]}' for i in q2]),
                     W_qp=np.full((len(q2), 6, st.Rp), np.nan), valid_qp=np.zeros(len(q2), bool),
                     rho=np.full((len(q2), 6), np.nan),
                     W_qpp=np.full((len(dev), 6, st.Rp), np.nan), valid_qpp=np.zeros(len(dev), bool),
                     q3_svals=np.full((len(dev), 3, st.Rp), np.nan))
            co = co_of(scheme)
            for j, ci in enumerate(ids):
                coh, c, ph = cases[ci]
                u0, nu, n0 = U0[ci]
                v, f, secs = go(st, 'main', form, scheme, dt, 'prod', ci, record=True)
                stt = v['stats']
                R['W'][j], R['w0'][j], R['f257'][j], R['n0'][j], R['secs'][j] = np.asarray(v['W']), np.asarray(v['w0']), f, n0, secs
                R['states'][j] = np.asarray(stt['states'])
                R['it'][j], R['reason'][j] = (np.asarray(stt[k])[:S] for k in ('step_it', 'step_reason'))
                if form == 'LSPG':
                    g, _rn = certs['LSPG'](stt['states'], nu, st.data[('main', 'LSPG')], sch)
                    R['cert'][j] = R['gamma_lit'][j] = np.asarray(g)
                    R['rec'][j] = np.asarray(stt['step_gn'])[:S]
                    gok = bool(P.gcert_ok(R['cert'][j], R['rec'][j]) and np.isfinite(R['cert'][j]).all() and np.isfinite(R['rec'][j]).all())
                    nonst = int(np.sum(~(R['cert'][j] <= GAMMA)))
                else:
                    eta, gam, jc, rn_, scl = certs['GAL'](stt['states'], nu, st.data[('main', 'GAL')], sch)
                    R['cert'][j], R['gamma_lit'][j], R['jcond'][j] = np.asarray(eta), np.asarray(gam), np.asarray(jc)
                    R['raw'][j], R['scale'][j] = np.asarray(rn_), np.asarray(scl)
                    R['tolratio'][j] = np.asarray(stt['step_tolratio'])[:S]
                    R['rec'][j] = QC.recorded_eta(R['tolratio'][j], R['scale'][j], t_prod)
                    gok = QC.gcert_gal(R['cert'][j], R['raw'][j], R['scale'][j], R['tolratio'][j], t_prod)
                    nonst = int(np.sum(~QC.certified_gal(R['cert'][j])))
                R['valid'][j] = P.run_valid(v)
                row = dict(setting=sc_['name'], arm=ak, form=form, scheme=scheme, factor=fac, dt=dt, cohort=coh, case=c,
                           kind='prod', seconds=secs, valid=bool(R['valid'][j]), it_sum=int(stt['it_sum']),
                           noncertified=nonst, gcert_ok=gok, budget_exits=P.budget_exits(v), W_sha=sha(v['W']))
                del v
                vt, ft, st_ = go(st, 'main', form, scheme, dt, 'tight', ci)
                R['W_tight'][j], R['valid_tight'][j], R['secs_tight'][j] = np.asarray(vt['W']), P.run_valid(vt), st_
                row.update(dist_tight=P.fdist(f, ft, n0), tight_valid=bool(R['valid_tight'][j]), seconds_tight=st_)
                if ci in dev:
                    k = dev.index(ci)
                    vtt, ftt, _ = go(st, 'main', form, scheme, dt, 'tighter', ci)
                    R['W_tighter'][k], R['valid_tighter'][k] = np.asarray(vtt['W']), P.run_valid(vtt)
                    row.update(dist_tight_tighter=P.fdist(ft, ftt, n0), tighter_valid=bool(R['valid_tighter'][k]))
                    del vtt
                    if arm.get('q3', True):
                        for t_, kk in enumerate((1, 3, 5)):
                            J = np.asarray(jacs[form](jnp.asarray(R['W'][j, kk]), jnp.asarray(co), dt, nu, st.data[('main', form)]))
                            R['q3_svals'][k, t_] = np.linalg.svd(J, compute_uv=False)
                del vt
                if ci in q2:
                    k = q2.index(ci)
                    vq, fq, _ = go(st, 'qp', form, scheme, dt, 'prod', ci)
                    R['W_qp'][k], R['valid_qp'][k] = np.asarray(vq['W']), P.run_valid(vq)
                    if ci in dev and st.rules.get('qpp'):
                        kd = dev.index(ci)
                        vqq, fqq, _ = go(st, 'qpp', form, scheme, dt, 'prod', ci)
                        R['W_qpp'][kd], R['valid_qpp'][kd] = np.asarray(vqq['W']), P.run_valid(vqq)
                        row.update(dist_qp_qpp=P.fdist(fq, fqq, n0), qpp_valid=bool(R['valid_qpp'][kd]))
                        del vqq
                    for kk in range(6):
                        cvec = jnp.asarray(R['W'][j, kk])
                        R['rho'][k, kk] = P.rho(np.asarray(nl(cvec, st.data[('main', 'LSPG')])),
                                                np.asarray(nl(cvec, st.data[('qp', 'LSPG')])))
                    row.update(dist_qp=P.fdist(f, fq, n0), qp_valid=bool(R['valid_qp'][k]), rho_max=float(np.nanmax(R['rho'][k])))
                    del vq
                rep['rows'].append(row)
                print('RUN', sc_['name'], ak, coh, c, el(), flush=True)
            R.update(A=A, lam=np.asarray(st.base['lam']), kx=np.asarray(st.kx), ky=np.asarray(st.ky), time_tag='t0,t1,t2,t3,t4,t5',
                     scheme=scheme, form=form, factor=fac, dt=dt, steps=S, rule=st.rules['main'], Rp=st.Rp, mesh=L,
                     policy=st.policy, mode_hash=st.mode_hash, level='prod', tol=np.array(P.tol_of(form, 'prod')), budget=600,
                     weighting='LSPG D = 1/(a0 + dt b0 nu lam)' if form == 'LSPG' else 'GAL Q^T R = 0, A = Q Rm',
                     cert_kind='gamma (LSPG, autodiff)' if form == 'LSPG' else 'eta = ||Q^T R|| / ||A c_n|| (GAL, autodiff)',
                     metric_id_target='2|lattice=256|times=0:0.05:0.25|norm=fixed_t0|max_k>=1')
            # jcp-regimes edit R2: the decoded lattice fields go to their own (cluster-only) file; the pulled R file keeps
            # per-time norms and a fixed probe of 4096 lattice values per field for the offline decode check
            F = R.pop('f257')
            R['f257_tnorm'] = np.sqrt(np.sum(F.reshape(F.shape[0], 6, -1) ** 2, axis=2))
            R['f257_probe_idx'] = PROBE
            R['f257_probe'] = F.reshape(F.shape[0], 6, -1)[:, :, PROBE]
            R['f257_sha256'] = np.array(sha(F))
            np_save(out / f"F_{sc_['name']}_{ak}.npz", f257=F, case_ids=R['case_ids'])
            del F
            np_save(out / f"R_{sc_['name']}_{ak}.npz", **R)
            srep['arms'][ak] = dict(n=n, mem=P.mem(), steps=S, dt=dt)
            rep['elapsed'][f"{sc_['name']}|{ak}"] = time.perf_counter() - t_arm
            save()
        for blk in cfg.get('blocks', []):
            if blk['setting'] != sc_['name']:
                continue
            t_blk = time.perf_counter()
            if blk['kind'] == 'controls':
                rep['controls'] = run_controls(blk, bank, st, sc_, cases, go, U0, out, certs)
            elif blk['kind'] == 'order':
                rep.setdefault('order', {})[sc_['name']] = run_order(blk, st, cases, go, U0, out)
            rep['elapsed'][f"{sc_['name']}|block|{blk['kind']}"] = time.perf_counter() - t_blk
            save()
        rep['memory'][sc_['name']] = P.mem()
        QQ.c.clear()
        del st
        gc.collect()
        save()
    rep['memory']['end'] = P.mem()
    rep['elapsed_seconds'] = el()
    rep['complete'] = True
    save()
    G.check_write(out / 'COMPLETE').write_text('complete\n')
    print('QACC COMPLETE', el(), flush=True)


def run_order(blk, st, cases, go, U0, out):
    """B4 (DESIGN 5): per (form, scheme) and dev case, tight solves at factors + anchor + anchor check; tighter at the
    finest ladder step; q+ at the finest step and at the anchor; q++ at the finest step. Errors and checks offline."""
    res = {}
    fin, an, an2 = blk['finest'], blk['anchor'], blk['anchor2']
    for arm in blk['arms']:
        form, scheme = arm['form'], arm['scheme']
        ak = f'{form}_{scheme}'
        ids = P.select(cases, arm.get('cases', blk['cases']))
        facs = list(arm['factors']) + [an, an2]
        Wd = {}
        valid = {}
        for ci in ids:
            cid = f'{cases[ci][0]}|{cases[ci][1]}'
            runs = [('main', f_, 'tight') for f_ in facs] + [('main', fin, 'tighter'), ('qp', fin, 'tight'), ('qp', an, 'tight'),
                                                              ('qpp', fin, 'tight')]
            for role, f_, lev in runs:
                v, _f, _ = go(st, role, form, scheme, DT0 * f_, lev, ci)
                k = f'{cid}|{role}|{f_:g}|{lev}'
                Wd[k] = np.asarray(v['W'])
                valid[k] = P.run_valid(v)
                del v
            print('ORDER', st.name, ak, cid, flush=True)
        np_save(out / f'ORDER_{st.name}_{ak}.npz', keys=np.array(list(Wd)), W=np.array(list(Wd.values())))
        res[ak] = dict(cases=[f'{cases[i][0]}|{cases[i][1]}' for i in ids], factors=facs, finest=fin, anchor=an, anchor2=an2,
                       valid=valid, n0={f'{cases[i][0]}|{cases[i][1]}': U0[i][2] for i in ids})
    return res


def run_controls(blk, bank, st, sc_, cases, go, U0, out, certs):
    """Smoke numerical controls (DESIGN 7). Predicates are evaluated offline (qscore / smoke_accept) from saved arrays."""
    res = {}
    ids = P.select(cases, blk['cases'])
    # under-resolved rule (P1 A1.1 fixture): gauss32 vs gauss64, LSPG-BE dt0
    st.add_role(bank, 'ur', blk['under_resolved'], forms=('LSPG', 'GAL'))
    st.add_role(bank, 'ur_succ', blk['under_resolved_successor'], forms=('LSPG', 'GAL'))
    nl = st.q['LSPG']['nl']
    rows, Wu, Ws = [], [], []
    for ci in ids:
        n0 = U0[ci][2]
        v, f, _ = go(st, 'ur', 'LSPG', 'BE', DT0, 'prod', ci)
        w, g, _ = go(st, 'ur_succ', 'LSPG', 'BE', DT0, 'prod', ci)
        Wu.append(np.asarray(v['W']))
        Ws.append(np.asarray(w['W']))
        rho = [P.rho(np.asarray(nl(jnp.asarray(Wu[-1][kk]), st.data[('ur', 'LSPG')])),
                     np.asarray(nl(jnp.asarray(Wu[-1][kk]), st.data[('ur_succ', 'LSPG')]))) for kk in range(6)]
        rows.append(dict(case=f'{cases[ci][0]}|{cases[ci][1]}', dist=P.fdist(f, g, n0), rho=rho, valid=P.run_valid(v),
                         succ_valid=P.run_valid(w), n0=n0))
    res['under_resolved'] = dict(rule=blk['under_resolved'], successor=blk['under_resolved_successor'], rows=rows,
                                 fails_rollout_bar=any(r_['dist'] > 2.5e-4 for r_ in rows),
                                 rho_max=max(max(r_['rho']) for r_ in rows))
    np_save(out / 'CTRL_under_resolved.npz', W=np.array(Wu), W_succ=np.array(Ws), Rp=st.Rp,
                        rho=np.array([r_['rho'] for r_ in rows]), n0=np.array([r_['n0'] for r_ in rows]),
                        succ_valid=np.array([r_['succ_valid'] for r_ in rows]), case_ids=np.array([r_['case'] for r_ in rows]))
    st.drop_role('ur')
    st.drop_role('ur_succ')
    # premature-exit solver (LM budget 0): Q1 negative control, LSPG (P1 A1.6) and GAL (DESIGN 7)
    for form, scheme in (('LSPG', 'CNR'), ('GAL', 'CN')):
        rows, Sb = [], []
        g_, t_ = P.tol_of(form, 'prod')
        sch = T2.sched(scheme, DT0, g_, t_)
        for ci in P.select(cases, blk['b0_cases']):
            v, f, _ = go(st, 'main', form, scheme, DT0, 'prod', ci, record=True, budget=0)
            o = certs[form](v['stats']['states'], U0[ci][1], st.data[('main', form)], sch)
            c_ = np.asarray(o[0])
            bar = GAMMA if form == 'LSPG' else QC.ETA_BAR
            Sb.append(np.asarray(v['stats']['states']))
            rows.append(dict(case=f'{cases[ci][0]}|{cases[ci][1]}', noncertified=int(np.sum(~(c_ <= bar))), steps=len(c_),
                             cert=c_.tolist(), valid=P.run_valid(v)))
        res[f'budget0_{form}'] = dict(rows=rows, scheme=scheme,
                                      pooled_fraction=sum(r_['noncertified'] for r_ in rows) / sum(r_['steps'] for r_ in rows))
        np_save(out / f'CTRL_budget0_{form}.npz', states=np.array(Sb), case_ids=np.array([r_['case'] for r_ in rows]),
                            cert=np.array([r_['cert'] for r_ in rows]),
                            nu=np.array([U0[ci][1] for ci in P.select(cases, blk['b0_cases'])]),
                            A=np.asarray(st.base['A']), lam=np.asarray(st.base['lam']), kx=np.asarray(st.kx), ky=np.asarray(st.ky),
                            scheme=scheme, form=form, rule=st.rules['main'], L=bank.L)
    # zero-update: tight trajectory of dev6|0 per form is saved by the regular arms; the constant trajectory is formed offline
    # wrong-mesh table: a table built on another mesh must be refused by this bank
    small = P.Bank(256 if bank.L != 256 else 128, [0, st.Rp])
    sts = P.Setting(small, 'wrongmesh', st.Rp, st.rules['main'], 'frozen')
    raised = []
    for fn in (lambda: sts.check('main', bank),
               lambda: run(bank, sts, 'main', 'LSPG', 'BE', DT0, 'prod', U0[ids[0]][0], U0[ids[0]][1])):
        try:
            fn()
            raised.append(False)
        except ValueError as e_:
            raised.append(str(e_))
    saved = st.data[('main', 'LSPG')]
    st.data[('main', 'LSPG')] = dict(saved, lam=saved['lam'] * (1 + 1e-6))
    try:
        st.check('main', bank)
        raised.append(False)
    except ValueError as e_:
        raised.append(str(e_))
    st.data[('main', 'LSPG')] = saved
    res['wrong_mesh'] = dict(raised=raised, passed=all(bool(x) for x in raised))
    del sts, small
    gc.collect()
    return res


if __name__ == '__main__':
    main()
