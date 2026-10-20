"""jcp-heads accuracy / qualification job (DESIGN 3, 4, A1.3, A1.4): one mesh per job, one GPU, LSPG at dt0.

Config: {"mesh", "edges", "arms": [{"name", "kind": "head"|"lin", "head" | "Rp", "M"?, "rule", "qp", "qpp",
         "schemes": ["BE", "CNR"], "cases": "all38", "q2_cases"?: "sub16", "q3"?: true}], "blocks": [...]}
Per arm and scheme, per case:
  prod      fused query, internal trajectory recorded, certified in-job (autodiff gamma in the unknowns, harm.cert),
            G-cert against the solver's recorded gamma; validity incl. the head initial-fit veto (harm.run_valid)
  tight     same configuration at the tight level (gtol 1e-7, step budget 2000, head initial fit 2000 / 1e-9), recorded
            and certified: anchor verified iff valid, no budget exit, certified gamma <= 1e-6 at every step and (head)
            initial fit stationary (A1.3); distance to prod on every case
  tighter   dev6 (gtol 1e-8, initial fit 1e-10): verified iff certified gamma <= 1e-7 at every step, initial fit
            stationary; tight/tighter distance
  qp        successor-rule replay on sub16; rho of the arm's own six output coefficient states (main vs qp rule)
  qpp       check-rule replay on dev6; q+/q++ distance
  Q3        singular values of the step Jacobian at output indices 1, 3, 5 of each dev6 trajectory
Blocks (smoke): 'controls' (reproduction gates, Jacobian test, budget-0, under-resolved rule, wrong-mesh).
Outputs: result.json, R_<arm>_<scheme>.npz, CTRL_*.npz, COMPLETE.

    python hacc.py --config configs/<attempt>.json --out <dir>
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
import harm as HA
import jax
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
GAMMA, ANCHOR_T, ANCHOR_TT = 1e-3, 1e-6, 1e-7


def certify(arm, v, nu, role, sch, S):
    """(cert gamma per step, recorded gamma per step, G-cert ok, non-certified count)."""
    st = v['stats']
    g, _ = arm.cert(st['states'], nu, role, sch)
    c_ = np.asarray(g)
    rec = np.asarray(st['step_gn'])[:S]
    gok = bool(P.gcert_ok(c_, rec) and np.isfinite(c_).all() and np.isfinite(rec).all())
    return c_, rec, gok, int(np.sum(~(c_ <= GAMMA)))


def ic_info(v):
    st = v['stats']
    if 'ic_reason' not in st:
        return dict(ic_reason=-1, ic_it=-1, ic_gn=np.nan, ic_rel=np.nan, ic_idx=-1)
    return dict(ic_reason=int(st['ic_reason']), ic_it=int(st['ic_it']), ic_gn=float(st['ic_gn']),
                ic_rel=float(st['ic_rel']), ic_idx=int(st['ic_idx']))


def anchor_verified(v, cert, bar):
    st = v['stats']
    ok = HA.run_valid(v) and int(np.asarray(st['exits'])[0]) == 0 and bool(np.all(np.isfinite(cert) & (cert <= bar)))
    if 'ic_reason' in st:
        ok = ok and int(st['ic_reason']) == 4
    return bool(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    P.check_env(cfg)
    begin = time.perf_counter()
    el = lambda: round(time.perf_counter() - begin, 1)
    L = int(cfg['mesh'])
    rep = dict(config=cfg, commit=P.os.environ.get('SOURCE_COMMIT'), job_id=P.os.environ.get('SLURM_JOB_ID'),
               backend=jax.default_backend(), gpu=jax.devices()[0].device_kind, nvidia_smi=P.gpu_info(),
               jax_version=jax.__version__, mesh=L, checkpoint_sha256=QS.sha_file(T2.CKPT),
               rotation_sha256=QS.sha_file(Q.ROTATION), cohorts=P.cohort_hashes(), arms={}, rows=[], memory={},
               elapsed={}, symdiff={}, complete=False)
    save = lambda: json.dump(QS.clean(rep), open(out / 'result.json', 'w'), indent=1)
    QS.burn(.05)
    rep['memory']['start'] = P.mem()
    bank = P.Bank(L, cfg['edges'], nrb=cfg.get('bank_blocks'))
    rep['bank'] = dict(edges=bank.edges, seconds=bank.build_seconds, bytes=bank.bytes, mem=P.mem())
    cases = P.case_list(cfg)
    s256 = L // 256
    U0 = {}
    for i, (coh, c, ph) in enumerate(cases):
        u0n = Q.e.initial(L, ph)
        U0[i] = (jnp.asarray(u0n), float(ph[4]), float(np.linalg.norm(u0n[::s256, ::s256])))
    cid = lambda ci: f'{cases[ci][0]}|{cases[ci][1]}'
    heads = {}
    S = HA.steps_of(DT0)
    save()

    for spec in cfg['arms']:
        t_arm = time.perf_counter()
        b0 = P.mem()['in_use']
        arm = HA.make_arm(bank, spec, heads, S)
        for role in ('qp', 'qpp'):
            if spec.get(role):
                arm.add_role(role, spec[role])
        M = arm.M
        if str(M) not in rep['symdiff']:
            rep['symdiff'][str(M)] = P.symdiff(L, M)
        d_ = arm.describe()
        d_.update(setting_bytes=P.mem()['in_use'] - b0, spec=spec, arms={})
        if arm.kind == 'head':
            d_['jacobian'] = jacobian_check(arm, cases, U0)
        rep['arms'][spec['name']] = d_
        ids = P.select(cases, spec.get('cases', 'all38'))
        q2 = [i for i in P.select(cases, spec.get('q2_cases', 'sub16')) if i in ids] if spec.get('qp') else []
        dev = [i for i in P.select(cases, 'dev6') if i in ids]
        dim = arm.dim
        for scheme in spec.get('schemes', ['BE']):
            t_s = time.perf_counter()
            sch = HA.sched(scheme, DT0, 'prod')
            sch_t, sch_tt = HA.sched(scheme, DT0, 'tight'), HA.sched(scheme, DT0, 'tighter')
            co = np.array(T2.SCHEMES[scheme][:5], float)
            n, nd = len(ids), len(dev)
            R = dict(case_ids=np.array([cid(i) for i in ids]), dev_case_ids=np.array([cid(i) for i in dev]),
                     q2_case_ids=np.array([cid(i) for i in q2]),
                     W=np.full((n, 6, dim), np.nan), C=np.full((n, 6, 512 if arm.kind == 'head' else dim), np.nan),
                     w0=np.full((n, dim), np.nan), states=np.full((n, S + 1, dim), np.nan),
                     cert=np.full((n, S), np.nan), rec=np.full((n, S), np.nan), it=np.full((n, S), -1),
                     reason=np.full((n, S), -1), f257=np.full((n, 6, 257, 257), np.nan), n0=np.full(n, np.nan),
                     valid=np.zeros(n, bool), secs=np.full(n, np.nan),
                     ic=np.full((n, 5), np.nan),
                     W_tight=np.full((n, 6, dim), np.nan), states_tight=np.full((n, S + 1, dim), np.nan),
                     cert_tight=np.full((n, S), np.nan), valid_tight=np.zeros(n, bool), verified_tight=np.zeros(n, bool),
                     ic_tight=np.full((n, 5), np.nan), budget_exits_tight=np.full(n, -1),
                     W_tighter=np.full((nd, 6, dim), np.nan), states_tighter=np.full((nd, S + 1, dim), np.nan),
                     cert_tighter=np.full((nd, S), np.nan),
                     valid_tighter=np.zeros(nd, bool), verified_tighter=np.zeros(nd, bool), ic_tighter=np.full((nd, 5), np.nan),
                     budget_exits_tighter=np.full(nd, -1),
                     W_qp=np.full((len(q2), 6, dim), np.nan), valid_qp=np.zeros(len(q2), bool),
                     rho=np.full((len(q2), 6), np.nan),
                     W_qpp=np.full((nd, 6, dim), np.nan), valid_qpp=np.zeros(nd, bool),
                     q3_svals=np.full((nd, 3, dim), np.nan))
            icv = lambda v: np.array([ic_info(v)[k] for k in ('ic_reason', 'ic_it', 'ic_gn', 'ic_rel', 'ic_idx')], float)
            for j, ci in enumerate(ids):
                u0, nu, n0 = U0[ci]
                t1 = time.perf_counter()
                v = arm.run('main', scheme, 'prod', u0, nu, DT0, record=True)
                jax.block_until_ready(v)
                secs = time.perf_counter() - t1
                f = P.restrict(v['fields'], L)
                stt = v['stats']
                c_, rec, gok, nonst = certify(arm, v, nu, 'main', sch, S)
                R['W'][j], R['w0'][j], R['f257'][j], R['n0'][j], R['secs'][j] = np.asarray(v['W']), np.asarray(v['w0']), f, n0, secs
                R['C'][j] = arm.coef(v['W'])
                R['states'][j] = np.asarray(stt['states'])
                R['it'][j], R['reason'][j] = (np.asarray(stt[k])[:S] for k in ('step_it', 'step_reason'))
                R['cert'][j], R['rec'][j] = c_, rec
                R['valid'][j] = HA.run_valid(v)
                R['ic'][j] = icv(v)
                row = dict(arm=spec['name'], scheme=scheme, case=cid(ci), seconds=secs, valid=bool(R['valid'][j]),
                           it_sum=int(stt['it_sum']), noncertified=nonst, gcert_ok=gok, budget_exits=P.budget_exits(v),
                           W_sha=sha(v['W']), **ic_info(v))
                del v
                vt = arm.run('main', scheme, 'tight', u0, nu, DT0, record=True)
                jax.block_until_ready(vt)
                ct, _, _, _ = certify(arm, vt, nu, 'main', sch_t, S)
                ft = P.restrict(vt['fields'], L)
                R['W_tight'][j], R['states_tight'][j], R['cert_tight'][j] = np.asarray(vt['W']), np.asarray(vt['stats']['states']), ct
                R['valid_tight'][j], R['verified_tight'][j] = HA.run_valid(vt), anchor_verified(vt, ct, ANCHOR_T)
                R['ic_tight'][j], R['budget_exits_tight'][j] = icv(vt), P.budget_exits(vt)
                row.update(dist_tight=P.fdist(f, ft, n0), tight_valid=bool(R['valid_tight'][j]),
                           tight_verified=bool(R['verified_tight'][j]))
                del vt
                if ci in dev:
                    k = dev.index(ci)
                    vtt = arm.run('main', scheme, 'tighter', u0, nu, DT0, record=True)
                    jax.block_until_ready(vtt)
                    ctt, _, _, _ = certify(arm, vtt, nu, 'main', sch_tt, S)
                    R['W_tighter'][k], R['cert_tighter'][k] = np.asarray(vtt['W']), ctt
                    R['states_tighter'][k] = np.asarray(vtt['stats']['states'])
                    R['valid_tighter'][k], R['verified_tighter'][k] = HA.run_valid(vtt), anchor_verified(vtt, ctt, ANCHOR_TT)
                    R['ic_tighter'][k], R['budget_exits_tighter'][k] = icv(vtt), P.budget_exits(vtt)
                    row.update(dist_tight_tighter=P.fdist(ft, P.restrict(vtt['fields'], L), n0),
                               tighter_valid=bool(R['valid_tighter'][k]), tighter_verified=bool(R['verified_tighter'][k]))
                    del vtt
                    if spec.get('q3', True):
                        for t_, kk in enumerate((1, 3, 5)):
                            J = arm.jac(R['W'][j, kk], co, DT0, nu)
                            R['q3_svals'][k, t_] = np.linalg.svd(J, compute_uv=False) if np.isfinite(J).all() else np.nan
                if ci in q2:
                    k = q2.index(ci)
                    vq = arm.run('qp', scheme, 'prod', u0, nu, DT0, record=False)
                    jax.block_until_ready(vq)
                    fq = P.restrict(vq['fields'], L)
                    R['W_qp'][k], R['valid_qp'][k] = np.asarray(vq['W']), HA.run_valid(vq)
                    for kk in range(6):
                        R['rho'][k, kk] = P.rho(arm.nl(R['C'][j, kk], 'main'), arm.nl(R['C'][j, kk], 'qp'))
                    row.update(dist_qp=P.fdist(f, fq, n0), qp_valid=bool(R['valid_qp'][k]), rho_max=float(np.nanmax(R['rho'][k])))
                    if ci in dev and spec.get('qpp'):
                        kd = dev.index(ci)
                        vqq = arm.run('qpp', scheme, 'prod', u0, nu, DT0, record=False)
                        jax.block_until_ready(vqq)
                        R['W_qpp'][kd], R['valid_qpp'][kd] = np.asarray(vqq['W']), HA.run_valid(vqq)
                        row.update(dist_qp_qpp=P.fdist(fq, P.restrict(vqq['fields'], L), n0), qpp_valid=bool(R['valid_qpp'][kd]))
                        del vqq
                    del vq
                rep['rows'].append(row)
                print('RUN', spec['name'], scheme, cid(ci), el(), flush=True)
            R.update(A=np.asarray(arm.data('main')['A']), lam=np.asarray(arm.data('main')['lam']),
                     time_tag='t0,t1,t2,t3,t4,t5', scheme=scheme, form='LSPG', factor=1., dt=DT0, steps=S,
                     rule=arm.rules['main'], kind=arm.kind, dim=dim, M=M, mesh=L, mode_hash=arm.describe()['mode_hash'],
                     level='prod', tol=np.array(P.tol_of('LSPG', 'prod')), budget=600,
                     head_sha256=arm.describe().get('head_sha256', ''), ic_cols=np.array(['reason', 'it', 'gn', 'rel', 'idx']),
                     metric_id_target='2|lattice=256|times=0:0.05:0.25|norm=fixed_t0|max_k>=1')
            np.savez_compressed(out / f"R_{spec['name']}_{scheme}.npz", **R)
            rep['arms'][spec['name']]['arms'][scheme] = dict(n=n, seconds=time.perf_counter() - t_s, mem=P.mem())
            save()
        for blk in cfg.get('blocks', []):
            if blk.get('arm') == spec['name'] and blk['kind'] == 'controls':
                t_b = time.perf_counter()
                rep['controls'] = run_controls(blk, bank, arm, spec, cases, U0, out, heads, S)
                rep['elapsed']['controls'] = time.perf_counter() - t_b
                save()
        rep['elapsed'][spec['name']] = time.perf_counter() - t_arm
        rep['memory'][spec['name']] = P.mem()
        del arm
        gc.collect()
        save()
    rep['memory']['end'] = P.mem()
    rep['elapsed_seconds'] = el()
    rep['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('HACC COMPLETE', el(), flush=True)


def jacobian_check(arm, cases, U0):
    """DESIGN 7 'head Jacobian': the solver's analytic J_z vs forward-mode autodiff of the same residual, at 4 fixtures
    (states 1 -> 2 and 25 -> 26 of dev6 cases 0-1, BE), for EVERY head arm."""
    q = arm.parts('prod', True)
    rows = []
    for ci in P.select(cases, 'dev6')[:2]:
        u0, nu, n0 = U0[ci]
        v = arm.run('main', 'BE', 'prod', u0, nu, DT0, record=True)
        Zs = np.asarray(v['stats']['states'])
        d = arm.data('main')
        A = d['A']
        be = jnp.asarray(T2.BE_CO)
        for k in (1, 25):
            ck = arm.hz(jnp.asarray(Zs[k]))
            args = ((A @ ck, A @ ck, jnp.zeros(A.shape[0])), nu, d, be, DT0)
            r_an, J_an = q['evalJ'](jnp.asarray(Zs[k + 1]), args)
            J_ad = jax.jacfwd(lambda z: q['res'](z, args))(jnp.asarray(Zs[k + 1]))
            rows.append(dict(case=f'{cases[ci][0]}|{cases[ci][1]}', step=k,
                             rel=float(jnp.linalg.norm(J_an - J_ad) / jnp.linalg.norm(J_ad))))
    return dict(rows=rows, worst=max(r_['rel'] for r_ in rows), passed=all(r_['rel'] <= 1e-10 for r_ in rows))


def run_controls(blk, bank, arm, spec, cases, U0, out, heads, S):
    """Smoke controls (DESIGN 7, A1.3, A1.7). Predicates are recomputed offline (smoke_accept.py) from the saved arrays."""
    L = bank.L
    res = {}
    cid = lambda ci: f'{cases[ci][0]}|{cases[ci][1]}'
    dev = P.select(cases, 'dev6')
    sch = HA.sched('BE', DT0, 'prod')
    # (1) head reproduction gate: lane head stepper vs quadrature-study's archived internal trajectories (incumbent)
    if blk.get('repro_head'):
        rows = []
        for rl, fx in blk['repro_head'].items():
            sp = dict(name=f'repro_{rl}', kind='head', head=spec['head'], rule=rl)
            ar = HA.make_arm(bank, sp, heads, S)
            for cname, path in fx.items():
                ci = P.select(cases, [tuple(cname.split('|')[:1]) + (int(cname.split('|')[1]),)])[0]
                u0, nu, n0 = U0[ci]
                v = ar.run('main', 'BE', 'prod', u0, nu, DT0, record=True)
                Zi = np.asarray(v['stats']['states'])
                ref = np.load(HERE / path)['internal']
                rel = float(np.max(np.linalg.norm(Zi - ref, axis=1)) / np.max(np.linalg.norm(ref, axis=1)))
                rows.append(dict(rule=rl, case=cname, rel=rel, fixture=path))
            del ar
            gc.collect()
        res['repro_head'] = dict(rows=rows, worst=max(r_['rel'] for r_ in rows), passed=all(r_['rel'] <= 1e-10 for r_ in rows))
    # (2) linear reproduction gate: lane driver vs combined's c2grid run (R384 gauss128, LSPG-BE dt0, dev6)
    if blk.get('repro_lin'):
        rl = blk['repro_lin']
        ar = HA.make_arm(bank, dict(name='repro_lin', kind='lin', Rp=rl['Rp'], rule=rl['rule']), heads, S)
        rows = []
        for scheme, fixture in rl['fixtures'].items():
            fx = np.load(HERE / fixture)
            fc = [str(x) for x in fx['cases']]
            for ci in dev:
                u0, nu, n0 = U0[ci]
                v = ar.run('main', scheme, 'prod', u0, nu, DT0, record=False)
                Wa = fx['W'][fc.index(cid(ci))]
                rows.append(dict(scheme=scheme, case=cid(ci), fixture=fixture,
                                 rel=float(np.max(np.linalg.norm(np.asarray(v['W']) - Wa, axis=1)) / np.linalg.norm(Wa[0]))))
        res['repro_lin'] = dict(rows=rows, worst=max(r_['rel'] for r_ in rows), passed=all(r_['rel'] <= 1e-10 for r_ in rows))
        del ar
        gc.collect()
    # (4) Q1 negative: step budget 0 (returns the predictor), BE, dev6
    sp0 = dict(spec, name='budget0')
    if arm.kind == 'head':
        import hcommon as HM
        st0 = HM.HeadSetting(bank, 'budget0', arm.head, spec['rule'], nrec=S + 1, nstep=max(T2.NSTEP, S + 1), budget=0, M=spec.get('M'))
        runner = lambda u0, nu: st0.query('prod', True)['fused'](u0, nu, st0.data['main'], st0.cold, sch)
        data0 = st0.data['main']
    else:
        q0 = T2.make_split_query(Q, 'point', 'LSPG', arm.dim, L, arm.st.trust, budget=0, nrec=S + 1, nstep=max(T2.NSTEP, S + 1))
        runner = lambda u0, nu: q0['fused'](u0, nu, arm.data('main'), arm.cold, sch)
        data0 = arm.data('main')
    rows, Sb, NU = [], [], []
    for ci in dev:
        u0, nu, n0 = U0[ci]
        v = runner(u0, nu)
        c_, _ = arm.cert(v['stats']['states'], nu, 'main', sch)
        c_ = np.asarray(c_)
        Sb.append(np.asarray(v['stats']['states']))
        NU.append(nu)
        rows.append(dict(case=cid(ci), noncertified=int(np.sum(~(c_ <= GAMMA))), steps=len(c_), valid=HA.run_valid(v)))
    res['budget0'] = dict(rows=rows, pooled_fraction=sum(r_['noncertified'] for r_ in rows) / sum(r_['steps'] for r_ in rows))
    np.savez_compressed(out / 'CTRL_budget0.npz', states=np.array(Sb), nu=np.array(NU), case_ids=np.array([cid(c) for c in dev]),
                        A=np.asarray(data0['A']), lam=np.asarray(data0['lam']), rule=spec['rule'], kind=arm.kind,
                        head_sha256=arm.describe().get('head_sha256', ''), M=arm.M, L=L)
    # (5) under-resolved rule: q_u -> q_u+ (q_u++ required valid), BE, dev6
    ur = blk['under_resolved']
    for role, rl in zip(('ur', 'ur_p', 'ur_pp'), ur):
        arm.add_role(role, rl)
    rows, Wu, Ws, Wss = [], [], [], []
    for ci in dev:
        u0, nu, n0 = U0[ci]
        v = arm.run('ur', 'BE', 'prod', u0, nu, DT0, record=False)
        w = arm.run('ur_p', 'BE', 'prod', u0, nu, DT0, record=False)
        x = arm.run('ur_pp', 'BE', 'prod', u0, nu, DT0, record=False)
        Wu.append(np.asarray(v['W'])); Ws.append(np.asarray(w['W'])); Wss.append(np.asarray(x['W']))
        Cu = arm.coef(v['W'])
        rho = [P.rho(arm.nl(Cu[kk], 'ur'), arm.nl(Cu[kk], 'ur_p')) for kk in range(6)]
        rows.append(dict(case=cid(ci), dist=P.fdist(P.restrict(v['fields'], L), P.restrict(w['fields'], L), n0), rho=rho,
                         valid=HA.run_valid(v), succ_valid=HA.run_valid(w), check_valid=HA.run_valid(x),
                         dist_check=P.fdist(P.restrict(w['fields'], L), P.restrict(x['fields'], L), n0)))
    res['under_resolved'] = dict(rules=ur, rows=rows, fails_rollout_bar=any(r_['dist'] > 2.5e-4 for r_ in rows),
                                 rho_max=max(max(r_['rho']) for r_ in rows),
                                 successors_valid=all(r_['succ_valid'] and r_['check_valid'] for r_ in rows))
    np.savez_compressed(out / 'CTRL_under_resolved.npz', W=np.array(Wu), W_succ=np.array(Ws), W_check=np.array(Wss),
                        case_ids=np.array([cid(c) for c in dev]))
    for role in ('ur', 'ur_p', 'ur_pp'):
        arm.drop_role(role)
    # (6) wrong-mesh table and substituted eigenvalues must be refused
    small = P.Bank(256 if L != 256 else 128, [0, 4, 512] if arm.kind == 'head' else [0, arm.dim])
    sm = HA.make_arm(small, dict(spec, name='wrongmesh'), heads, S)
    raised = []
    for fn in (lambda: sm.st.check('main', bank), lambda: arm.st.check('main', small)):
        try:
            fn()
            raised.append(False)
        except ValueError as e_:
            raised.append(str(e_))
    d0 = arm.data('main')
    key = 'main' if arm.kind == 'head' else ('main', 'LSPG')
    arm.st.data[key] = dict(d0, lam=d0['lam'] * (1 + 1e-6))
    try:
        arm.check('main')
        raised.append(False)
    except ValueError as e_:
        raised.append(str(e_))
    arm.st.data[key] = d0
    res['wrong_mesh'] = dict(raised=raised, passed=all(bool(x) for x in raised))
    del sm, small
    gc.collect()
    return res


if __name__ == '__main__':
    main()
