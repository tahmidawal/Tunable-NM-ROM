"""jcp-p1-scaling accuracy / qualification job (DESIGN 2, 5, 6, 7): one mesh per job, one GPU.

Per configuration (name, r, rule, mode policy) and scheme (LSPG BE / CN-R at dt0), per case of its case set:
  prod      fused query, internal trajectory recorded; certified in-job (p1core.make_certifier, autodiff Jacobian, every
            step) and compared with the solver's recorded ratio (G-cert)
  tight     same configuration at the tight tolerance (Q1 distance), every case
  tighter   dev6 cases (Q1 tight/tighter agreement)
  qp        successor rule replay on the Q2 population (sub16); rho of the arm's own six output states vs the successor
  qpp       check rule replay on dev6 (finer-rule convergence)
  Q3        singular values of A and of the LSPG Jacobian at output indices 1, 3, 5 of each dev6 case
Optional blocks: reproduction gate vs the combined archive (1024^2), P1e-2D failing rule, the CN-R temporal-order check,
the smoke controls. In-job distances use the decoded full-field outputs restricted to the 257^2 scoring lattice; the
offline checker recomputes them from the saved coefficients with an independent NumPy decode.

    python p1acc.py --config configs/<attempt>.json --out <dir> [--dry-run]
Outputs: result.json, R_<config>_<scheme>.npz (per-run arrays), extra npz per block, COMPLETE.
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
import jax
import jax.numpy as jnp

NREC = int(round(.25 / DT0)) + 1
BE_CO = np.array(T2.BE_CO)


def co_of(scheme):
    return np.array(T2.SCHEMES[scheme][:5], float)


def plan_runs(cfg, cases):
    """Dry-run inventory: (config, scheme, kind, n cases, steps per run)."""
    inv = []
    for c in cfg['configs']:
        n = len(P.select(cases, c['cases']))
        nq = len(set(P.select(cases, c.get('q2_cases', 'sub16'))) & set(P.select(cases, c['cases'])))
        nd = len(set(P.select(cases, 'dev6')) & set(P.select(cases, c['cases'])))
        for sc in c['schemes']:
            inv += [(c['name'], sc, 'prod', n), (c['name'], sc, 'tight', n), (c['name'], sc, 'tighter', nd),
                    (c['name'], sc, 'qp', nq if c.get('qp') else 0), (c['name'], sc, 'qpp', nd if c.get('qpp') else 0)]
    return inv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    cases = P.case_list(cfg)
    if a.dry_run:
        inv = plan_runs(cfg, cases)
        print(json.dumps(dict(runs=inv, total=sum(x[3] for x in inv)), indent=0))
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    P.check_env(cfg)
    begin = time.perf_counter()
    el = lambda: round(time.perf_counter() - begin, 1)
    L = int(cfg['mesh'])
    rep = dict(config=cfg, commit=P.os.environ.get('SOURCE_COMMIT'), job_id=P.os.environ.get('SLURM_JOB_ID'),
               backend=jax.default_backend(), gpu=jax.devices()[0].device_kind, nvidia_smi=P.gpu_info(),
               jax_version=jax.__version__, mesh=L, checkpoint_sha256=QS.sha_file(T2.CKPT),
               rotation_sha256=QS.sha_file(Q.ROTATION), cohorts=P.cohort_hashes(), setup={}, rows=[], configs={},
               memory={}, phases={}, complete=False)
    save = lambda: json.dump(QS.clean(rep), open(out / 'result.json', 'w'), indent=1)
    QS.burn(.05)
    rep['memory']['start'] = P.mem()
    bank = P.Bank(L, cfg['edges'], nrb=cfg.get('bank_blocks'))
    rep['setup']['bank'] = dict(edges=bank.edges, seconds=bank.build_seconds, bytes=bank.bytes, mem=P.mem())
    for pol in sorted({c.get('policy', 'frozen') for c in cfg['configs']}):
        ops = bank.ops(pol)
        rep['setup'][f'ops_{pol}'] = dict(seconds=ops['seconds'], list_hash_2048=P.list_hash(ops['kx'], ops['ky']),
                                          mem=P.mem())
    rep['setup']['symdiff'] = {str(M): P.symdiff(L, M) for M in (128, 256, 512, 1024, 1536, 2048)}
    s256 = L // 256
    U0 = {}
    for i, (coh, c, ph) in enumerate(cases):
        u0n = Q.e.initial(L, ph)
        U0[i] = (jnp.asarray(u0n), float(ph[4]), float(np.linalg.norm(u0n[::s256, ::s256])))
    rep['phases']['setup'] = el()
    print('SETUP', el(), P.mem(), flush=True)
    save()

    def go(st, role, form, scheme, dt, level, ci, record=False):
        u0, nu, n0 = U0[ci]
        t1 = time.perf_counter()
        v = P.run_query(bank, st, role, form, scheme, dt, level, u0, nu, record=record)
        jax.block_until_ready(v)
        secs = time.perf_counter() - t1
        f = P.restrict(v['fields'], L)
        return v, f, secs

    # ------------------------------------------------------------ configurations ----
    for cc in cfg['configs']:
        roles = {}
        if cc.get('qp'):
            roles['qp'] = cc['qp']
        if cc.get('qpp'):
            roles['qpp'] = cc['qpp']
        roles.update(cc.get('extra_roles', {}))
        b0 = P.mem()['in_use']
        t0 = time.perf_counter()
        st = P.Setting(bank, cc['name'], cc['Rp'], cc['rule'], cc.get('policy', 'frozen'), roles=roles,
                       forms=tuple(cc.get('forms', ['LSPG'])), nrec=NREC, budget=cc.get('budget', 600),
                       mutation=cc.get('mutation'))
        cert = T2.make_certifier(st.q['LSPG']['nl'], st.Rp)
        jac = T2.make_jac(st.q['LSPG']['nl'])
        A = np.asarray(st.base['A'])
        sA = np.linalg.svd(A, compute_uv=False)
        crep = rep['configs'][cc['name']] = dict(Rp=st.Rp, M=st.M, policy=st.policy, rules=st.rules, m=st.m,
                                                 meta=st.meta, trust=st.trust, table_bytes=st.table_bytes,
                                                 setting_bytes=P.mem()['in_use'] - b0, setup_seconds=time.perf_counter() - t0,
                                                 A_svals=sA.tolist(), A_cond=float(sA[0] / sA[-1]),
                                                 A_rank=int(np.sum(sA > sA[0] * 1e-13)), schemes={})
        ids = P.select(cases, cc['cases'])
        q2 = [i for i in P.select(cases, cc.get('q2_cases', 'sub16')) if i in ids]
        dev = [i for i in P.select(cases, 'dev6') if i in ids]
        for sc in cc['schemes']:
            n = len(ids)
            R = dict(case_ids=np.array([f'{cases[i][0]}|{cases[i][1]}' for i in ids]),
                     W=np.full((n, 6, st.Rp), np.nan), w0=np.full((n, st.Rp), np.nan),
                     states=np.full((n, NREC, st.Rp), np.nan), gamma=np.full((n, NREC - 1), np.nan),
                     gn_rec=np.full((n, NREC - 1), np.nan), it=np.full((n, NREC - 1), -1), reason=np.full((n, NREC - 1), -1),
                     f257=np.full((n, 6, 257, 257), np.nan), n0=np.full(n, np.nan), W_tight=np.full((n, 6, st.Rp), np.nan),
                     valid=np.zeros(n, bool), valid_tight=np.zeros(n, bool), secs=np.full(n, np.nan),
                     secs_tight=np.full(n, np.nan))
            if cc.get('qp'):
                R.update(q2_ids=np.array(q2), W_qp=np.full((len(q2), 6, st.Rp), np.nan), valid_qp=np.zeros(len(q2), bool),
                         rho=np.full((len(q2), 6), np.nan))
            R.update(dev_ids=np.array(dev), W_tighter=np.full((len(dev), 6, st.Rp), np.nan),
                     valid_tighter=np.zeros(len(dev), bool))
            if cc.get('qpp'):
                R.update(W_qpp=np.full((len(dev), 6, st.Rp), np.nan), valid_qpp=np.zeros(len(dev), bool))
            if cc.get('q3', True):
                R.update(q3_svals=np.full((len(dev), 3, st.Rp), np.nan))
            co = co_of(sc)
            for j, ci in enumerate(ids):
                coh, c, ph = cases[ci]
                u0, nu, n0 = U0[ci]
                v, f, secs = go(st, 'main', 'LSPG', sc, DT0, 'prod', ci, record=True)
                stt = v['stats']
                R['W'][j], R['w0'][j], R['f257'][j], R['n0'][j], R['secs'][j] = np.asarray(v['W']), np.asarray(v['w0']), f, n0, secs
                R['states'][j] = np.asarray(stt['states'])
                S = NREC - 1
                R['gn_rec'][j], R['it'][j], R['reason'][j] = (np.asarray(stt[k])[:S] for k in ('step_gn', 'step_it', 'step_reason'))
                g, _rn = cert(stt['states'], nu, st.data[('main', 'LSPG')], st.sched('LSPG', sc, DT0, 'prod'))
                R['gamma'][j] = np.asarray(g)
                R['valid'][j] = P.run_valid(v)
                row = dict(config=cc['name'], scheme=sc, cohort=coh, case=c, kind='prod', seconds=secs, valid=bool(R['valid'][j]),
                           it_sum=int(stt['it_sum']), nonstationary=int(np.sum(~(np.asarray(g) <= 1e-3))),
                           gcert_rel=float(np.max(np.abs(np.asarray(g) - R['gn_rec'][j]) / np.maximum(R['gn_rec'][j], 1e-300))),
                           gcert_ok=P.gcert_ok(g, R['gn_rec'][j]), budget_exits=P.budget_exits(v), W_sha=sha(v['W']))
                del v
                vt, ft, st_ = go(st, 'main', 'LSPG', sc, DT0, 'tight', ci)
                R['W_tight'][j], R['valid_tight'][j], R['secs_tight'][j] = np.asarray(vt['W']), P.run_valid(vt), st_
                row.update(dist_tight=P.fdist(f, ft, n0), tight_valid=bool(R['valid_tight'][j]), seconds_tight=st_)
                if ci in dev:
                    k = dev.index(ci)
                    vtt, ftt, _ = go(st, 'main', 'LSPG', sc, DT0, 'tighter', ci)
                    R['W_tighter'][k], R['valid_tighter'][k] = np.asarray(vtt['W']), P.run_valid(vtt)
                    row.update(dist_tight_tighter=P.fdist(ft, ftt, n0), tighter_valid=bool(R['valid_tighter'][k]))
                    del vtt
                    if cc.get('q3', True):
                        for t_, kk in enumerate((1, 3, 5)):
                            J = np.asarray(jac(jnp.asarray(R['W'][j, kk]), jnp.asarray(co), DT0, nu, st.data[('main', 'LSPG')]))
                            R['q3_svals'][k, t_] = np.linalg.svd(J, compute_uv=False)
                del vt
                if ci in q2 and cc.get('qp'):
                    k = q2.index(ci)
                    vq, fq, _ = go(st, 'qp', 'LSPG', sc, DT0, 'prod', ci)
                    R['W_qp'][k], R['valid_qp'][k] = np.asarray(vq['W']), P.run_valid(vq)
                    if ci in dev and cc.get('qpp'):
                        kd = dev.index(ci)
                        vqq, fqq, _ = go(st, 'qpp', 'LSPG', sc, DT0, 'prod', ci)
                        R['W_qpp'][kd], R['valid_qpp'][kd] = np.asarray(vqq['W']), P.run_valid(vqq)
                        row.update(dist_qp_qpp=P.fdist(fq, fqq, n0), qpp_valid=bool(R['valid_qpp'][kd]))
                        del vqq
                    nl = st.q['LSPG']['nl']
                    for kk in range(6):
                        cvec = jnp.asarray(R['W'][j, kk])
                        a_ = np.asarray(nl(cvec, st.data[('main', 'LSPG')]))
                        b_ = np.asarray(nl(cvec, st.data[('qp', 'LSPG')]))
                        R['rho'][k, kk] = P.rho(a_, b_)
                    row.update(dist_qp=P.fdist(f, fq, n0), qp_valid=bool(R['valid_qp'][k]), rho_max=float(np.nanmax(R['rho'][k])))
                    del vq
                rep['rows'].append(row)
                print('RUN', cc['name'], sc, coh, c, el(), flush=True)
            R.update(A=A, lam=np.asarray(st.base['lam']), kx=np.asarray(st.kx), ky=np.asarray(st.ky))
            np.savez_compressed(out / f"R_{cc['name']}_{sc}.npz", **R)
            crep['schemes'][sc] = dict(n=len(ids), mem=P.mem())
            save()
        # reproduction gate (AM-1)
        rp = (cfg.get('reproduce') or {}).get(cc['name'])
        if rp:
            z = np.load(rp['archive'])
            runs, acases = list(z['runs']), list(z['cases'])
            g = []
            for sc in cc['schemes']:
                Rz = np.load(out / f"R_{cc['name']}_{sc}.npz")
                ri = runs.index(rp['runs'][sc])
                for j, cid in enumerate(Rz['case_ids']):
                    Wa = z['W'][acases.index(cid), ri]
                    g.append(dict(scheme=sc, case=str(cid), rel=float(np.max(np.linalg.norm(Rz['W'][j] - Wa, axis=1)) /
                                                                 np.linalg.norm(Wa[0])),
                                  w0_rel=float(np.linalg.norm(Rz['w0'][j] - z['w0'][acases.index(cid)]) /
                                               np.linalg.norm(z['w0'][acases.index(cid)]))))
            crep['reproduction'] = dict(archive=rp['archive'], archive_sha256=QS.sha_file(rp['archive']), rows=g,
                                        worst=max(x['rel'] for x in g), passed=bool(max(x['rel'] for x in g) <= 1e-10))
            print('REPRODUCTION', cc['name'], crep['reproduction']['worst'], flush=True)
        # extra blocks bound to this configuration
        for blk in cfg.get('blocks', []):
            if blk['config'] != cc['name']:
                continue
            if blk['kind'] == 'p1e':
                rep.setdefault('p1e', {})[cc['name']] = run_p1e(blk, bank, st, cases, go, U0, out)
            elif blk['kind'] == 'order':
                rep['order'] = run_order(blk, bank, cc, cases, go, U0, out)
            elif blk['kind'] == 'controls':
                rep['controls'] = run_controls(blk, bank, st, cc, cases, go, U0, out, cert)
            elif blk['kind'] == 'vendor':
                rep.setdefault('vendor_bridge', {})[cc['name']] = run_vendor(blk, bank, st, cases, go, U0)
            save()
        rep['memory'][cc['name']] = P.mem()
        del st
        gc.collect()
        save()
    rep['memory']['end'] = P.mem()
    rep['elapsed_seconds'] = el()
    rep['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('P1ACC COMPLETE', el(), flush=True)


def run_p1e(blk, bank, st, cases, go, U0, out):
    """P1e-2D (DESIGN 2.5): failing rule and its successor on dev6, BE; the deployed / finer / check runs come from the
    configuration's own prod / qp / qpp runs (recomputed here for the pairing)."""
    st.add_role(bank, 'fail', blk['failing'])
    st.add_role(bank, 'fail_succ', blk['successor'])
    ids = P.select(cases, blk['cases'])
    res = dict(rules=dict(fail=blk['failing'], fail_succ=blk['successor'], deployed=st.rules['main'], finer=st.rules.get('qp'),
                          check=st.rules.get('qpp')), m=dict(st.m), rows=[])
    Wd = {k: np.full((len(ids), 6, st.Rp), np.nan) for k in ('fail', 'fail_succ', 'main', 'qp', 'qpp')}
    nl = st.q['LSPG']['nl']
    for j, ci in enumerate(ids):
        n0 = U0[ci][2]
        F = {}
        for role in ('fail', 'fail_succ', 'main', 'qp', 'qpp'):
            v, f, _ = go(st, role, 'LSPG', blk['scheme'], DT0, 'prod', ci)
            Wd[role][j], F[role] = np.asarray(v['W']), f
            F[role + '_valid'] = P.run_valid(v)
        rho = {}
        for a_, b_ in (('fail', 'fail_succ'), ('fail', 'main'), ('main', 'qp'), ('qp', 'qpp')):
            r_ = []
            for kk in range(6):
                c_ = jnp.asarray(Wd[a_][j, kk])
                x_, y_ = np.asarray(nl(c_, st.data[(a_, 'LSPG')])), np.asarray(nl(c_, st.data[(b_, 'LSPG')]))
                r_.append(P.rho(x_, y_))
            rho[f'{a_}|{b_}'] = r_
        res['rows'].append(dict(cohort=cases[ci][0], case=cases[ci][1],
                                dist={f'{a_}|{b_}': P.fdist(F[a_], F[b_], n0) for a_, b_ in
                                      (('fail', 'fail_succ'), ('fail', 'main'), ('main', 'qp'), ('qp', 'qpp'))},
                                valid={r_: F[r_ + '_valid'] for r_ in Wd}, rho=rho))
    np.savez_compressed(out / f'P1E_{st.name}.npz', case_ids=np.array([f'{cases[i][0]}|{cases[i][1]}' for i in ids]),
                        **{f'W_{k}': v for k, v in Wd.items()})
    st.drop_role('fail')
    st.drop_role('fail_succ')
    return res


def run_order(blk, bank, cc, cases, go, U0, out):
    """CN-R temporal-order check (DESIGN 6): GAL-CN-R (gate) and LSPG-CN-R (reported) at tight tolerance."""
    st = P.Setting(bank, cc['name'] + '_order', cc['Rp'], cc['rule'], cc.get('policy', 'frozen'),
                   roles={'qp': blk['qplus'], 'qpp': blk['qplusplus']}, forms=('LSPG', 'GAL'), nrec=0)
    ci = P.select(cases, [blk['case']])[0]
    n0 = U0[ci][2]
    res = {}
    for form in ('GAL', 'LSPG'):
        F, Wr, ok = {}, {}, {}
        for f_ in blk['factors'] + [blk['anchor'], blk['anchor2']]:
            for role in ('main', 'qp'):
                if role == 'qp' and f_ == blk['anchor2']:
                    continue
                v, f, secs = go(st, role, form, 'CNR', DT0 * f_, 'tight', ci)
                F[(role, f_)], Wr[f'{role}|{f_:g}'], ok[f'{role}|{f_:g}'] = f, np.asarray(v['W']), P.run_valid(v)
        v, f, _ = go(st, 'main', form, 'CNR', DT0 * blk['factors'][-1], 'tighter', ci)
        F[('tighter', blk['factors'][-1])], Wr[f"tighter|{blk['factors'][-1]:g}"] = f, np.asarray(v['W'])
        v, f, _ = go(st, 'qpp', form, 'CNR', DT0 * blk['factors'][-1], 'tight', ci)
        F[('qpp', blk['factors'][-1])], Wr[f"qpp|{blk['factors'][-1]:g}"] = f, np.asarray(v['W'])
        an = blk['anchor']
        e = {f_: P.fdist(F[('main', f_)], F[('main', an)], n0) for f_ in blk['factors']}
        eq = {f_: P.fdist(F[('qp', f_)], F[('qp', an)], n0) for f_ in blk['factors']}
        hs = np.array(blk['factors'], float)
        es = np.array([e[f_] for f_ in blk['factors']])
        slope = float(np.polyfit(np.log(hs), np.log(es), 1)[0])
        pw = [float(np.log(es[i] / es[i + 1]) / np.log(hs[i] / hs[i + 1])) for i in range(len(hs) - 1)]
        fin = blk['factors'][-1]
        diff = e[blk['factors'][-2]] - e[fin]
        checks = dict(asymptotic=abs(pw[0] - pw[-1]) <= .3,
                      anchor=P.fdist(F[('main', an)], F[('main', blk['anchor2'])], n0) <= .1 * e[fin],
                      solver=P.fdist(F[('main', fin)], F[('tighter', fin)], n0) <= .1 * diff,
                      quadrature=qcont(F, fin, an, n0) <= .1 * diff,
                      finer_rule=P.fdist(F[('qp', fin)], F[('qpp', fin)], n0) <= 2.5e-4)
        res[form] = dict(e=e, e_qp=eq, slope=slope, pairwise=pw, checks=checks, quad_contamination=qcont(F, fin, an, n0),
                         finer_rule_dist=P.fdist(F[('qp', fin)], F[('qpp', fin)], n0),
                         anchor_dist=P.fdist(F[('main', an)], F[('main', blk['anchor2'])], n0),
                         solver_dist=P.fdist(F[('main', fin)], F[('tighter', fin)], n0), valid=ok,
                         interpretable=bool(all(checks.values())),
                         passed=bool(all(checks.values()) and 1.8 <= slope <= 2.2) if form == 'GAL' else None)
        np.savez_compressed(out / f'ORDER_{form}.npz', **Wr)
    del st
    gc.collect()
    return res


def qcont(F, h, ha, n0):
    """A1.7: change of the time-error VECTOR under the finer rule: max_k ||(u_q(h)-u_q(ha)) - (u_q+(h)-u_q+(ha))|| / n0."""
    return max(float(np.linalg.norm((F[('main', h)][k] - F[('main', ha)][k]) - (F[('qp', h)][k] - F[('qp', ha)][k]))) / n0
               for k in range(1, 6))


def run_vendor(blk, bank, st, cases, go, U0):
    """A1.5 vendor-vs-generic bridge: qcore.make_linear_query('point') (the historical BE query) vs generic LSPG-BE dt0,
    for each listed rule on the listed cases (combined G1a bars: W 1e-8, w0 1e-12)."""
    vq, _ = Q.make_linear_query('point', st.Rp, bank.L, DT0, st.trust, step_budget=600, gtol=1e-3)
    out = {}
    for rule in blk['rules']:
        role = 'v_' + rule
        if rule == st.rules['main']:
            role = 'main'
        else:
            st.add_role(bank, role, rule)
        d = st.data[(role, 'LSPG')]
        vdata = dict(A=d['A'], lam=d['lam'], G=d['G'], Gq=d['Gq'], Gs=d['Gs'], Psi=d['Psi'])
        rows = []
        for ci in P.select(cases, blk['cases']):
            u0, nu, n0 = U0[ci]
            v, f, _ = go(st, role, 'LSPG', 'BE', DT0, 'prod', ci)
            vv = vq(u0, nu, vdata, st.cold)
            Wv = np.asarray(vv['internal'])[::int(round(.05 / DT0))]
            W = np.asarray(v['W'])
            rows.append(dict(case=f'{cases[ci][0]}|{cases[ci][1]}',
                             W_rel=float(np.max(np.linalg.norm(W - Wv, axis=1)) / np.linalg.norm(Wv[0])),
                             w0_rel=float(np.linalg.norm(np.asarray(v['w0']) - Wv[0]) / np.linalg.norm(Wv[0]))))
        out[rule] = dict(m=st.m[role], rows=rows, passed=bool(all(r_['W_rel'] <= 1e-8 and r_['w0_rel'] <= 1e-12 for r_ in rows)))
        if role != 'main':
            st.drop_role(role)
    return out


def run_controls(blk, bank, st, cc, cases, go, U0, out, cert):
    """Smoke controls on the configuration's bank (DESIGN 6). Predicates are evaluated offline (p1score.py) from the saved
    arrays; the in-job values are recorded for comparison."""
    res = {}
    ids = P.select(cases, blk['cases'])
    sc = blk['ur_scheme']
    # under-resolved rule (A1.1): gauss32 configuration vs its successor gauss64 (actual Q2 predicate), LSPG-BE
    st.add_role(bank, 'ur', blk['under_resolved'])
    st.add_role(bank, 'ur_succ', blk['under_resolved_successor'])
    nl = st.q['LSPG']['nl']
    rows, Wu, Ws = [], [], []
    for ci in ids:
        n0 = U0[ci][2]
        v, f, _ = go(st, 'ur', 'LSPG', sc, DT0, 'prod', ci)
        w, g, _ = go(st, 'ur_succ', 'LSPG', sc, DT0, 'prod', ci)
        Wu.append(np.asarray(v['W']))
        Ws.append(np.asarray(w['W']))
        rho = []
        for kk in range(6):
            c_ = jnp.asarray(Wu[-1][kk])
            x_, y_ = np.asarray(nl(c_, st.data[('ur', 'LSPG')])), np.asarray(nl(c_, st.data[('ur_succ', 'LSPG')]))
            rho.append(P.rho(x_, y_))
        rows.append(dict(case=f'{cases[ci][0]}|{cases[ci][1]}', dist=P.fdist(f, g, n0), rho=rho, valid=P.run_valid(v)))
    res['under_resolved'] = dict(rule=blk['under_resolved'], successor=blk['under_resolved_successor'], rows=rows,
                                 fails_rollout_bar=any(r_['dist'] > 2.5e-4 for r_ in rows),
                                 rho_max=max(max(r_['rho']) for r_ in rows))
    np.savez_compressed(out / 'CTRL_under_resolved.npz', W=np.array(Wu), W_succ=np.array(Ws))
    st.drop_role('ur')
    st.drop_role('ur_succ')
    # premature-exit solver (LM budget 0, A1.6): Q1 stationarity negative control on the pooled dev6 steps
    sc = blk['b0_scheme']
    stb = P.Setting(bank, cc['name'] + '_b0', cc['Rp'], cc['rule'], cc.get('policy', 'frozen'), nrec=NREC, budget=0)
    rows, Sb = [], []
    for ci in P.select(cases, blk['b0_cases']):
        v, f, _ = go(stb, 'main', 'LSPG', sc, DT0, 'prod', ci, record=True)
        g, _ = cert(v['stats']['states'], U0[ci][1], stb.data[('main', 'LSPG')], stb.sched('LSPG', sc, DT0, 'prod'))
        g = np.asarray(g)
        Sb.append(np.asarray(v['stats']['states']))
        rows.append(dict(case=f'{cases[ci][0]}|{cases[ci][1]}', nonstationary=int(np.sum(~(g <= 1e-3))), steps=len(g),
                         gamma=g.tolist(), valid=P.run_valid(v)))
    res['budget0'] = dict(rows=rows, pooled_fraction=sum(r_['nonstationary'] for r_ in rows) / sum(r_['steps'] for r_ in rows))
    np.savez_compressed(out / 'CTRL_budget0.npz', states=np.array(Sb), case_ids=np.array([r_['case'] for r_ in rows]),
                        gamma=np.array([r_['gamma'] for r_ in rows]), nu=np.array([U0[ci][1] for ci in P.select(cases, blk['b0_cases'])]),
                        A=np.asarray(stb.base['A']), lam=np.asarray(stb.base['lam']), kx=np.asarray(stb.kx), ky=np.asarray(stb.ky),
                        scheme=sc, rule=cc['rule'], L=bank.L)
    del stb
    # sign-flipped advection Jacobian (probe, reported only)
    stj = P.Setting(bank, cc['name'] + '_jflip', cc['Rp'], cc['rule'], cc.get('policy', 'frozen'), nrec=NREC, mutation='jflip')
    rows = []
    for ci in ids:
        v, f, _ = go(stj, 'main', 'LSPG', sc, DT0, 'prod', ci, record=True)
        g, _ = cert(v['stats']['states'], U0[ci][1], stj.data[('main', 'LSPG')], stj.sched('LSPG', sc, DT0, 'prod'))
        g = np.asarray(g)
        rows.append(dict(case=f'{cases[ci][0]}|{cases[ci][1]}', nonstationary=int(np.sum(~(g <= 1e-3))), steps=len(g),
                         valid=P.run_valid(v), it_sum=int(v['stats']['it_sum'])))
    res['jflip_probe'] = rows
    del stj
    # wrong-mesh table: a table built on another mesh (256^2 in the smoke) must be refused by this bank
    small = P.Bank(256 if bank.L != 256 else 128, [0, cc['Rp']])
    sts = P.Setting(small, 'wrongmesh', cc['Rp'], cc['rule'], 'frozen')
    raised = []
    for fn in (lambda: sts.check('main', bank),
               lambda: P.run_query(bank, sts, 'main', 'LSPG', sc, DT0, 'prod', U0[ids[0]][0], U0[ids[0]][1])):
        try:
            fn()
            raised.append(False)
        except ValueError as e_:
            raised.append(str(e_))
    res['wrong_mesh'] = dict(raised=raised, passed=all(bool(x) for x in raised))
    del sts, small
    gc.collect()
    return res


if __name__ == '__main__':
    main()
