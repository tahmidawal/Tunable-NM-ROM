"""imp-multimesh smoke S1 (DESIGN A1.6, A2.6, A2.7): every must-fail control on real data at 1024^2, plus the
4096^2 memory / time pilot. Control 1 (CPU preflight) and control 9 (p1acc replication fixture) are separate sbatch
steps; control 8 (decision logic) runs offline on these outputs (cost.py --selftest).

    python mmsmoke.py <config.json> <outdir>
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import mmcore as M
import mmbuild as B

H, BD, RM, PL = M.H, M.BD, M.RM, M.PL
D = 2
log = B.log


def ctl(rec, name, passed, **kw):
    rec.d.setdefault('controls', {})[name] = dict(passed=bool(passed), **kw)
    log(f'CONTROL {name}: {"PASS" if passed else "FAIL"} {kw}')
    rec.save()


def main():
    cfg = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    B.check_env(cfg)
    rec = B.Rec(out / 'smoke.json', dict(config=cfg, gpu=jax.devices()[0].device_kind, job=os.environ.get('SLURM_JOB_ID'),
                                         commit=os.environ.get('SOURCE_COMMIT'), complete=False))
    s, Lf, Lc = 1024, 4096, 256
    # ---------------------------------------------------------------- reduced source at 1024^2 (A2.7) -----
    src = B.build_source(rec, 'src1024', s, '12', [128, 384], [4, 8], cfg['nnls_seconds'], out, ntraj=cfg['ntraj'],
                         ecsw_ranks=[128])
    Phi_s, C_s, rules_s, trust_s, states_s = src['Phi'], src['C'], src['rules'], src['trust'], src['states']
    Phi_h = np.asarray(Phi_s)
    # ---------------------------------------------------------------- control 2: restriction to 256 -----
    Pc, Rr, ic = M.resample_basis(Phi_h, s, Lc)
    recon = None
    Praw = M.resample(Phi_h, s, Lc)
    recon = float(jnp.linalg.norm(Pc @ jnp.asarray(Rr) - Praw) / jnp.linalg.norm(Praw))
    bad = Pc.at[:, 5].multiply(1 + 1e-6)
    dev_bad = M.orth_dev(bad)
    cvec = np.random.default_rng(0).standard_normal(128)
    u_s = Phi_h[:, :128] @ cvec
    u_c = np.asarray(M.resample(u_s[:, None], s, Lc))[:, 0] * (Lc / s)
    cc = np.asarray(Pc[:, :128]).T @ u_c
    ct = M.transform_codes(cvec[None], s, Lc, 128, Rr)[0]
    code_err = float(np.max(np.abs(cc - ct)) / np.max(np.abs(cc)))
    ctl(rec, '2c_restriction', ic['orth'] <= 1e-10 and recon <= 1e-12 and dev_bad > 1e-10 and code_err <= 1e-10,
        orth=ic['orth'], orth_raw=ic['orth_raw'], qr_recon=recon, scaled_column_orth=dev_bad, code_err=code_err,
        rank_ratio=ic['rank_ratio'])
    del Pc, Praw, bad
    gc.collect()
    # ---------------------------------------------------------------- control 2: prolongation to 4096 (pilot) -----
    t0 = time.perf_counter()
    Pf, _, pinfo = M.resample_basis(Phi_h, s, Lf)
    t_res = time.perf_counter() - t0
    cons = M.node_consistency(Phi_h, s, Pf, Lf)
    Ct = M.transform_codes(C_s, s, Lf, 384)
    tr_f = BD.trust_radii(Ct, [128, 384], D)
    tr_ratio = max(abs(tr_f[r] / (trust_s[r] * Lf / s) - 1) for r in (128, 384))
    rec.d['pilot'] = dict(resample_seconds=t_res, mem_after_resample=B.dev_mem(), pinfo=pinfo)
    mut = {}
    for mname in ('shift', 'transpose'):
        Pm = M.resample(Phi_h[:, :16], s, Lf, mutate=mname)
        mut[mname] = M.node_consistency(Phi_h[:, :16], s, Pm, Lf)
        del Pm
    ctl(rec, '2ab_prolongation', pinfo['orth'] <= 1e-10 and cons <= 1e-12 and tr_ratio <= 1e-12
        and all(v > 1e-6 for v in mut.values()), orth=pinfo['orth'], node_consistency=cons, trust_ratio_dev=tr_ratio,
        mutated=mut)
    # ---------------------------------------------------------------- control 3: rule mapping -----
    nodes, w = rules_s[(128, 'k4')]
    pm = M.map_nodes(nodes, s, Lf)
    i, j = np.unravel_index(nodes, (s - 1, s - 1))
    k, l = np.unravel_index(pm, (Lf - 1, Lf - 1))
    coord_ok = bool(np.all((i + 1) * Lf == (k + 1) * s) and np.all((j + 1) * Lf == (l + 1) * s))
    a_ = (Lf / s) * np.asarray(Pf[jnp.asarray(pm), :128])
    b_ = Phi_h[nodes, :128]
    val = float(np.max(np.abs(a_ - b_)) / np.max(np.abs(b_)))
    pu = M.map_nodes(nodes, s, Lf, mutate=True)
    k2, l2 = np.unravel_index(pu, (Lf - 1, Lf - 1))
    coord_bad = bool(np.any((i + 1) * Lf != (k2 + 1) * s) or np.any((j + 1) * Lf != (l2 + 1) * s))
    val_bad = float(np.max(np.abs((Lf / s) * np.asarray(Pf[jnp.asarray(pu), :128]) - b_)) / np.max(np.abs(b_)))
    ctl(rec, '3_mapping', coord_ok and val <= 1e-12 and val_bad > 1e-6 and coord_bad, coord_ok=coord_ok, value_err=val,
        unmapped_value_err=val_bad, unmapped_coords_differ=coord_bad)
    # ---------------------------------------------------------------- 4096^2 pilot: states, blocked contrib, NNLS, query -----
    old = BD.N_ECSW_TRAJ
    BD.N_ECSW_TRAJ = cfg['pilot_traj']
    t0 = time.perf_counter()
    st_f, _ = BD.ecsw_states(D, Lf, H.Family(D, 'train'), None)
    BD.N_ECSW_TRAJ = old
    rec.d['pilot'].update(states=len(st_f), states_seconds=time.perf_counter() - t0, mem_states=B.dev_mem(),
                          host_rss_states=PL.host_rss())
    rec.save()
    st_t = st_f * int(cfg['pilot_tile'])                 # A3.6: production-shaped (120-state) contribution matrix
    t0 = time.perf_counter()
    G, b, cand, cinfo = M.ecsw_contrib_blocked(st_t, Pf, 384, Lf, log=log)
    rec.d['pilot'].update(real_states=len(st_f), tiled_states=len(st_t), G_shape=list(G.shape), G_bytes=int(G.size * 8),
                          G_shape_ok=bool(G.shape[0] == 120 * 384 and G.shape[1] == 32768), host_rss=PL.host_rss())
    rec.d['pilot'].update(contrib_seconds=time.perf_counter() - t0, contrib=cinfo, mem_contrib=B.dev_mem())
    rec.save()
    t0 = time.perf_counter()
    rr, summ = BD.ecsw_weights(G, b, cand, 384, (4, 8), max_seconds=cfg['pilot_nnls_seconds'])
    summ.pop('checkpoints_all', None)
    rec.d['pilot'].update(nnls_seconds=time.perf_counter() - t0, nnls=summ, mem_nnls=B.dev_mem(), host_rss_nnls=PL.host_rss(),
                          supports={k: (None if v is None else len(v['nodes'])) for k, v in rr.items()})
    del G, b
    rec.save()
    fams = B.families()
    refs = B.refs_for(fams)
    cellf = M.Cell(Lf, Pf, {128: tr_f[128], 384: tr_f[384]}, log=log)
    n128, w128 = rules_s[(128, 'k4')]
    data, _ = cellf.ecsw_data(128, M.map_nodes(n128, s, Lf), w128)       # the reused (CPU-1024) r=128 rule
    u0 = fams['dev6'].ic(Lf, 0)
    q = RM.make_query('ecsw', D, Lf, 'common', trust=tr_f[128])
    sch = RM.schedule(D, 'CNR', 0.005, 'prod')
    cinfo = M.aot_compile(q, jnp.asarray(u0), jnp.float64(fams['dev6'].nu(0)), data, sch)
    t_comp = cinfo
    r1 = cellf.run('ecsw', 128, data, 'common', ('CNR', 0.005), 'prod', u0, fams['dev6'].nu(0))
    r2 = cellf.run('ecsw', 128, data, 'common', ('CNR', 0.005), 'prod', u0, fams['dev6'].nu(0))
    e = refs['dev6'].score(r2['fields_lat'], 0)['e']
    rec.d['pilot'].update(query_seconds=[r1['seconds'], r2['seconds']], aot_compile_seconds=t_comp,
                          e_dev6_0=e, mem_query=B.dev_mem(), cert=r2['cert'])
    rec.save()
    del cellf, data, Pf, q
    gc.collect()
    rec.d['pilot']['mem_after_free'] = B.dev_mem()
    rec.save()
    # PM snapshot-cost estimate, one trajectory (time only)
    B.pm_estimate(rec, Lf, 2, 20261021)
    # ---------------------------------------------------------------- control 5: blocked == dense at 1024 -----
    t0 = time.perf_counter()
    G0, b0, c0, i0 = BD.ecsw_contrib(states_s, Phi_s[:, :128], D, s)
    G1, b1, c1, i1 = M.ecsw_contrib_blocked(states_s, Phi_s, 128, s, log=None)
    same_c = bool(np.array_equal(c0, c1))
    eg = float(jnp.linalg.norm(G0 - G1) / jnp.linalg.norm(G0)) if G0.shape == G1.shape else float('inf')
    eb = float(jnp.linalg.norm(b0 - b1) / jnp.linalg.norm(b0)) if b0.shape == b1.shape else float('inf')
    G2, b2, _, _ = M.ecsw_contrib_blocked(states_s, Phi_s, 128, s, cand_nodes=c0, mutation='drop_adv_jac', log=None)
    em = float(jnp.linalg.norm(G0 - G2) / jnp.linalg.norm(G0))
    kd = np.asarray(i0['block_norms']) >= max(1e-8 * float(np.median(i0['block_norms'])),
                                               1e-14 * float(np.max(i0['block_norms'])), 1e-300)
    same_keep = bool(np.array_equal(kd, np.asarray(i1['keep'])))
    bn = float(np.max(np.abs(np.asarray(i0['block_norms']) - np.asarray(i1['block_norms'])) /
                     np.max(np.asarray(i0['block_norms']))))
    ctl(rec, '5_blocked', same_c and same_keep and eg <= 1e-10 and eb <= 1e-10 and bn <= 1e-10 and em > 1e-6,
        same_candidates=same_c, same_keep=same_keep, G_rel=eg, b_rel=eb, block_norm_rel=bn, mutation_rel=em,
        seconds=time.perf_counter() - t0)
    del G0, G1, G2, b0, b1, b2
    gc.collect()
    # ---------------------------------------------------------------- control 4: Q1 must fail with LM budget 0 -----
    cell = M.Cell(s, Phi_s, trust_s, log=log)
    nodes, w = rules_s[(128, 'k4')]
    data, _ = cell.ecsw_data(128, nodes, w)
    dev6 = B.cases_all('dev6')
    good, _ = M.run_config(cell, 128, data, ('CNR', 0.005), dev6, fams, refs, tag='fixture')
    with M.budget_override(0):
        cell0 = M.Cell(s, Phi_s, trust_s, log=log)
        bad, _ = M.run_config(cell0, 128, data, ('CNR', 0.005), dev6, fams, refs, tag='budget0')
    fallback = None
    if bad['q1']['q1']:                                  # A3.5: registered fallback mutation
        with M.unsolved_override():
            cell1 = M.Cell(s, Phi_s, trust_s, log=log)
            fallback, _ = M.run_config(cell1, 128, data, ('CNR', 0.005), dev6, fams, refs, tag='unsolved')
    fired = bool(good['q1']['q1']) and ((not bad['q1']['q1']) or (fallback is not None and not fallback['q1']['q1']))
    ctl(rec, '4_q1_mutation', fired, fixture_q1=good['q1'], budget0_q1=bad['q1'],
        fallback_q1=None if fallback is None else fallback['q1'], fixture_summary=M.summarise(good['rows']))
    rec.d['fixture_records'] = good
    # ---------------------------------------------------------------- control 6: case identity -----
    ok_desc = bool(np.array_equal(fams['dev6'].D[0], fams['dev6'].D[0]))
    tripped = False
    try:
        M.assert_case_identity(fams, refs, ('dev6', 1), fams['dev6'].D[0])
    except AssertionError:
        tripped = True
    M.assert_case_identity(fams, refs, ('dev6', 0), fams['dev6'].D[0])
    rec0 = cell.run('ecsw', 128, data, 'common', ('CNR', 0.005), 'prod', fams['dev6'].ic(s, 0), fams['dev6'].nu(0))
    e0_true = M.e0_of(refs['dev6'], rec0['fields_lat'], 0)
    e0_wrong = M.e0_of(refs['dev6'], rec0['fields_lat'], 1)
    ctl(rec, '6_case_identity', ok_desc and tripped, tripped=tripped, e0_true=e0_true, e0_wrong=e0_wrong)
    # ---------------------------------------------------------------- control 7: timing injection -----
    import mmtime as TM
    q7 = cell.query('ecsw', 128, 'common')
    xs = [(fams['dev6'].ic(s, j), fams['dev6'].nu(j)) for j in range(6)]
    sch = RM.schedule(D, 'CNR', 0.005, 'prod')
    f7 = lambda u, nu: q7(u, jnp.float64(nu), data, sch)
    jax.block_until_ready(f7(*xs[0]))
    s0 = float(np.median(TM.time_calls(f7, [xs[0]] * 3)))          # sentinel before
    a1 = TM.time_calls(lambda u, nu: q7(u, jnp.float64(nu), data, sch), xs)
    med = float(np.median(a1))
    a2 = TM.time_calls(lambda u, nu: q7(u, jnp.float64(nu), data, sch), xs, inject=0.25 * med)
    a3 = TM.time_calls(lambda u, nu: q7(u, jnp.float64(nu), data, sch), xs)
    s1 = float(np.median(TM.time_calls(f7, [xs[0]] * 3)))          # sentinel after
    ok_clean = TM.in_band(np.median(a3) / med)
    sent_ok = TM.in_band(s1 / s0)
    ctl(rec, '7_timing', ok_clean and sent_ok and not TM.in_band(np.median(a2) / med), a1=a1, a2_injected=a2, a3=a3,
        sentinel=[s0, s1], sentinel_ok=sent_ok)
    del cell, data
    gc.collect()
    # ---------------------------------------------------------------- ours: 4096^2 bank table pilot (A1.7) -----
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'jcp-p1-scaling'))
    import p1common as P
    t0 = time.perf_counter()
    bank = P.Bank(Lf, [0, 128, 256, 384, 512])
    bank.ops('frozen')
    st = P.Setting(bank, 'R384', 384, 'gauss128', 'frozen', roles={}, nrec=51)
    ph = P.case_list({})[0][2]
    u0 = jnp.asarray(P.Q.e.initial(Lf, ph))
    v = P.run_query(bank, st, 'main', 'LSPG', 'CNR', P.DT0, 'prod', u0, float(ph[4]))
    jax.block_until_ready(v['W'])
    rec.d['ours_pilot'] = dict(seconds=time.perf_counter() - t0, bank_bytes=bank.bytes, mem=B.dev_mem(),
                               host_rss=PL.host_rss())
    del bank, st, v, u0
    gc.collect()
    rec.d['complete'] = True
    rec.save()
    log('MMSMOKE COMPLETE')


if __name__ == '__main__':
    main()
