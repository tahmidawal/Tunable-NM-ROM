"""quadrature-burgers3d panel job (DESIGN.md): one mesh, one allocation.

Phases (every phase writes into output/result.json as it completes):
  1. tables  : the lean per-mesh tables of burgers3d-retry (vendor tables.build_tables), its table gates;
  2. offmesh : B, D, P for every rule of the config; evaluator gates G1-G4 (DESIGN section 6);
  3. rho     : the tensor solve on the certification draws; rho of every rule, the tensor and the dense stencil
               against the continuum target (Gauss 80^3) and the mesh target (sign-upwind on every node);
  4. cohort  : same-grid references, every arm once per case (errors, iterations, exits, coefficients), the FOM grid;
  5. timing  : A1 (ROM) - B (FOM) - A2 (ROM) on the first `timing_cases` cases, gates as burgers3d-retry R1-5;
               a microbenchmark of one advection-Jacobian evaluation per rule;
  6. refined : errors against the refined reference (513 nodes, dt/2) on the 65-node lattice x = k/64, read from the
               reference job's output (waits for it).
Output: output/result.json, output/fields/*.npz (restricted fields, coefficients, rho states).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import offmesh as OM  # noqa: E402
from offmesh import C, TB  # noqa: E402

block = jax.block_until_ready


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def lattice65_index(n):
    """Interior indices of the nodes x = k/64, k = 1..63 per axis (common to every mesh n with 64 | n - 1)."""
    s = (n - 1) // 64
    assert s * 64 == n - 1, n
    k = np.arange(1, 64) * s
    I, J, K = np.meshgrid(k, k, k, indexing='ij')
    ni = n - 2
    return (((I - 1) * ni + (J - 1)) * ni + (K - 1)).ravel()


def lattice65_coords():
    ax = np.arange(1, 64) / 64.0
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing='ij')
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)


def load_model(model_dir, R_truncate=None):
    b = pickle.loads((Path(model_dir) / 'bank.pkl').read_bytes())
    T = np.asarray(b['rotation'])
    if R_truncate:
        T = T[:, :R_truncate]
    return dict(bank=jax.tree_util.tree_map(jnp.asarray, b['params']), T=T, spread=b['coefficient_rms_spread'],
                sha=sha_file(Path(model_dir) / 'bank.pkl'))


def load_rules(path, names):
    z = np.load(path)
    return {nm: (np.asarray(z[f'{nm}_X']), np.asarray(z[f'{nm}_w'])) for nm in names}


def rel_max(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    (out / 'fields').mkdir(parents=True, exist_ok=True)
    local = bool(cfg.get('local_smoke'))
    if not local:
        assert jax.default_backend() == 'gpu', jax.default_backend()
    assert jax.config.jax_enable_x64 and os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    begin = time.perf_counter()
    el = lambda: round(time.perf_counter() - begin, 1)
    log = lambda s: print(f'[{el()}s] {s}', flush=True)
    try:
        smi = subprocess.check_output(['nvidia-smi', '-L'], text=True).strip().splitlines()
    except Exception:  # noqa: BLE001
        smi = []
    n = int(cfg['mesh'])
    root = HERE.parent.parent
    model = load_model(root / cfg['model'], cfg.get('R_truncate'))
    want = cfg.get('expected_model_sha256')
    assert want is None or want == model['sha'], (model['sha'], want)
    rules_path = root / cfg['rules_npz']
    rules_sha = sha_file(rules_path)
    assert cfg.get('expected_rules_sha256') in (None, rules_sha), rules_sha
    rule_names = list(cfg['rules'])
    rules = load_rules(rules_path, rule_names + [cfg['continuum_target'], cfg['continuum_check']])
    rep = dict(config=cfg, mesh=n, commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
               backend=jax.default_backend(), gpu=jax.devices()[0].device_kind, nvidia_smi=smi, x64=True,
               precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'], jax_version=jax.__version__,
               model_sha256=model['sha'], rules_sha256=rules_sha, gates={}, rho={}, arms={}, fom={}, timing={},
               microbench={}, sizes={}, complete=False,
               timing_contract=('GPU query: supplied dense initial interior field resident on the GPU -> six dense GPU '
                                'output fields, block_until_ready on the whole output; compilation excluded'))
    save = lambda: C.dump(out / 'result.json', C.clean(rep))
    save()

    # ================================================================ 1. tables
    R = int(model['T'].shape[1])
    Rps = sorted(cfg['Rps'], reverse=True)
    ladder = sorted(set(Rps) | {R})
    order, lam_all = C.mode_order(n)
    M_of = lambda r: C.complete_M(n, 4 * r, order, lam_all)
    t0 = time.perf_counter()
    tb = TB.build_tables(n, model['bank'], model['T'], ladder, M_of, log=log)
    rep['table_seconds'] = time.perf_counter() - t0
    rep['gates']['gram_condition'] = tb['gram_cond'] ** 2
    assert rep['gates']['gram_condition'] <= 1e8, rep['gates']
    kx = tb['kxyz']
    rng = np.random.default_rng(20261001)
    c = rng.normal(size=R) / np.sqrt(R)
    u = TB.decode(jnp.asarray(c)[None], tb)[0]
    idx = tuple(jnp.asarray(kx[:, j]) for j in range(3))
    direct = np.asarray(C.phiT(C.backward_adv(u, n)[None], n, idx)[0])
    via = np.asarray(0.5 * jnp.einsum('mij,i,j->m', tb['Tsym'], jnp.asarray(c), jnp.asarray(c)))
    rep['gates']['tensor_vs_direct'] = float(np.linalg.norm(via - direct) / np.linalg.norm(direct))
    assert rep['gates']['tensor_vs_direct'] < 1e-10, rep['gates']
    del u
    log(f"tables done; gates {rep['gates']}")
    save()

    # ================================================================ 2. off-mesh tables + evaluator gates
    om_full = {}
    for nm in rule_names:
        X, w = rules[nm]
        om_full[nm] = OM.offmesh_tables(n, model['bank'], model['T'], X, w, kx)
        rep['sizes'][nm] = dict(m=int(len(w)), weight_sum=float(np.sum(w)))
    log('off-mesh tables built')
    # G1: analytic directional derivative vs centred finite differences
    Xg = rules[rule_names[0]][0][:32]
    Bg, Dg = OM.point_blocks(model['bank'], model['T'], Xg)
    h = 1e-5
    fp, _ = OM.point_blocks(model['bank'], model['T'], Xg + h)
    fm, _ = OM.point_blocks(model['bank'], model['T'], Xg - h)
    rep['gates']['G1_derivative_vs_fd'] = rel_max((fp - fm) / (2 * h), Dg)
    # G2: the off-mesh assembly fed with the mesh nodes, weights (n-1)^-3 and the backward difference D^- reproduces
    #     the tensor contraction (validates P scaling, test ordering, the Jacobian formula), on a column/test subset
    Rs, Ms = min(32, R), 64
    Xn = C.interior_coords(n)
    assert tb['GTb'][0].shape[0] >= Rs
    rows = jnp.array(tb['GTb'][0][:Rs])                                                         # (Rs, N)
    Drows = C.dminus(rows, n)
    cs = jnp.asarray(rng.normal(size=Rs) / np.sqrt(Rs))
    Ju_mesh = 0.
    for s in range(0, Xn.shape[0], 1 << 20):
        Pq = OM.test_block(n, Xn[s:s + (1 << 20)], np.full(min(1 << 20, Xn.shape[0] - s), float(n - 1) ** -3),
                           kx[:Ms])
        Ju_mesh = Ju_mesh + OM.contract_offmesh(dict(B=rows[:, s:s + (1 << 20)].T, D=Drows[:, s:s + (1 << 20)].T,
                                                     P=Pq), cs)
    Ju_t = OM.contract_tensor(dict(Ts=tb['Tsym'][:Ms, :Rs, :Rs]), cs)
    rep['gates']['G2_meshnodes_offmesh_vs_tensor'] = rel_max(Ju_mesh, Ju_t)
    del rows, Drows, Xn, Pq, Ju_mesh
    # G3: Jacobian formula vs jax.jacfwd of the off-mesh advection, and adv = 0.5 Ju c (deployed slices)
    Rp0 = Rps[0]
    M0 = M_of(Rp0)
    d3 = dict(B=om_full[rule_names[0]]['B'][:, :Rp0], D=om_full[rule_names[0]]['D'][:, :Rp0],
              P=om_full[rule_names[0]]['P'][:, :M0])
    c3 = jnp.asarray(rng.normal(size=Rp0) / np.sqrt(Rp0))
    J_ad = jax.jacfwd(lambda cc: OM.adv_offmesh_batch(d3, cc[None])[0])(c3)
    J_f = OM.contract_offmesh(d3, c3)
    rep['gates']['G3_jacobian_vs_jacfwd'] = rel_max(J_f, J_ad)
    rep['gates']['G3_adv_half_Jc'] = rel_max(0.5 * J_f @ c3, OM.adv_offmesh_batch(d3, c3[None])[0])
    del d3, J_ad
    log(f"evaluator gates {rep['gates']}")
    assert rep['gates']['G1_derivative_vs_fd'] < 1e-6, rep['gates']
    assert rep['gates']['G2_meshnodes_offmesh_vs_tensor'] < 1e-12, rep['gates']
    assert rep['gates']['G3_jacobian_vs_jacfwd'] < 1e-12 and rep['gates']['G3_adv_half_Jc'] < 1e-12, rep['gates']
    save()

    # ================================================================ arms
    dense_cases = cfg.get('dense_cases', 0)
    arms = {}
    for Rp in Rps:
        M = int(M_of(Rp))
        trust = float(cfg['trust_fraction'] * model['spread'][str(Rp)])
        base = TB.arm_data(tb, Rp, M)                    # GTb prefix, L, A, lam, Ts (tensor slice)
        arms[f'tensor_R{Rp}'] = dict(Rp=Rp, M=M, rule='tensor', family='tensor', trust=trust, data=base,
                                     q=OM.make_fsc_rule(n, Rp, M, OM.contract_tensor, dt=cfg['dt'], gtol=cfg['gtol'],
                                                        trust=trust))
        lean = {k: v for k, v in base.items() if k != 'Ts'}
        for nm in rule_names:
            d = dict(lean, B=om_full[nm]['B'][:, :Rp], D=om_full[nm]['D'][:, :Rp], P=om_full[nm]['P'][:, :M])
            d = jax.tree_util.tree_map(lambda x: block(jnp.asarray(x)), d)
            arms[f'{nm}_R{Rp}'] = dict(Rp=Rp, M=M, rule=nm, family='offmesh', trust=trust, data=d,
                                       q=OM.make_fsc_rule(n, Rp, M, OM.contract_offmesh, dt=cfg['dt'],
                                                          gtol=cfg['gtol'], trust=trust))
        if dense_cases:
            arms[f'dense_R{Rp}'] = dict(Rp=Rp, M=M, rule='dense', family='dense', trust=trust, data=lean,
                                        q=OM.make_fsc_rule(n, Rp, M, OM.make_contract_dense(n, M, kx), dt=cfg['dt'],
                                                           gtol=cfg['gtol'], trust=trust, linear_jac=False))
    for nm, s in arms.items():
        rep['arms'][nm] = dict(spec={k: v for k, v in s.items() if k not in ('data', 'q')})
    del om_full
    tsym_bytes = {Rp: int(M_of(Rp)) * Rp * Rp * 8 for Rp in Rps}
    for nm, s in arms.items():
        if s['family'] == 'tensor':
            rep['sizes'][nm] = dict(bytes=tsym_bytes[s['Rp']], flops_jacobian=2 * s['M'] * s['Rp'] ** 2)
        elif s['family'] == 'offmesh':
            m = s['data']['B'].shape[0]
            rep['sizes'][nm] = dict(m=int(m), bytes=8 * m * (2 * s['Rp'] + s['M']),
                                    flops_jacobian=2 * m * s['Rp'] * (2 + s['M']) + 4 * m * s['Rp'],
                                    flops_residual=2 * m * (2 * s['Rp'] + s['M']))
    # G4: solver equivalence, tensor rule through make_fsc_rule vs the vendor make_fsc (one case)
    tab_g = C.table(cfg['cert_draws'][0][0], 1)
    u0g = jnp.asarray(C.initial_interior(n, tab_g, 0))
    Rg = Rps[-1]
    sg = arms[f'tensor_R{Rg}']
    qv, _ = TB.make_fsc(n, Rg, sg['M'], dt=cfg['dt'], gtol=cfg['gtol'], trust=sg['trust'])
    o_v = qv(u0g, float(tab_g['nu'][0]), sg['data'], {})
    o_r = sg['q'](u0g, float(tab_g['nu'][0]), sg['data'], {})
    rep['gates']['G4_solver_vs_vendor_fields'] = rel_max(o_r[0], o_v[0])
    rep['gates']['G4_solver_vs_vendor_coefs'] = rel_max(o_r[1], o_v[1])
    assert rep['gates']['G4_solver_vs_vendor_fields'] <= 1e-12, rep['gates']
    del o_v, o_r, qv
    log(f"G4 {rep['gates']['G4_solver_vs_vendor_fields']:.2e}")
    save()

    def run(nm, u0, nu):
        s = arms[nm]
        return s['q'](u0, nu, s['data'], {})

    # ================================================================ 3. rho on reached states (certification draws)
    Xc, wc = rules[cfg['continuum_target']]
    Xk, wk = rules[cfg['continuum_check']]
    for Rp in Rps:
        st = arms[f'tensor_R{Rp}']
        M = st['M']
        states, kidx = [], []
        for seed, count in cfg['cert_draws']:
            tab = C.table(seed, count)
            for j in range(count):
                o = run(f'tensor_R{Rp}', jnp.asarray(C.initial_interior(n, tab, j)), float(tab['nu'][j]))
                ws = np.asarray(o[1])
                states.append(ws)
                kidx.append(np.arange(ws.shape[0]))
        Cs = np.concatenate(states, 0)
        kk = np.concatenate(kidx)
        mesh_t = OM.make_mesh_target(n, M, kx)
        tgt_mesh, umin = map(np.asarray, mesh_t(jnp.asarray(Cs), st['data']))
        tgt_cont = np.asarray(OM.continuum_adv(n, model['bank'], model['T'], Xc, wc, kx[:M], Cs))
        chk_cont = np.asarray(OM.continuum_adv(n, model['bank'], model['T'], Xk, wk, kx[:M], Cs))
        ten = np.asarray(jax.jit(lambda d, X: jax.lax.map(lambda cc: 0.5 * OM.contract_tensor(d, cc) @ cc, X))(
            st['data'], jnp.asarray(Cs)))

        def rho(v, tgt):
            return np.linalg.norm(v - tgt, axis=1) / np.maximum(np.linalg.norm(tgt, axis=1), 1e-300)

        def summ(r):
            ev, k0 = r[kk >= 1], r[kk == 0]
            return dict(worst=float(ev.max()), median=float(np.median(ev)), p90=float(np.quantile(ev, 0.9)),
                        worst_k0=float(k0.max()), median_k0=float(np.median(k0)))
        res = dict(states=int(len(Cs)), states_evolved=int((kk >= 1).sum()), M=M, umin=float(umin.min()),
                   continuum_check=dict(vs_target=summ(rho(chk_cont, tgt_cont))),
                   rules={})
        res['rules']['tensor'] = dict(cont=summ(rho(ten, tgt_cont)), mesh=summ(rho(ten, tgt_mesh)))
        res['rules']['dense_upwind'] = dict(cont=summ(rho(tgt_mesh, tgt_cont)), mesh=summ(np.zeros(len(Cs))))
        for nm in rule_names:
            v = np.asarray(OM.adv_offmesh_batch(arms[f'{nm}_R{Rp}']['data'], jnp.asarray(Cs)))
            rc, rm = rho(v, tgt_cont), rho(v, tgt_mesh)
            res['rules'][nm] = dict(cont=summ(rc), mesh=summ(rm))
            if nm == cfg.get('audit_rule'):
                np.savez(out / 'fields' / f'rho_R{Rp}_{nm}.npz', Cs=Cs[:: max(1, len(Cs) // 32)],
                         k=kk[:: max(1, len(Cs) // 32)], v=v[:: max(1, len(Cs) // 32)],
                         tgt_cont=tgt_cont[:: max(1, len(Cs) // 32)], tgt_mesh=tgt_mesh[:: max(1, len(Cs) // 32)])
        rep['rho'][str(Rp)] = res
        log(f"RHO R{Rp}: check {res['continuum_check']['vs_target']['worst']:.2e}; " +
            '; '.join(f"{k} c{v['cont']['worst']:.2e} m{v['mesh']['worst']:.2e}" for k, v in res['rules'].items()))
        save()
        del tgt_mesh, tgt_cont, chk_cont, Cs

    if cfg.get('stop_after_rho'):
        rep['complete'] = True
        rep['seconds'] = el()
        save()
        print('PANEL COMPLETE', flush=True)
        return

    # ================================================================ 4. cohort
    tab = C.table(cfg['cohort_seed'], cfg['cohort_count'])
    rep['cohort'] = dict(seed=cfg['cohort_seed'], count=cfg['cohort_count'], table_sha256=tab['sha256'])
    assert cfg.get('expected_cohort_sha256') in (None, tab['sha256']), 'cohort differs'
    cases = list(range(cfg['cohort_count']))
    U0 = [block(jnp.asarray(C.initial_interior(n, tab, j))) for j in cases]
    NU = [float(tab['nu'][j]) for j in cases]
    ref_fom = C.make_fom(n, C.DT, cfg['ref_ntol'], cfg['ref_ltol'])
    REF, refrec = [], []
    for j in cases:
        f, it, rn = ref_fom(U0[j], NU[j])
        REF.append(np.asarray(f))
        refrec.append(dict(max_newton=int(jnp.max(it)), max_rel_residual=float(jnp.max(rn)),
                           finite=bool(np.isfinite(REF[-1]).all())))
        del f
    rep['reference'] = dict(ntol=cfg['ref_ntol'], ltol=cfg['ref_ltol'], cases=refrec)
    rep['gates']['reference_residual'] = max(r['max_rel_residual'] for r in refrec)
    assert rep['gates']['reference_residual'] < 1e-9 and all(r['finite'] for r in refrec), refrec
    log(f'same-grid references done, worst residual {rep["gates"]["reference_residual"]:.2e}')
    has65 = (n - 1) % 64 == 0
    ridx_np = C.restrict_index(n, 16)
    ridx = jnp.asarray(ridx_np)
    i65 = jnp.asarray(lattice65_index(n)) if has65 else None
    errf = jax.jit(lambda f, r: jnp.linalg.norm(f - r, axis=1) / jnp.linalg.norm(r[0]))
    np.savez(out / 'fields' / 'reference_restricted.npz', **{f'c{j}': REF[j][:, ridx_np] for j in cases})
    ref0 = [float(np.linalg.norm(REF[j][0])) for j in cases]
    rep['reference']['norm0'] = ref0
    save()

    COEF = {}
    quick16 = {}
    for nm, s in arms.items():
        if s['family'] == 'dense' and dense_cases != 'all':
            run_cases = cases[:int(dense_cases)]
        else:
            run_cases = cases
        recs = []
        t_c = time.perf_counter()
        for j in run_cases:
            o = run(nm, U0[j], NU[j])
            block(o)
            if j == run_cases[0]:
                compile_s = time.perf_counter() - t_c
            t_q = time.perf_counter()
            r = jnp.asarray(REF[j])
            e = np.asarray(errf(o[0], r))
            del r
            keep = int(round(0.05 / cfg['dt']))
            Cout = np.asarray(o[1])[::keep]                                       # (6, R') coefficients at outputs
            COEF[(nm, j)] = Cout
            quick16[(nm, j)] = np.asarray(o[0][:, ridx])
            recs.append(dict(case=j, err_same_grid=e.tolist(), worst_same_grid=float(e[1:].max()),
                             iterations=np.asarray(o[2]).tolist(), reasons=np.asarray(o[3]).tolist(),
                             gradients=np.asarray(o[4]).tolist(), rejected=np.asarray(o[6]).tolist(),
                             finite=bool(np.isfinite(np.asarray(o[0])).all() and np.isfinite(Cout).all())))
            np.savez(out / 'fields' / f'{nm}_c{j}.npz', f16=quick16[(nm, j)], internal=np.asarray(o[1]))
            del o
        its = np.array([r['iterations'] for r in recs])
        rs = np.array([r['reasons'] for r in recs])
        rep['arms'][nm].update(quick=recs, compile_and_first_s=compile_s, cases_run=len(run_cases),
                               worst_same_grid=max(r['worst_same_grid'] for r in recs),
                               median_same_grid=float(np.median([r['worst_same_grid'] for r in recs])),
                               lm_iterations_per_query_median=float(np.median(its.sum(1))),
                               lm_iterations_per_query_max=int(its.sum(1).max()),
                               reason_counts={str(k): int((rs == k).sum()) for k in range(5)},
                               all_finite=all(r['finite'] for r in recs))
        log(f"ARM {nm}: same-grid worst {rep['arms'][nm]['worst_same_grid']:.4%} median "
            f"{rep['arms'][nm]['median_same_grid']:.4%} its {rep['arms'][nm]['lm_iterations_per_query_median']:.0f} "
            f"reasons {rep['arms'][nm]['reason_counts']}")
        save()

    # distances between rollouts in the field metric of the mesh (||G_hat (c1 - c2)|| = ||L^T (c1 - c2)||)
    for nm, s in arms.items():
        Lt = np.asarray(s['data']['L']).T
        dist = {}
        for other in (f"tensor_R{s['Rp']}", f"dense_R{s['Rp']}", f"{cfg['converged_rule']}_R{s['Rp']}"):
            if other == nm or other not in arms:
                continue
            per = []
            for j in cases:
                if (nm, j) in COEF and (other, j) in COEF:
                    d = np.linalg.norm((COEF[(nm, j)] - COEF[(other, j)]) @ Lt.T, axis=1) / ref0[j]
                    per.append(float(d[1:].max()))
            if per:
                dist[other.rsplit('_R', 1)[0]] = dict(worst=max(per), median=float(np.median(per)), cases=len(per),
                                                      per_case=per)
        rep['arms'][nm]['distance'] = dist
    save()

    foms = {}
    FOM65 = {}
    for dt, nt, lt in cfg['fom_grid']:
        name = f'fom_dt{dt:g}_nt{nt:g}_lt{lt:g}'
        foms[name] = C.make_fom(n, dt, nt, lt)
        recs = []
        for j in cases:
            f, it, rn = foms[name](U0[j], NU[j])
            block(f)
            r = jnp.asarray(REF[j])
            e = np.asarray(errf(f, r))
            del r
            quick16[(name, j)] = np.asarray(f[:, ridx])
            if has65:
                FOM65[(name, j)] = np.asarray(f[:, i65])
            recs.append(dict(case=j, err_same_grid=e.tolist(), worst_same_grid=float(e[1:].max()),
                             newton=np.asarray(it).tolist(), max_rel_residual=float(jnp.max(rn)),
                             hit_newton_cap=bool(int(jnp.max(it)) >= C.MAX_NEWTON),
                             finite=bool(np.isfinite(np.asarray(f)).all())))
            np.savez(out / 'fields' / f'{name}_c{j}.npz', f16=quick16[(name, j)],
                     **({'f65': FOM65[(name, j)]} if has65 else {}))
            del f
        rep['fom'][name] = dict(dt=dt, ntol=nt, ltol=lt, quick=recs,
                                worst_same_grid=max(r['worst_same_grid'] for r in recs),
                                median_same_grid=float(np.median([r['worst_same_grid'] for r in recs])),
                                all_finite=all(r['finite'] for r in recs))
        log(f"FOM {name}: same-grid worst {rep['fom'][name]['worst_same_grid']:.4%}")
        save()
    REF65 = {j: np.asarray(REF[j][:, np.asarray(i65)]) for j in cases} if has65 else {}
    del REF

    # ================================================================ 5. timing A-B-A + microbenchmark
    reps = cfg['reps']
    tcases = cases[:cfg['timing_cases']]
    timed_arms = [nm for nm, s in arms.items() if s['family'] != 'dense']
    prng = np.random.default_rng(cfg.get('timing_seed', 20261002))
    x = jnp.ones((2048, 2048))

    def burn(sec=2.0):
        t0_ = time.perf_counter()
        while time.perf_counter() - t0_ < sec:
            block(x @ x)
    inv = []

    def timed(kind, nm, j, phase):
        t = time.perf_counter()
        o = run(nm, U0[j], NU[j]) if kind == 'rom' else foms[nm](U0[j], NU[j])
        block(o)
        dt_ = time.perf_counter() - t
        d = float(jnp.max(jnp.abs(o[0][:, ridx] - jnp.asarray(quick16[(nm, j)]))))
        inv.append(dict(kind=kind, name=nm, case=j, phase=phase, seconds=dt_, max_diff_vs_quick=d))
        if dt_ >= 0.5:
            time.sleep(min(1.0, dt_))
            burn(0.2)

    for phase, kind, names in (('A1', 'rom', timed_arms), ('B', 'fom', list(foms)), ('A2', 'rom', timed_arms)):
        order_ = [(nm, j) for nm in names for j in tcases for _ in range(reps)]
        prng.shuffle(order_)
        burn()
        for nm, j in order_:
            timed(kind, nm, j, phase)
        log(f'timing phase {phase} done ({len(order_)} invocations)')
    rep['timing']['invocations'] = inv
    tim = {}
    for nm in timed_arms + list(foms):
        rows_ = [r for r in inv if r['name'] == nm]
        tim[nm] = dict(median_ms=1e3 * float(np.median([r['seconds'] for r in rows_])), n=len(rows_),
                       max_diff_vs_quick=max(r['max_diff_vs_quick'] for r in rows_))
        if nm in arms:
            a1 = [r['seconds'] for r in rows_ if r['phase'] == 'A1']
            a2 = [r['seconds'] for r in rows_ if r['phase'] == 'A2']
            tim[nm]['drift_A2_over_A1'] = float(np.median(a2) / np.median(a1))
    rep['timing']['summary'] = tim
    rom_inv = [r for r in inv if r['kind'] == 'rom']
    med = {}
    for r in rom_inv:
        med.setdefault((r['name'], r['case']), []).append(r['seconds'])
    med = {k: np.median(v) for k, v in med.items()}
    after_long, after_short = [], []
    for ph in ('A1', 'A2'):
        seq = [r for r in rom_inv if r['phase'] == ph]
        for prev, cur in zip(seq[:-1], seq[1:]):
            v = cur['seconds'] / med[(cur['name'], cur['case'])]
            (after_long if tim[prev['name']]['median_ms'] >= 4 * tim[cur['name']]['median_ms'] else after_short).append(v)
    drift = [v['drift_A2_over_A1'] for v in tim.values() if 'drift_A2_over_A1' in v]
    rep['gates']['timing_drift_worst'] = float(max(max(drift), 1 / min(drift)))
    rep['gates']['timing_drift_pass'] = bool(rep['gates']['timing_drift_worst'] <= 1.10)
    nb = float(np.mean(after_long) / np.mean(after_short)) if after_long and after_short else 1.0
    fom_inv = [r for r in inv if r['kind'] == 'fom']
    fmed = {}
    for r in fom_inv:
        fmed.setdefault((r['name'], r['case']), []).append(r['seconds'])
    fmed = {k: np.median(v) for k, v in fmed.items()}
    fl, fs = [], []
    for prev, cur in zip(fom_inv[:-1], fom_inv[1:]):
        v = cur['seconds'] / fmed[(cur['name'], cur['case'])]
        (fl if tim[prev['name']]['median_ms'] >= 4 * tim[cur['name']]['median_ms'] else fs).append(v)
    nbf = float(np.mean(fl) / np.mean(fs)) if fl and fs else 1.0
    rep['gates']['timing_neighbour_ratio'] = nb
    rep['gates']['timing_neighbour_ratio_fom'] = nbf
    rep['gates']['timing_neighbour_pass'] = bool(1 / 1.10 <= nb <= 1.10 and 1 / 1.10 <= nbf <= 1.10)
    rep['gates']['deterministic_outputs'] = max(r['max_diff_vs_quick'] for r in inv)
    rep['gates']['deterministic_pass'] = bool(rep['gates']['deterministic_outputs'] <= 1e-12)
    log(f"timing gates drift {rep['gates']['timing_drift_worst']:.3f} neighbour {nb:.3f} / {nbf:.3f} "
        f"determinism {rep['gates']['deterministic_outputs']:.2e}")
    # microbenchmark: one advection-Jacobian evaluation (the only thing the rule changes), jitted, median of 50
    for nm, s in arms.items():
        if s['family'] == 'dense':
            continue
        fn = OM.contract_tensor if s['family'] == 'tensor' else OM.contract_offmesh
        f_j = jax.jit(fn)
        cc = jnp.asarray(COEF[(nm, 0)][3])
        block(f_j(s['data'], cc))
        ts = []
        for _ in range(50):
            t = time.perf_counter()
            block(f_j(s['data'], cc))
            ts.append(time.perf_counter() - t)
        rep['microbench'][nm] = dict(jacobian_ms_median=1e3 * float(np.median(ts)),
                                     jacobian_ms_min=1e3 * float(np.min(ts)))
    for nm, v in tim.items():
        log(f"TIME {nm}: {v['median_ms']:.2f} ms" + (f" (Ju {rep['microbench'][nm]['jacobian_ms_median']:.3f} ms)"
                                                       if nm in rep['microbench'] else ''))
    rep['complete_except_refined'] = True
    save()

    # ================================================================ 6. refined reference errors (65-node lattice)
    # in-job only if the reference job has finished within refined_wait_hours; otherwise the NumPy script refine_q.py
    # computes the same errors offline from the saved coefficients and f65 fields (it always runs, as the audit)
    np.savez(out / 'fields' / 'coefficients.npz', **{f'{nm}__c{j}': v for (nm, j), v in COEF.items()})
    np.savez(out / 'fields' / 'same_grid_ref65.npz', **{f'c{j}': v for j, v in REF65.items()})
    ref_ready = False
    if has65 and cfg.get('refined_ref'):
        ref_path = Path(cfg['refined_ref'])
        done = ref_path.with_suffix('.done')
        t_w = time.perf_counter()
        while not done.exists() and time.perf_counter() - t_w < 3600 * cfg.get('refined_wait_hours', 1.5):
            time.sleep(60)
        ref_ready = done.exists()
        rep['refined_in_job'] = ref_ready
    if ref_ready:
        rep['refined'] = dict(path=str(ref_path), sha256=sha_file(ref_path), done=done.read_text().strip())
        z = np.load(ref_path)
        assert int(z['seed']) == cfg['cohort_seed'] and int(z['count']) >= cfg['cohort_count'], (z['seed'], z['count'])
        RR = {j: np.asarray(z[f'c{j}']) for j in cases}
        n0 = {j: float(np.linalg.norm(RR[j][0])) for j in cases}
        rep['refined']['initial_match'] = max(rel_max(REF65[j][0], RR[j][0]) for j in cases)
        G65 = np.asarray(TB.feature_rows(model['bank'], model['T'], lattice65_coords()))      # (R, 63^3)
        rep['refined']['same_grid_reference'] = [float((np.linalg.norm(REF65[j] - RR[j], axis=1) / n0[j])[1:].max())
                                                 for j in cases]
        for nm, s in arms.items():
            per = []
            for j in cases:
                if (nm, j) not in COEF:
                    continue
                F = COEF[(nm, j)] @ G65[:s['Rp']]
                e = np.linalg.norm(F - RR[j], axis=1) / n0[j]
                per.append(float(e[1:].max()))
                rep['arms'][nm]['quick'][[q['case'] for q in rep['arms'][nm]['quick']].index(j)]['err_refined'] = \
                    e.tolist()
            rep['arms'][nm]['worst_refined'] = max(per)
            rep['arms'][nm]['median_refined'] = float(np.median(per))
        for name in foms:
            per = []
            for j in cases:
                e = np.linalg.norm(FOM65[(name, j)] - RR[j], axis=1) / n0[j]
                per.append(float(e[1:].max()))
                rep['fom'][name]['quick'][j]['err_refined'] = e.tolist()
            rep['fom'][name]['worst_refined'] = max(per)
            rep['fom'][name]['median_refined'] = float(np.median(per))
        np.save(out / 'fields' / 'G65_case_check.npy', G65[:, :64])
        for nm in arms:
            log(f"REFINED {nm}: worst {rep['arms'][nm]['worst_refined']:.4%} median "
                f"{rep['arms'][nm]['median_refined']:.4%}")
        for name in foms:
            log(f"REFINED {name}: worst {rep['fom'][name]['worst_refined']:.4%}")
    rep['complete'] = True
    rep['seconds'] = el()
    save()
    print('PANEL COMPLETE', flush=True)


if __name__ == '__main__':
    main()
