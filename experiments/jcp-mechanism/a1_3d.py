"""jcp-mechanism A1, Burgers 3D (DESIGN.md sections 2, 3, 5): the `nodes` arm (interior mesh nodes, weights h^3, the bank's
analytic (1,1,1)-derivative) run end-to-end beside the tensor, the selected off-mesh rules and the converged lattice, on
the validation cohort, one mesh per invocation.

Every solver, table and rule function is the unchanged vendor code (vendor/quad3d/offmesh.py and its vendor modules);
this file only (i) builds the `nodes` rule through the same `offmesh.point_blocks` / `offmesh.test_block` path,
(ii) drops the phases A1 does not need (same-grid FOM references, FOM grid, timing) and (iii) scores against the
refined and same-grid references staged with the job.

Phases (result.json is rewritten after each):
  1 tables   vendor tables (bank rows, Gram, A, tensor) and the tensor-vs-direct gate
  2 rules    B, D, P for gl24, lat4096, lat32768 (vendor rules.npz) and for `nodes`; gates G1, G2a-c, G3
  3 rho      tensor rollouts on the certification draws; rho of tensor, dense upwind, each rule and `nodes` against the
             continuum target (Gauss 80^3, check 64^3) and the mesh target; G4a (rho reproduces the 2026-10-01 job)
  4 cohort   every arm on every validation case: coefficients, iterations, exits; errors against the refined reference
             (513 nodes, 63^3 lattice x = k/64) and the same-grid reference restricted to that lattice; distances
Usage: python a1_3d.py --config configs/a1d3_n65.json --out <dir>
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
sys.path.insert(0, str(HERE / 'vendor' / 'quad3d'))
import offmesh as OM  # noqa: E402
from offmesh import C, TB  # noqa: E402

block = jax.block_until_ready
ROOT = HERE.parents[1]


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def rel_max(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def lattice65_coords():
    ax = np.arange(1, 64) / 64.0
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing='ij')
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)


def nodes_rule(n):
    """Interior mesh nodes (x slowest, the mesh ordering of common.interior_coords) and equal weights h^3."""
    X = C.interior_coords(n)
    return X, np.full(len(X), float(n - 1) ** -3)


def nodes_tables(n, bank, T, kx, Rp, M, chunk=1 << 18):
    """B, D (m, R'), P (m, M) for the `nodes` rule through the vendor point_blocks / test_block path, built in point
    chunks so that no (m, M) temporary larger than one chunk exists."""
    X, w = nodes_rule(n)
    B, D = OM.point_blocks(bank, np.asarray(T)[:, :Rp], X, chunk=1 << 15)
    kk = np.asarray(kx)[:M]
    P = jnp.concatenate([block(OM.test_block(n, X[s:s + chunk], w[s:s + chunk], kk)) for s in range(0, len(X), chunk)], 0)
    return dict(B=block(B), D=block(D), P=block(P)), X, w


def adv_chunked(data, Cs, sc=64):
    """offmesh.adv_offmesh_batch over state chunks (bounded (S, m) temporaries)."""
    f = jax.jit(OM.adv_offmesh_batch)
    return np.concatenate([np.asarray(f(data, jnp.asarray(Cs[i:i + sc]))) for i in range(0, len(Cs), sc)], 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    (out / 'fields').mkdir(parents=True, exist_ok=True)
    if not cfg.get('local_smoke'):
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
    b = pickle.loads((ROOT / cfg['model'] / 'bank.pkl').read_bytes())
    model = dict(bank=jax.tree_util.tree_map(jnp.asarray, b['params']), T=np.asarray(b['rotation']),
                 spread=b['coefficient_rms_spread'], sha=sha_file(ROOT / cfg['model'] / 'bank.pkl'))
    assert model['sha'] == cfg['expected_model_sha256'], model['sha']
    rules_path = ROOT / cfg['rules_npz']
    rules_sha = sha_file(rules_path)
    assert rules_sha == cfg['expected_rules_sha256'], rules_sha
    z = np.load(rules_path)
    rule_names = list(cfg['rules'])
    rules = {nm: (np.asarray(z[f'{nm}_X']), np.asarray(z[f'{nm}_w']))
             for nm in rule_names + [cfg['continuum_target'], cfg['continuum_check']]}
    rep = dict(config=cfg, mesh=n, commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
               host=os.uname().nodename, backend=jax.default_backend(), gpu=jax.devices()[0].device_kind,
               nvidia_smi=smi, x64=True, precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'],
               jax_version=jax.__version__, model_sha256=model['sha'], rules_sha256=rules_sha, gates={}, rho={},
               arms={}, sizes={}, complete=False)
    save = lambda: C.dump(out / 'result.json', C.clean(rep))
    save()

    # ================================================================ 1. tables
    R = int(model['T'].shape[1])
    Rps = sorted(cfg['Rps'], reverse=True)
    order, lam_all = C.mode_order(n)
    M_of = lambda r: C.complete_M(n, 4 * r, order, lam_all)
    tb = TB.build_tables(n, model['bank'], model['T'], sorted(set(Rps) | {R}), M_of, log=log)
    rep['gates']['gram_condition'] = tb['gram_cond'] ** 2
    assert rep['gates']['gram_condition'] <= 1e8, rep['gates']
    kx = tb['kxyz']
    rng = np.random.default_rng(20261008)
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

    # ================================================================ 2. rules + gates
    om_full = {nm: OM.offmesh_tables(n, model['bank'], model['T'], *rules[nm], kx) for nm in rule_names}
    for nm in rule_names:
        rep['sizes'][nm] = dict(m=int(len(rules[nm][1])), weight_sum=float(np.sum(rules[nm][1])))
    Xn, wn = nodes_rule(n)
    rep['sizes']['nodes'] = dict(m=int(len(wn)), weight_sum=float(np.sum(wn)))
    # G1: analytic directional derivative vs centred finite differences, at 32 interior mesh nodes
    sel = np.linspace(0, len(Xn) - 1, 32).astype(int)
    Xg = Xn[sel]
    _, Dg = OM.point_blocks(model['bank'], model['T'], Xg)
    hfd = 1e-5
    fp, _ = OM.point_blocks(model['bank'], model['T'], Xg + hfd)
    fm, _ = OM.point_blocks(model['bank'], model['T'], Xg - hfd)
    rep['gates']['G1_derivative_vs_fd_nodes'] = rel_max((fp - fm) / (2 * hfd), Dg)
    assert rep['gates']['G1_derivative_vs_fd_nodes'] < 1e-6, rep['gates']
    # G3 on the nodes path is checked per width below (same Jacobian formula as every off-mesh rule)
    log(f"rules built; G1 {rep['gates']['G1_derivative_vs_fd_nodes']:.2e}")
    save()

    # ================================================================ 3 + 4 per width (one nodes table resident at a time)
    tab = C.table(cfg['cohort_seed'], cfg['cohort_count'])
    rep['cohort'] = dict(seed=cfg['cohort_seed'], count=cfg['cohort_count'], table_sha256=tab['sha256'])
    assert cfg.get('expected_cohort_sha256') in (None, tab['sha256']), 'cohort differs'
    cases = list(range(cfg['cohort_count']))
    keep = int(round(0.05 / cfg['dt']))
    # references on the 63^3 lattice x = k/64 (refined: 513 nodes; same grid: this mesh, tight tolerances)
    refz = np.load(cfg['refined_ref'])
    rep['refined'] = dict(path=cfg['refined_ref'], sha256=sha_file(cfg['refined_ref']))
    assert rep['refined']['sha256'] == cfg['expected_refined_sha256'], rep['refined']
    assert int(refz['seed']) == cfg['cohort_seed'] and int(refz['count']) >= cfg['cohort_count']
    RR = {j: np.asarray(refz[f'c{j}']) for j in cases}
    sgz = np.load(cfg['same_grid_ref65'])
    rep['same_grid'] = dict(path=cfg['same_grid_ref65'], sha256=sha_file(cfg['same_grid_ref65']))
    assert rep['same_grid']['sha256'] == cfg['expected_same_grid_sha256'], rep['same_grid']
    SG = {j: np.asarray(sgz[f'c{j}']) for j in cases}
    n0 = {j: float(np.linalg.norm(RR[j][0])) for j in cases}
    G65 = block(TB.feature_rows(model['bank'], model['T'], lattice65_coords()))                       # (R, 63^3)
    lat_err = jax.jit(lambda Cm, G, ref, n0_: jnp.linalg.norm(Cm @ G - ref, axis=1) / n0_)
    rep['refined']['initial_match_same_grid'] = max(rel_max(SG[j][0], RR[j][0]) for j in cases)
    assert rep['refined']['initial_match_same_grid'] <= 1e-12, rep['refined']
    U0 = [block(jnp.asarray(C.initial_interior(n, tab, j))) for j in cases]
    NU = [float(tab['nu'][j]) for j in cases]
    Xc, wc = rules[cfg['continuum_target']]
    Xk, wk = rules[cfg['continuum_check']]

    for Rp in Rps:
        M = int(M_of(Rp))
        trust = float(cfg['trust_fraction'] * model['spread'][str(Rp)])
        base = TB.arm_data(tb, Rp, M)
        lean = {k: v for k, v in base.items() if k != 'Ts'}
        nd, _, _ = nodes_tables(n, model['bank'], model['T'], kx, Rp, M)
        # G2a: the nodes value rows are the mesh bank rows; G2b: the nodes test block is Phi at the nodes
        dmax, rmax, off = 0., 0., 0
        for g in TB.rows_prefix(tb['GTb'], Rp):                                                     # (r_b, N) blocks
            dmax = max(dmax, float(jnp.max(jnp.abs(nd['B'][:, off:off + g.shape[0]].T - g))))
            rmax = max(rmax, float(jnp.max(jnp.abs(g))))
            off += g.shape[0]
        rep['gates'][f'G2a_nodes_B_vs_mesh_bank_R{Rp}'] = dmax / rmax
        ii = np.linspace(0, len(Xn) - 1, 4096).astype(int)
        ijk = np.stack(np.unravel_index(ii, (n - 2,) * 3), 1) + 1
        rep['gates'][f'G2b_nodes_P_vs_Phi_R{Rp}'] = rel_max(np.asarray(nd['P'][jnp.asarray(ii)]),
                                                            C.phi_explicit(n, kx[:M], ijk))
        # G2c: the nodes advection by the off-mesh GEMM equals Phi^T (u * (1 . grad u)) by the DST, 4 states
        cs4 = jnp.asarray(rng.normal(size=(4, Rp)) / np.sqrt(Rp))
        v_gemm = adv_chunked(nd, np.asarray(cs4))
        uu = cs4 @ nd['B'].T
        du = cs4 @ nd['D'].T
        v_dst = np.asarray(C.phiT(uu * du, n, tuple(jnp.asarray(kx[:M, j]) for j in range(3))))
        rep['gates'][f'G2c_nodes_gemm_vs_dst_R{Rp}'] = rel_max(v_gemm, v_dst)
        # G3: the Jacobian formula vs jacfwd, and adv = 0.5 J c, on the nodes path
        c3 = jnp.asarray(rng.normal(size=Rp) / np.sqrt(Rp))
        J_f = OM.contract_offmesh(nd, c3)
        J_ad = jax.jacfwd(lambda cc: OM.adv_offmesh_batch(nd, cc[None])[0])(c3)
        rep['gates'][f'G3_nodes_jacobian_vs_jacfwd_R{Rp}'] = rel_max(J_f, J_ad)
        rep['gates'][f'G3_nodes_adv_half_Jc_R{Rp}'] = rel_max(0.5 * J_f @ c3, OM.adv_offmesh_batch(nd, c3[None])[0])
        del uu, du, J_ad
        log(f'R{Rp} M{M} nodes gates ' + ', '.join(f'{k} {v:.2e}' for k, v in rep['gates'].items() if f'R{Rp}' in k))
        assert rep['gates'][f'G2a_nodes_B_vs_mesh_bank_R{Rp}'] < 1e-12, rep['gates']
        assert rep['gates'][f'G2b_nodes_P_vs_Phi_R{Rp}'] < 1e-12, rep['gates']
        assert rep['gates'][f'G2c_nodes_gemm_vs_dst_R{Rp}'] < 1e-11, rep['gates']
        assert rep['gates'][f'G3_nodes_jacobian_vs_jacfwd_R{Rp}'] < 1e-12, rep['gates']
        assert rep['gates'][f'G3_nodes_adv_half_Jc_R{Rp}'] < 1e-12, rep['gates']
        save()

        arms = {f'tensor_R{Rp}': dict(family='tensor', data=base,
                                      q=OM.make_fsc_rule(n, Rp, M, OM.contract_tensor, dt=cfg['dt'], gtol=cfg['gtol'],
                                                         trust=trust))}
        for nm in rule_names:
            d = dict(lean, B=om_full[nm]['B'][:, :Rp], D=om_full[nm]['D'][:, :Rp], P=om_full[nm]['P'][:, :M])
            arms[f'{nm}_R{Rp}'] = dict(family='offmesh', data=jax.tree_util.tree_map(lambda x: block(jnp.asarray(x)), d),
                                       q=OM.make_fsc_rule(n, Rp, M, OM.contract_offmesh, dt=cfg['dt'],
                                                          gtol=cfg['gtol'], trust=trust))
        arms[f'nodes_R{Rp}'] = dict(family='nodes', data=dict(lean, **nd),
                                    q=OM.make_fsc_rule(n, Rp, M, OM.contract_offmesh, dt=cfg['dt'], gtol=cfg['gtol'],
                                                       trust=trust))
        for nm, s in arms.items():
            rep['arms'][nm] = dict(Rp=Rp, M=M, family=s['family'], trust=trust)
            if s['family'] != 'tensor':
                m_ = int(s['data']['B'].shape[0])
                rep['arms'][nm]['m'] = m_
                rep['arms'][nm]['bytes'] = 8 * m_ * (2 * Rp + M)
                rep['arms'][nm]['flops_jacobian'] = 2 * m_ * Rp * (2 + M) + 4 * m_ * Rp
            else:
                rep['arms'][nm]['bytes'] = 8 * M * Rp * Rp
                rep['arms'][nm]['flops_jacobian'] = 2 * M * Rp * Rp

        def run(nm, u0, nu):
            s = arms[nm]
            return s['q'](u0, nu, s['data'], {})

        # ---------------------------------------------------- 3. rho on the certification draws (tensor-reached)
        states, kidx = [], []
        for seed, count in cfg['cert_draws']:
            tc = C.table(seed, count)
            for j in range(count):
                o = run(f'tensor_R{Rp}', jnp.asarray(C.initial_interior(n, tc, j)), float(tc['nu'][j]))
                ws = np.asarray(o[1])
                assert np.isfinite(ws).all() and int((np.asarray(o[3]) == 3).sum()) == 0, ('cert rollout', seed, j)
                states.append(ws)
                kidx.append(np.arange(ws.shape[0]))
                del o
        Cs = np.concatenate(states, 0)
        kk = np.concatenate(kidx)
        tgt_mesh, umin = map(np.asarray, OM.make_mesh_target(n, M, kx)(jnp.asarray(Cs), base))
        tgt_cont = np.asarray(OM.continuum_adv(n, model['bank'], model['T'], Xc, wc, kx[:M], Cs))
        chk_cont = np.asarray(OM.continuum_adv(n, model['bank'], model['T'], Xk, wk, kx[:M], Cs))
        ten = np.asarray(jax.jit(lambda d, X: jax.lax.map(lambda cc: 0.5 * OM.contract_tensor(d, cc) @ cc, X))(
            base, jnp.asarray(Cs)))
        rho = lambda v, t: np.linalg.norm(v - t, axis=1) / np.maximum(np.linalg.norm(t, axis=1), 1e-300)

        def summ(r):
            ev, k0 = r[kk >= 1], r[kk == 0]
            return dict(worst=float(ev.max()), median=float(np.median(ev)), p90=float(np.quantile(ev, 0.9)),
                        worst_k0=float(k0.max()), median_k0=float(np.median(k0)))
        res = dict(states=int(len(Cs)), states_evolved=int((kk >= 1).sum()), M=M, umin=float(umin.min()),
                   continuum_check=summ(rho(chk_cont, tgt_cont)), rules={})
        per = {'tensor': (rho(ten, tgt_cont), rho(ten, tgt_mesh)), 'dense_upwind': (rho(tgt_mesh, tgt_cont), 0 * kk)}
        for nm in rule_names + ['nodes']:
            v = adv_chunked(arms[f'{nm}_R{Rp}']['data'], Cs)
            per[nm] = (rho(v, tgt_cont), rho(v, tgt_mesh))
        for nm, (rc, rm) in per.items():
            res['rules'][nm] = dict(cont=summ(rc), mesh=summ(rm))
        np.savez(out / 'fields' / f'rho_R{Rp}.npz', Cs=Cs, k=kk, tgt_cont=tgt_cont, tgt_mesh=tgt_mesh, chk_cont=chk_cont,
                 **{f'rho_cont_{nm}': v[0] for nm, v in per.items()}, **{f'rho_mesh_{nm}': v[1] for nm, v in per.items()})
        # G4a: rho of the vendor arms reproduces the 2026-10-01 validation job at this mesh (same states, same code)
        prev = cfg['expected_rho'][str(Rp)]
        rep['gates'][f'G4a_rho_reproduction_R{Rp}'] = max(
            abs(res['rules'][nm]['cont']['worst'] - prev[nm]) / prev[nm] for nm in prev)
        rep['rho'][str(Rp)] = res
        log(f"RHO R{Rp}: check {res['continuum_check']['worst']:.2e}; G4a {rep['gates'][f'G4a_rho_reproduction_R{Rp}']:.2e}; "
            + '; '.join(f"{k} c{v['cont']['worst']:.2e}/{v['cont']['median']:.2e} m{v['mesh']['worst']:.2e}"
                        for k, v in res['rules'].items()))
        save()
        del tgt_mesh, tgt_cont, chk_cont, Cs, ten

        # ---------------------------------------------------- 4. cohort
        COEF = {}
        for nm in arms:
            recs = []
            t_c = time.perf_counter()
            internal = []
            for j in cases:
                t_q = time.perf_counter()
                o = run(nm, U0[j], NU[j])
                block(o)
                ws = np.asarray(o[1])
                Cout = ws[::keep]
                COEF[(nm, j)] = Cout
                internal.append(ws)
                e_ref = np.asarray(lat_err(jnp.asarray(Cout), G65[:Rp], jnp.asarray(RR[j]), n0[j]))
                e_sg = np.asarray(lat_err(jnp.asarray(Cout), G65[:Rp], jnp.asarray(SG[j]), n0[j]))
                its, rs = np.asarray(o[2]), np.asarray(o[3])
                recs.append(dict(case=j, seconds=time.perf_counter() - t_q, err_refined=e_ref.tolist(),
                                 worst_refined=float(e_ref[1:].max()), err_same_grid65=e_sg.tolist(),
                                 worst_same_grid65=float(e_sg[1:].max()), iterations=int(its.sum()),
                                 reasons={str(k_): int((rs == k_).sum()) for k_ in range(5)},
                                 rejected=int(np.asarray(o[6]).sum()), finite=bool(np.isfinite(ws).all())))
                del o
            np.savez(out / 'fields' / f'internal_{nm}.npz', internal=np.stack(internal, 0))
            wr = [r['worst_refined'] for r in recs]
            ws_ = [r['worst_same_grid65'] for r in recs]
            rep['arms'][nm].update(cases=recs, seconds_total=time.perf_counter() - t_c,
                                   worst_refined=max(wr), median_refined=float(np.median(wr)),
                                   worst_same_grid65=max(ws_), median_same_grid65=float(np.median(ws_)),
                                   all_finite=all(r['finite'] for r in recs),
                                   reason3_total=sum(r['reasons']['3'] for r in recs),
                                   iterations_median=float(np.median([r['iterations'] for r in recs])))
            log(f"ARM {nm}: refined worst {max(wr):.4%} median {np.median(wr):.4%}; same-grid65 worst {max(ws_):.4%}; "
                f"its {rep['arms'][nm]['iterations_median']:.0f}; {rep['arms'][nm]['seconds_total']:.0f}s")
            save()
        zero = jnp.zeros((6, G65.shape[1]))
        for nm in arms:                     # field distance on the 63^3 lattice, normalised like the errors
            dist = {}
            for other in (f'tensor_R{Rp}', f"{cfg['converged_rule']}_R{Rp}", f'nodes_R{Rp}'):
                if other == nm:
                    continue
                per_c = [float(np.asarray(lat_err(jnp.asarray(COEF[(nm, j)] - COEF[(other, j)]), G65[:Rp], zero,
                                                  n0[j]))[1:].max()) for j in cases]
                dist[other.rsplit('_R', 1)[0]] = dict(worst=max(per_c), median=float(np.median(per_c)), per_case=per_c)
            rep['arms'][nm]['distance'] = dist
        np.savez(out / 'fields' / f'coefficients_R{Rp}.npz', **{f'{nm}__c{j}': v for (nm, j), v in COEF.items()})
        save()
        del arms, nd, base, lean, COEF

    rep['complete'] = True
    rep['seconds'] = el()
    save()
    print('A1-3D COMPLETE', flush=True)


if __name__ == '__main__':
    main()
