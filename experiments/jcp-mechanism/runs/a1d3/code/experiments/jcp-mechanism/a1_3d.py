"""jcp-mechanism A1, Burgers 3D (DESIGN.md sections 2, 3, 5): the `nodes` arm (interior mesh nodes, weights h^3, the bank's
analytic (1,1,1)-derivative) run end-to-end beside the tensor, the selected off-mesh rules and the converged lattice, on
the validation cohort, one mesh per invocation.

Every solver, table and rule function is the unchanged vendor code (vendor/quad3d/offmesh.py and its vendor modules);
this file only (i) builds the `nodes` rule: B, D by the off-mesh `offmesh.point_blocks` path at the interior mesh nodes,
tests applied by the exact identity P = Phi (weights h^3) through the vendor DST `common.phiT` (DESIGN amendment A1-2:
the explicit P at 128^3 would have 4.2e9 elements), gated against the off-mesh GEMM with `offmesh.test_block`;
(ii) drops the phases A1 does not need (same-grid FOM references, FOM grid, timing); (iii) scores against the refined
and same-grid references staged with the job; (iv) adds the amendment A1-6 diagnostics (rho on nodes-reached states,
adaptive-LM sensitivity rerun at the first mesh).

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


def gate(rep, name, a, b, tol):
    """DESIGN amendment 3 (A3-2): pass iff a, b finite, max|b| >= 1e-8 and max|a - b| <= tol max|b|. Aborts on failure."""
    a, b = np.asarray(a), np.asarray(b)
    ok_ = bool(np.isfinite(a).all() and np.isfinite(b).all() and np.abs(b).max() >= 1e-8)
    v = rel_max(a, b) if ok_ else float('inf')
    rep['gates'][name] = v
    assert ok_ and v <= tol, (name, v, tol, float(np.abs(b).max()))
    return v


def task_path(p):
    """Config paths of staged references are relative to the job's TASK_ROOT (absolute paths are used as given)."""
    p = Path(p)
    return p if p.is_absolute() else Path(os.environ['TASK_ROOT']) / p


def lattice65_coords():
    ax = np.arange(1, 64) / 64.0
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing='ij')
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)


def nodes_rule(n):
    """Interior mesh nodes (x slowest, the mesh ordering of common.interior_coords) and equal weights h^3."""
    X = C.interior_coords(n)
    return X, np.full(len(X), float(n - 1) ** -3)


def nodes_blocks(n, bank, T, Rp):
    """B, D (N, R') of the `nodes` rule: the vendor off-mesh point_blocks at the interior mesh nodes."""
    X, _ = nodes_rule(n)
    B, D = OM.point_blocks(bank, np.asarray(T)[:, :Rp], X, chunk=1 << 15)
    return dict(B=block(B), D=block(D))


def make_nodes_ops(n, M, kx, ch=16):
    """Jacobian and value of the `nodes` rule with the tests applied by the DST (P = Phi at the nodes, weights h^3):
    Ju(c) = Phi^T (diag(D c) B + diag(B c) D), adv(c) = Phi^T ((B c) * (D c)) = 0.5 Ju(c) c."""
    idx = tuple(jnp.asarray(np.asarray(kx)[:M, a]) for a in range(3))

    def contract(data, w):
        B, D = data['B'], data['D']
        cols = ((D @ w)[:, None] * B + (B @ w)[:, None] * D).T                       # (R', N)
        Rp, N = cols.shape
        pad = (-Rp) % ch
        cols = jnp.concatenate((cols, jnp.zeros((pad, N))), 0).reshape(-1, ch, N)
        return jax.lax.map(lambda blk: C.phiT(blk, n, idx), cols).reshape(-1, M)[:Rp].T

    def value(data, Cs):                                                             # (S, R') -> (S, M)
        return C.phiT((Cs @ data['B'].T) * (Cs @ data['D'].T), n, idx)
    return contract, value


def gemm_check(n, X, w, kx, M, B, D, cs, cols, chunk=1 << 17):
    """The off-mesh GEMM with the explicit test block offmesh.test_block, accumulated over point chunks:
    values P^T((B c)(D c)) for the states cs and the Jacobian columns `cols` at cs[0]."""
    kk = np.asarray(kx)[:M]
    val, jac = 0., 0.
    w0 = cs[0]
    for s in range(0, len(X), chunk):
        P = OM.test_block(n, X[s:s + chunk], w[s:s + chunk], kk)
        Bc, Dc = B[s:s + chunk], D[s:s + chunk]
        val = val + ((cs @ Bc.T) * (cs @ Dc.T)) @ P
        jac = jac + P.T @ ((Dc @ w0)[:, None] * Bc[:, cols] + (Bc @ w0)[:, None] * Dc[:, cols])
        del P
    return np.asarray(val), np.asarray(jac)


def adv_chunked(f, data, Cs, sc=32):
    """A value function f(data, Cs) over state chunks (bounded (S, N) temporaries)."""
    fj = jax.jit(f)
    return np.concatenate([np.asarray(fj(data, jnp.asarray(Cs[i:i + sc]))) for i in range(0, len(Cs), sc)], 0)


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
    fd = 0.
    for ax_ in range(3):                    # centred differences per coordinate, summed (amendment A2-4)
        e_ = np.zeros(3)
        e_[ax_] = hfd
        fp, _ = OM.point_blocks(model['bank'], model['T'], Xg + e_)
        fm, _ = OM.point_blocks(model['bank'], model['T'], Xg - e_)
        fd = fd + (fp - fm) / (2 * hfd)
    gate(rep, 'G1_derivative_vs_fd_nodes', fd, Dg, 1e-6)
    # G3 on the nodes path is checked per width below (same Jacobian formula as every off-mesh rule)
    log(f"rules built; G1 {rep['gates']['G1_derivative_vs_fd_nodes']:.2e}")
    save()

    # ================================================================ 3 + 4 per width (one nodes table resident at a time)
    tab = C.table(cfg['cohort_seed'], cfg['cohort_count'])
    rep['cohort'] = dict(seed=cfg['cohort_seed'], count=cfg['cohort_count'], table_sha256=tab['sha256'])
    # report-only (code audit 2): the table hash includes GPU-computed floats and differs between the historical jobs
    rep['cohort']['matches_source_job_hash'] = cfg.get('expected_cohort_sha256') == tab['sha256']
    cases = list(range(cfg['cohort_count']))
    keep = int(round(0.05 / cfg['dt']))
    # references on the 63^3 lattice x = k/64 (refined: 513 nodes; same grid: this mesh, tight tolerances)
    refp, sgp = task_path(cfg['refined_ref']), task_path(cfg['same_grid_ref65'])
    refz = np.load(refp)
    rep['refined'] = dict(path=str(refp), sha256=sha_file(refp))
    assert rep['refined']['sha256'] == cfg['expected_refined_sha256'], rep['refined']
    assert int(refz['seed']) == cfg['cohort_seed'] and int(refz['count']) >= cfg['cohort_count']
    RR = {j: np.asarray(refz[f'c{j}']) for j in cases}
    sgz = np.load(sgp)
    rep['same_grid'] = dict(path=str(sgp), sha256=sha_file(sgp))
    assert rep['same_grid']['sha256'] == cfg['expected_same_grid_sha256'], rep['same_grid']
    SG = {j: np.asarray(sgz[f'c{j}']) for j in cases}
    n0 = {j: float(np.linalg.norm(RR[j][0])) for j in cases}
    G65 = block(TB.feature_rows(model['bank'], model['T'], lattice65_coords()))                       # (R, 63^3)
    lat_err = jax.jit(lambda Cm, G, ref, n0_: jnp.linalg.norm(Cm @ G - ref, axis=1) / n0_)
    zero = jnp.zeros((6, G65.shape[1]))
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
        nd = nodes_blocks(n, model['bank'], model['T'], Rp)
        n_contract, n_value = make_nodes_ops(n, M, kx)
        # G2a: the nodes value rows are the mesh bank rows, block by block
        off, dmax, rmax = 0, 0., 0.
        for g in TB.rows_prefix(tb['GTb'], Rp):                                                     # (r_b, N) blocks
            assert bool(jnp.all(jnp.isfinite(g))) and bool(jnp.all(jnp.isfinite(nd['B'][:, off:off + g.shape[0]])))
            dmax = max(dmax, float(jnp.max(jnp.abs(nd['B'][:, off:off + g.shape[0]].T - g))))
            rmax = max(rmax, float(jnp.max(jnp.abs(g))))
            off += g.shape[0]
        assert rmax >= 1e-8 and np.isfinite(dmax)
        rep['gates'][f'G2a_nodes_B_vs_mesh_bank_R{Rp}'] = dmax / rmax
        assert dmax / rmax <= 1e-12, rep['gates']
        # G2b: the off-mesh test block at the nodes (weights h^3) is Phi, at 4096 nodes
        ii = np.linspace(0, len(Xn) - 1, 4096).astype(int)
        ijk = np.stack(np.unravel_index(ii, (n - 2,) * 3), 1) + 1
        gate(rep, f'G2b_nodes_P_vs_Phi_R{Rp}', OM.test_block(n, Xn[ii], wn[ii], kx[:M]), C.phi_explicit(n, kx[:M], ijk),
             1e-12)
        # G2c: values (4 states) and Jacobian columns at each of the 4 states by the DST equal the off-mesh GEMM with
        # offmesh.test_block (all columns at the first mesh n <= 65, 32 random columns above; amendment A1-2)
        cs4 = jnp.asarray(rng.normal(size=(4, Rp)) / np.sqrt(Rp))
        cols = np.arange(Rp) if n <= 65 else np.sort(rng.choice(Rp, 32, replace=False))
        v_dst = np.asarray(jax.jit(n_value)(nd, cs4))
        vg, jd, jg = None, [], []
        for i in range(4):
            v_gemm, j_gemm = gemm_check(n, Xn, wn, kx, M, nd['B'], nd['D'], cs4[i:i + 1], jnp.asarray(cols))
            vg = v_gemm if vg is None else np.concatenate([vg, v_gemm])
            jd.append(np.asarray(jax.jit(n_contract)(nd, cs4[i]))[:, cols])
            jg.append(j_gemm)
        gate(rep, f'G2c_nodes_value_gemm_vs_dst_R{Rp}', v_dst, vg, 1e-11)
        gate(rep, f'G2c_nodes_jacobian_gemm_vs_dst_R{Rp}', np.stack(jd), np.stack(jg), 1e-11)
        # G3: the full Jacobian (all R' columns) vs forward-mode JVPs of the value along the basis, chunks of 16
        J_f = jax.jit(n_contract)(nd, cs4[0])
        basis = jnp.eye(Rp).reshape(-1, 16, Rp)
        jv = jax.jit(lambda d, c0, Bs: jax.lax.map(
            lambda E: jax.vmap(lambda v: jax.jvp(lambda cc: n_value(d, cc[None])[0], (c0,), (v,))[1])(E), Bs))(
            nd, cs4[0], basis).reshape(Rp, M).T
        gate(rep, f'G3_nodes_jacobian_vs_jvp_R{Rp}', J_f, jv, 1e-12)
        gate(rep, f'G3_nodes_adv_half_Jc_R{Rp}', 0.5 * J_f @ cs4[0], v_dst[0], 1e-12)
        del J_f, jv
        log(f'R{Rp} M{M} nodes gates ' + ', '.join(f'{k} {v:.2e}' for k, v in rep['gates'].items() if f'R{Rp}' in k))
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
                                    q=OM.make_fsc_rule(n, Rp, M, n_contract, dt=cfg['dt'], gtol=cfg['gtol'],
                                                       trust=trust))
        for nm, s in arms.items():
            rep['arms'][nm] = dict(Rp=Rp, M=M, family=s['family'], trust=trust)
            if s['family'] == 'offmesh':
                m_ = int(s['data']['B'].shape[0])
                rep['arms'][nm]['m'] = m_
                rep['arms'][nm]['bytes'] = 8 * m_ * (2 * Rp + M)
                rep['arms'][nm]['flops_jacobian'] = 2 * m_ * Rp * (2 + M) + 4 * m_ * Rp
            elif s['family'] == 'nodes':                 # B, D stored; tests applied by R' DSTs of the mesh field
                m_ = int(s['data']['B'].shape[0])
                rep['arms'][nm]['m'] = m_
                rep['arms'][nm]['bytes'] = 8 * m_ * 2 * Rp
            else:
                rep['arms'][nm]['bytes'] = 8 * M * Rp * Rp
                rep['arms'][nm]['flops_jacobian'] = 2 * M * Rp * Rp

        def run(nm, u0, nu):
            s = arms[nm]
            return s['q'](u0, nu, s['data'], {})

        def value_of(nm):
            fam = arms[nm]['family']
            if fam == 'nodes':
                return n_value
            if fam == 'offmesh':
                return OM.adv_offmesh_batch
            return lambda d, X: jax.lax.map(lambda cc: 0.5 * OM.contract_tensor(d, cc) @ cc, X)

        rho = lambda v, t: np.linalg.norm(v - t, axis=1) / np.maximum(np.linalg.norm(t, axis=1), 1e-300)

        def rho_all(Cs):
            """rho of every arm's tested advection against the continuum and mesh targets, per state."""
            tgt_mesh, umin = map(np.asarray, OM.make_mesh_target(n, M, kx)(jnp.asarray(Cs), base))
            # no retry (job audit 1): a non-finite target is recorded and invalidates the configuration (label X)
            tgt_cont = np.asarray(OM.continuum_adv(n, model['bank'], model['T'], Xc, wc, kx[:M], Cs))
            chk_cont = np.asarray(OM.continuum_adv(n, model['bank'], model['T'], Xk, wk, kx[:M], Cs))
            if not (np.isfinite(tgt_cont).all() and np.isfinite(chk_cont).all()):
                rep.setdefault('nonfinite_targets', []).append(dict(Rp=Rp, states=len(Cs)))
            per = {'dense_upwind': (rho(tgt_mesh, tgt_cont), np.zeros(len(Cs)))}
            for nm in arms:
                v = adv_chunked(value_of(nm), arms[nm]['data'], Cs)
                per[nm.rsplit('_R', 1)[0]] = (rho(v, tgt_cont), rho(v, tgt_mesh))
            return per, rho(chk_cont, tgt_cont), float(umin.min()), (tgt_cont, tgt_mesh, chk_cont)

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
        per, chk, umin, tg = rho_all(Cs)

        def summ(r):
            ev, k0 = r[kk >= 1], r[kk == 0]
            return dict(worst=float(ev.max()), median=float(np.median(ev)), p90=float(np.quantile(ev, 0.9)),
                        worst_k0=float(k0.max()), median_k0=float(np.median(k0)))
        res = dict(states=int(len(Cs)), states_evolved=int((kk >= 1).sum()), M=M, umin=umin,
                   continuum_check=summ(chk), rules={nm: dict(cont=summ(rc), mesh=summ(rm)) for nm, (rc, rm) in per.items()})
        np.savez(out / 'fields' / f'rho_R{Rp}.npz', Cs=Cs, k=kk, tgt_cont=tg[0], tgt_mesh=tg[1], chk_cont=tg[2],
                 **{f'rho_cont_{nm}': v[0] for nm, v in per.items()}, **{f'rho_mesh_{nm}': v[1] for nm, v in per.items()})
        # G4a: rho of the vendor arms reproduces the 2026-10-01 validation job at this mesh (same states, same code)
        prev = cfg['expected_rho'][str(Rp)]
        rep['gates'][f'G4a_rho_reproduction_R{Rp}'] = max(
            abs(res['rules'][nm]['cont']['worst'] - prev[nm]) / prev[nm] for nm in prev)
        res['continuum_target_valid'] = bool(np.isfinite(tg[0]).all() and np.isfinite(tg[2]).all() and
                                             res['continuum_check']['worst'] <= 1e-6)
        rep['rho'][str(Rp)] = res
        log(f"RHO R{Rp}: check {res['continuum_check']['worst']:.2e}; G4a {rep['gates'][f'G4a_rho_reproduction_R{Rp}']:.2e}; "
            + '; '.join(f"{k} c{v['cont']['worst']:.2e}/{v['cont']['median']:.2e} m{v['mesh']['worst']:.2e}"
                        for k, v in res['rules'].items()))
        save()
        del tg, Cs, per

        # ---------------------------------------------------- 4. cohort
        COEF, INTERNAL = {}, {}
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
                INTERNAL[(nm, j)] = ws
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
        # A1-6: rho of every rule on the nodes-reached states of the first 8 validation cases (k = 1..25)
        Cn = np.concatenate([INTERNAL[(f'nodes_R{Rp}', j)][1:] for j in cases[:8]], 0)
        per_n, chk_n, _, _ = rho_all(Cn)
        rep['rho_nodes_reached'] = rep.get('rho_nodes_reached', {})
        rep['rho_nodes_reached'][str(Rp)] = dict(
            states=int(len(Cn)), continuum_check_worst=float(chk_n.max()),
            rules={nm: dict(cont_worst=float(rc.max()), cont_median=float(np.median(rc)), mesh_worst=float(rm.max()))
                   for nm, (rc, rm) in per_n.items()})
        log('RHO nodes-reached R%d: ' % Rp + '; '.join(f"{k} c{v['cont_worst']:.2e}" for k, v in
                                                      rep['rho_nodes_reached'][str(Rp)]['rules'].items()))
        save()
        # A1-6: adaptive-LM sensitivity (every step adaptive), first `adaptive_check_cases` cases, at the first mesh
        if cfg.get('adaptive_check_cases'):
            sens = {}
            for nm in (f'tensor_R{Rp}', f"{cfg['converged_rule']}_R{Rp}", f'nodes_R{Rp}'):
                fam = arms[nm]['family']
                ctr = OM.contract_tensor if fam == 'tensor' else (n_contract if fam == 'nodes' else OM.contract_offmesh)
                qa = OM.make_fsc_rule(n, Rp, M, ctr, dt=cfg['dt'], gtol=cfg['gtol'], trust=trust,
                                      adaptive_first=int(round(0.25 / cfg['dt'])))
                dd, rs3, recs_a = [], 0, []
                for j in cases[:cfg['adaptive_check_cases']]:
                    o = qa(U0[j], NU[j], arms[nm]['data'], {})
                    Ca = np.asarray(o[1])[::keep]
                    rsn = np.asarray(o[3])
                    rs3 += int((rsn == 3).sum())
                    recs_a.append(dict(case=j, finite=bool(np.isfinite(np.asarray(o[1])).all()),
                                       reasons={str(k_): int((rsn == k_).sum()) for k_ in range(5)},
                                       iterations=int(np.asarray(o[2]).sum())))
                    dd.append(float(np.asarray(lat_err(jnp.asarray(Ca - COEF[(nm, j)]), G65[:Rp], zero, n0[j]))[1:].max()))
                    del o
                sens[nm.rsplit('_R', 1)[0]] = dict(distance_fixed_vs_adaptive=dd, worst=max(dd), reason3=rs3,
                                                   cases=recs_a)
                del qa
            rep.setdefault('adaptive_sensitivity', {})[str(Rp)] = sens
            log(f'ADAPTIVE R{Rp}: ' + '; '.join(f"{k} {v['worst']:.2e}" for k, v in sens.items()))
            save()
        del arms, nd, base, lean, COEF, INTERNAL

    rep['complete'] = True
    rep['seconds'] = el()
    save()
    print('A1-3D COMPLETE', flush=True)


if __name__ == '__main__':
    main()
