"""burgers-eq-tol-knobs: the two solver-side cost knobs (empirical quadrature node count, LM stopping tolerance) on
the CURRENT frozen Burgers 2D model at the paper's Table-1 settings, ONE allocation per mesh (DESIGN.md).

This file is experiments/burgers2d-speed/b2speed.py @ fb4a9ff7 with exactly these changes (everything else --
cohort, bank, rotation, operators, truth, arms, parity, certificates, A-B-A timing, gates -- is that text):
  C1  quadrature rules may be ANISOTROPIC tensor lattices, `"lattice": [sx, sy]`: nodes (a L/sx, b L/sy),
      a = 1..sx-1, b = 1..sy-1, equal weights (L/sx)(L/sy) -- the same composite rule on a coarser lattice as
      hops.lattice_rule (which is used unchanged for square lattices);
  C2  E1 separable tables for such lattices (`lattice_tables2`) and b2fast.proj_vec / proj_cols generalised to an
      (nx, ny) node array; for nx == ny the emitted operations are the originals (same reshape, same einsums);
  C4  optional `bank_columns` Rb < R: the rotation keeps only the first Rb rotated columns, (G T)[:, :Rb] = G T[:, :Rb]
      (used at 4096^2 so the bank fits an 80 GB A100; every arm reads a prefix R' <= Rb, so no arm changes);
  C3  nothing else: the dense-residual arm is the existing exact-step path run for every step
      (variant exact_steps = 50, E2 analytic Jacobian + E3 dense projection), and the tolerance arms are the
      existing `gtol` field.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import engines as e
import iterative_paths as ip
import arms as A
import ladder as LD
import topfix as TF
import hops as H
import hfast as HF
import xfast as XF
import bkfast as BK            # patches hops.bank_apply for the nested rotated bank
import b2fast as B2

sha_array, sha_file, host, dump = LD.sha_array, LD.sha_file, LD.host, LD.dump


# ---------------------------------------------------------------- C1/C2 ----
def lattice_rule2(L, sx, sy):
    """hops.lattice_rule generalised to sx x sy intervals (sx == sy: hops.lattice_rule itself)."""
    if sx == sy:
        return H.lattice_rule(L, sx)
    assert L % sx == 0 and L % sy == 0
    kx = np.arange(1, sx) * (L // sx)
    ky = np.arange(1, sy) * (L // sy)
    ii, jj = np.meshgrid(kx, ky, indexing='ij')
    ij = np.stack((ii.ravel(), jj.ravel()), 1)
    return ij, np.full(len(ij), float((L // sx) * (L // sy)))


def lattice_tables2(L, sx, sy, kx, ky):
    """b2fast.lattice_tables generalised to an (sx-1) x (sy-1) lattice (a-major, as lattice_rule2)."""
    if sx == sy:
        return B2.lattice_tables(L, sx, kx, ky)
    ux, ix = np.unique(np.asarray(kx), return_inverse=True)
    uy, iy = np.unique(np.asarray(ky), return_inverse=True)
    xa = (np.arange(1, sx) * (L // sx)) / L
    xb = (np.arange(1, sy) * (L // sy)) / L
    return dict(SX=jnp.asarray(np.sin(np.pi * xa[:, None] * ux)), SY=jnp.asarray(np.sin(np.pi * xb[:, None] * uy)),
                ix=jnp.asarray(ix.astype(np.int32)), iy=jnp.asarray(iy.astype(np.int32)),
                coef=jnp.asarray((2. / L) * float((L // sx) * (L // sy))))


def _proj_vec(v, T):
    nx, ny = T['SX'].shape[0], T['SY'].shape[0]
    V = v.reshape(nx, ny)
    Z = T['SX'].T @ (V @ T['SY'])
    return T['coef'] * Z[T['ix'], T['iy']]


def _proj_cols(X, T):
    nx, ny = T['SX'].shape[0], T['SY'].shape[0]
    Xr = X.reshape(nx, ny, -1)
    T1 = jnp.einsum('abr,bv->avr', Xr, T['SY'])
    Z = jnp.einsum('au,avr->uvr', T['SX'], T1)
    return T['coef'] * Z[T['ix'], T['iy']]


B2.proj_vec, B2.proj_cols = _proj_vec, _proj_cols      # module globals used by b2fast.make_linear_query


def lat_spec(rs):
    v = rs['lattice']
    return (int(v[0]), int(v[1])) if isinstance(v, (list, tuple)) else (int(v), int(v))
TIMES = np.array([0., .05, .1, .15, .2, .25])
HF_LM_ORIGINAL = HF.make_fused_lm


def gt(g):
    return f'g{g:g}'.replace('-', 'm').replace('.', 'p')


def rel_per_time(f, truth, n0):
    return [float(np.linalg.norm(a - b)) / n0 for a, b in zip(f, truth)]


def arm_name(s):
    """bankknob.arm_name, then the implementation / compile-mode / cap suffixes."""
    v = s['variant']
    sfx = ''.join(f'_{k}' for k in ([v['solver']] if v.get('solver', 'lu') != 'lu' else []) +
                  [k for k in ('clip', 'lamcarry', 'pred2') if v.get(k)])
    if v.get('exact_steps'):
        sfx += f"_x{v['exact_steps']}"
    head = f"R{s['Rp']}"
    body = f"lin_M{s['M']}" if s['model'] == 'lin' else f"q{s.get('q', 0)}_M{s['M']}"
    base = f"{head}_{body}_{s['rule']}_{gt(s['gtol'])}_fast{sfx}"
    if s.get('cap'):
        base += f"_cap{s['cap']}"
    if s.get('budget'):
        base += f"_budget{s['budget']}"
    base += '__' + s.get('impl', 'parent') + ('_graphs' if s.get('graphs') else '')
    return base


def knob_key(s):
    """The deployment setting an arm realises (implementation and compile mode excluded)."""
    v = s['variant']
    return (s['model'], int(s['Rp']), int(s.get('q', 0)), int(s['M']), s['rule'], float(s['gtol']),
            v.get('solver', 'lu'), bool(v.get('clip')), bool(v.get('lamcarry')), bool(v.get('pred2')),
            int(v.get('exact_steps', 0)), s.get('cap'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--inputs', required=True, help='experiments/b-panel/inputs (directions, rule files)')
    p.add_argument('--rotation', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    inputs = Path(a.inputs)
    if not cfg.get('allow_cpu_smoke'):
        assert jax.default_backend() == 'gpu', jax.default_backend()
    assert jax.config.jax_enable_x64 and os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    begin = time.perf_counter()
    el = lambda: round(time.perf_counter() - begin, 1)
    try:
        smi = subprocess.check_output(['nvidia-smi', '-L'], text=True).strip().splitlines()
    except Exception:                               # noqa: BLE001
        smi = []
    graphs_opts = None if cfg.get('allow_cpu_smoke') and jax.default_backend() != 'gpu' else B2.GRAPHS

    ck = pickle.load(open(a.checkpoint, 'rb'))
    params = jax.tree_util.tree_map(jnp.asarray, host(ck['params']))
    Zold = np.asarray(ck['Z_tr'])
    K = int(Zold.shape[1])
    R = int(np.asarray(ck['params']['h_lin']).shape[1])
    L, dt = int(cfg['intervals']), cfg['dt']
    st = cfg['strict']
    BAR = cfg['rho_bar']
    edges = [0] + list(cfg['ladder'])
    Rb = int(cfg.get('bank_columns') or R)       # C4: keep only the first Rb rotated columns (memory at 4096^2)
    assert edges[-1] == Rb <= R and edges == sorted(edges)
    rep = dict(config=cfg, commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
               host=os.uname().nodename, backend=jax.default_backend(), gpu=jax.devices()[0].device_kind,
               nvidia_smi=smi, x64=True, matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'],
               jax_version=jax.__version__, graphs_compiler_options=graphs_opts,
               checkpoint_sha256=sha_file(a.checkpoint), K=K, R=R, intervals=L, dt=dt, weights_frozen=True,
               output_times=TIMES.tolist(),
               timing_contract=('gpu_seconds: supplied dense initial field resident on the GPU to six dense GPU '
                                'output fields, block_until_ready on both sides; host_seconds: the same invocation '
                                'plus the host upload of the input and the host copy of the six outputs; identical '
                                'for every subject'),
               rules=[], models={}, arm_setup=[], quick=[], parity=[], certificates={}, invocations=[],
               phases={}, gates={}, complete=False)

    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (float, np.floating)):
            return float(x) if np.isfinite(x) else None
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, np.bool_):
            return bool(x)
        return x

    save = lambda: dump(out / 'result.json', clean(rep))
    save()

    # ------------------------------------------------------------------ cohort --
    if cfg.get('eval_draws'):
        physical = np.concatenate([e.params_draw(int(sd), int(n_)) for sd, n_ in cfg['eval_draws']])
    else:
        physical = np.concatenate((e.params_draw(cfg['eval_seed'], cfg['eval_cases']),
                                   e.params_draw(cfg['eval_fresh_seed'], cfg['eval_fresh_cases'])))
    rep['cohort_name'] = cfg['cohort_name']
    rep['physical_sha256'] = sha_array(physical)
    want = cfg.get('expected_physical_sha256')
    rep['gates']['evaluation_cohort_hash'] = dict(expected=want, got=rep['physical_sha256'],
                                                  passed=(None if want is None else want == rep['physical_sha256']))
    assert want is None or want == rep['physical_sha256'] or cfg.get('local_smoke_waives_cohort_hash'), 'cohort differs'
    if cfg.get('case_subset') is not None:
        physical = physical[[int(i) for i in cfg['case_subset']]]
    ncase = len(physical)
    rep['physical_cases'] = physical.tolist()
    train_physical = e.params_draw(cfg['train_seed'], cfg['train_trajectories'])
    assert not any(np.allclose(t, s) for t in train_physical for s in physical), 'training/eval overlap'
    pc = cfg['population']
    pop_physical = e.params_draw(*pc['source_draw'])
    for other, nm in ((physical, 'evaluation cohort'), (e.params_draw(20260916, 64), 'hold64'),
                      (e.params_draw(7090702, 4), 'dev4'), (e.params_draw(911702, 2), 'dev2'),
                      (train_physical, 'train_physical')):
        assert not any(np.any(np.all(np.isclose(o, pop_physical), axis=1)) for o in other), f'population/{nm} overlap'
    draws = [list(d) for d in pc['cert_draws']] + ([list(pc['confirm_draw'])] if pc.get('confirm_draw') else [])
    ncert = len(pc['cert_draws'])
    flat = [i for d in draws for i in d]
    assert len(flat) == len(set(flat)) and max(flat) < len(pop_physical)
    rep['population_source'] = dict(source_draw=pc['source_draw'], sha256=sha_array(pop_physical), draws=draws,
                                    certification_draws=ncert,
                                    confirmation_draw_index=(ncert if pc.get('confirm_draw') else None))
    inputs_u = [e.initial(L, ph) for ph in physical]
    n0 = [float(np.linalg.norm(u)) for u in inputs_u]
    sub = max(1, L // cfg.get('restrict_to', 256))

    # ------------------------------------------------------ rotation, bank ----
    rz = np.load(a.rotation)
    T, Lrot = np.asarray(rz['T']), np.asarray(rz['L'])
    rep['rotation'] = dict(file=str(a.rotation), file_sha256=sha_file(a.rotation), T_sha256=sha_array(T),
                           L_sha256=sha_array(Lrot), expected_file_sha256=cfg['rotation_sha256'],
                           L_times_T_identity_deviation=float(np.linalg.norm(Lrot @ T - np.eye(R))))
    assert rep['rotation']['file_sha256'] == cfg['rotation_sha256'], 'rotation file differs from the committed one'
    Tj = jnp.asarray(T)
    base = A.CoordBank(params, K, R)
    t0 = time.perf_counter()
    n = (L - 1) ** 2
    nrb = int(cfg.get('bank_blocks') or np.ceil(n * R / H.MAX_GEMM_ELEMENTS))
    x = np.arange(1, L) / L
    redges = np.linspace(0, L - 1, nrb + 1).astype(int)
    rotate = jax.jit(lambda g, t: tuple((g @ t)[:, a_:b_] for a_, b_ in zip(edges[:-1], edges[1:])))
    Grot = []
    for i0, i1 in zip(redges[:-1], redges[1:]):
        xy = np.stack(np.meshgrid(x[i0:i1], x, indexing='ij'), -1).reshape(-1, 2)
        g = jax.block_until_ready(base.at(xy, chunk=8192))
        Grot.append(jax.block_until_ready(rotate(g, Tj[:, :Rb])))
        del g
    Grot = tuple(Grot)
    rep['phases']['bank'] = dict(rows=n, row_blocks=nrb, column_edges=edges, seconds=time.perf_counter() - t0,
                                 rotated_bytes=int(sum(c.nbytes for rb in Grot for c in rb)))
    print('BANK', n, 'rows', nrb, 'row blocks', el(), flush=True)
    Gf_cache = {}

    def Gfull(Rp):
        """E2: the bank prefix as one (L-1, L-1, R') array (only built for exact-step eng arms)."""
        if Rp not in Gf_cache:
            Gf_cache[Rp] = jax.block_until_ready(BK.bank_apply(BK.prefix(Grot, edges, Rp), jnp.eye(Rp)).reshape(
                L - 1, L - 1, Rp))
        return Gf_cache[Rp]

    # ---------------------------------------------------- full-order solvers ----
    foms = {}

    def fom_for(fs, graphs=False):
        key = (fs.get('mesh', L), fs['dt'], fs.get('impl', 'audited'), bool(graphs))
        if key not in foms:
            if key[0] == L and key[2] == 'audited':
                fn, pre = ip.make_fom(L, fs['dt'], 'fft')
            else:
                fn, pre = e.make_fom(key[0], fs['dt'], target=L)[0], None
            if graphs:
                fn = jax.jit(fn.__wrapped__, compiler_options=graphs_opts)
            foms[key] = (fn, pre)
        return foms[key]

    def run_fom(fs, u, nu, graphs=False):
        fn, pre = fom_for(fs, graphs)
        if pre is None:
            return fn(u, nu, fs['ntol'], fs['ltol'])
        return fn(u, nu, fs['ntol'], fs['ltol'], *pre)

    tight = next(fs for fs in cfg['fom_settings'] if fs['name'] == cfg['same_grid_reference'])
    truth = {}
    for c in range(ncase):
        t0 = time.perf_counter()
        v = run_fom(tight, jnp.asarray(inputs_u[c]), float(physical[c, 4]))
        jax.block_until_ready(v)
        f = np.asarray(v[0])
        rn = np.asarray(v[2])
        assert np.isfinite(f).all() and rn.max() <= tight['ntol'] * (1 + 1e-9), ('truth not converged', c)
        truth[c] = f
        rep['phases'].setdefault('truth', []).append(dict(case=c, seconds=time.perf_counter() - t0,
                                                          max_relative_residual=float(rn.max())))
    print('TRUTH done', el(), flush=True)
    save()

    def score(f, c):
        sg = rel_per_time(f, truth[c], n0[c])
        cur = [float(np.linalg.norm(a_ - b_) / max(np.linalg.norm(b_), 1e-300)) for a_, b_ in zip(f, truth[c])]
        return dict(same_grid_per_time=sg, same_grid_all=float(max(sg)), same_grid_evolved=float(max(sg[1:])),
                    t0_compression=float(sg[0]), current_relative_per_time=cur,
                    current_relative_evolved=float(max(cur[1:])))

    # -------------------------------------------------------- operators ----
    dfile = inputs / cfg['directions_file']
    Cfull = jnp.asarray(np.ascontiguousarray(np.load(dfile)['C']))
    rep['directions'] = dict(file=cfg['directions_file'], sha256=sha_file(dfile))
    assert cfg['directions_sha256'] == rep['directions']['sha256']
    ops_M = {}

    def operators(M):
        if M not in ops_M:
            kx, ky, lam = H.modes_lean(L, M)
            sx, sy = (jnp.asarray(t_) for t_ in H.sine_tables(L, kx, ky))
            o = dict(kx=kx, ky=ky, lam=jnp.asarray(lam), sx=sx, sy=sy,
                     Arot=BK.project_nested(Grot, edges, Rb, sx, sy, L), sepd=B2.dense_tables(L, kx, ky))
            ops_M[M] = jax.block_until_ready(o)
        return ops_M[M]

    Mg = max(s_['M'] for s_ in cfg['arms'])
    og = operators(Mg)
    if L <= cfg.get('phi_gate_max_mesh', 512):
        Phig, lamg, _ = e.modes(L, Mg)
        Gflat = BK.bank_apply(Grot, jnp.eye(Rb))
        Aref = jnp.asarray(Phig).T @ Gflat
        g0 = dict(M=Mg, lam_identical=bool(np.array_equal(lamg, np.asarray(og['lam']))),
                  A_relative=float(jnp.linalg.norm(og['Arot'] - Aref) / jnp.linalg.norm(Aref)))
        # E3 gate: the distinct-wave-number dense projection equals Phi^T on random fields
        vr = jnp.asarray(np.random.default_rng(1).normal(size=(n, 3)))
        pd = B2.proj_cols(vr, og['sepd'])
        g0['sep_dense_relative'] = float(jnp.linalg.norm(pd - jnp.asarray(Phig).T @ vr) / jnp.linalg.norm(pd))
        g0['passed'] = bool(g0['lam_identical'] and g0['A_relative'] <= 1e-12 and g0['sep_dense_relative'] <= 1e-12)
        rep['gates']['phi_free_operator_parity'] = g0
        assert g0['passed'], g0
        del Phig, Gflat, Aref
    save()

    # ------------------------------------------------------------ rules ----
    rule_nodes, G5 = {}, {}

    def rule(name):
        if name in rule_nodes:
            return rule_nodes[name]
        rs = cfg['rules'][name]
        if 'lattice' in rs:
            sx_, sy_ = lat_spec(rs)
            ij, w = lattice_rule2(L, sx_, sy_)
            src = dict(kind='lattice', s=[sx_, sy_])
        else:
            fp = inputs / rs['file']
            assert sha_file(fp) == rs['sha256'], rs['file']
            z = np.load(fp)
            ij = H.transfer_nodes(z['nodes'].astype(int), rs['mesh'], L)
            w = np.asarray(z['weights'], float) * (L / rs['mesh']) ** 2
            src = dict(kind='transfer', file=rs['file'], sha256=rs['sha256'], source_mesh=rs['mesh'])
        keep = w > 0
        ij, w = ij[keep], w[keep]
        g5 = base.stencil(ij, L)
        G5[name] = jnp.einsum('msr,rp->msp', g5, Tj)
        np.savez_compressed(out / f'rule_L{L}_{name}.npz', ij=ij, weights=w)
        rule_nodes[name] = (ij, w)
        rep['rules'].append(dict(rule=name, m=int(len(ij)), source=src, control=bool(rs.get('control')),
                                 note=rs.get('note')))
        return rule_nodes[name]

    Pq_cache, sepq_cache = {}, {}

    def Pq(name, M):
        if (name, M) not in Pq_cache:
            ij, w = rule(name)
            o = operators(M)
            Pq_cache[(name, M)] = jnp.asarray(H.phi_rows(L, o['kx'], o['ky'], ij) * w[:, None])
        return Pq_cache[(name, M)]

    def sepq(name, M):
        """E1 tables; gated against the dense Pq on random columns (<= 1e-12)."""
        if (name, M) not in sepq_cache:
            rs = cfg['rules'][name]
            assert 'lattice' in rs, 'E1 applies to lattice rules only'
            o = operators(M)
            tb = lattice_tables2(L, *lat_spec(rs), o['kx'], o['ky'])
            P = Pq(name, M)
            X = jnp.asarray(np.random.default_rng(2).normal(size=(P.shape[0], 4)))
            d = float(jnp.linalg.norm(B2.proj_cols(X, tb) - P.T @ X) / jnp.linalg.norm(P.T @ X))
            rep['gates'].setdefault('sep_lattice_projection', []).append(dict(rule=name, M=M, relative=d,
                                                                             passed=bool(d <= 1e-12)))
            assert d <= 1e-12, (name, M, d)
            sepq_cache[(name, M)] = tb
        return sepq_cache[(name, M)]

    # ----------------------------------------------------------- models ----
    trust0 = .01 * float(np.max(np.linalg.norm(Zold - Zold.mean(0), axis=1)))
    Zsub = np.asarray(Zold[::max(1, len(Zold) // cfg['decoder_code_subsample'])])
    hv0 = jax.jit(jax.vmap(lambda z: A.sc.head(params, z)))
    Hall = np.concatenate([np.asarray(hv0(jnp.asarray(Zold[s:s + 8192]))) for s in range(0, len(Zold), 8192)])
    Hsub = np.asarray(hv0(jnp.asarray(Zsub)))
    models = {}

    def model(kind, Rp):
        key = (kind, Rp)
        if key in models:
            return models[key]
        if kind == 'trunc':
            m_ = dict(params=BK.fold_head(params, Lrot, Rp), C=jnp.asarray(Lrot[:Rp]) @ Cfull,
                      bank=BK.RotBank(base, T, Rp), G=BK.prefix(Grot, edges, Rp), trust=trust0, codes=Zsub,
                      A=lambda M, Rp=Rp: operators(M)['Arot'][:, :Rp],
                      G5=lambda nm, Rp=Rp: G5[nm][:, :, :Rp])
        else:
            codes_all = Hall @ Lrot[:Rp].T
            trust = .01 * float(np.max(np.linalg.norm(codes_all - codes_all.mean(0), axis=1)))
            m_ = dict(params=None, C=None, bank=BK.RotBank(base, T, Rp), G=BK.prefix(Grot, edges, Rp), trust=trust,
                      codes=Hsub @ Lrot[:Rp].T, A=lambda M, Rp=Rp: operators(M)['Arot'][:, :Rp],
                      G5=lambda nm, Rp=Rp: G5[nm][:, :, :Rp])
        m_['colds'] = {}
        rep['models'][f'{kind}{Rp}'] = dict(kind=kind, R_prime=Rp, trust=m_['trust'])
        models[key] = m_
        return m_

    def cold_head(m_, kind, q):
        if q not in m_['colds']:
            if kind == 'lin':
                head, codes = (lambda w_: w_), m_['codes']
            else:
                head = TF.corrected_head(m_['params'], m_['C'][:, :q], K)
                codes = np.concatenate((m_['codes'], np.zeros((len(m_['codes']), q))), 1)
            m_['colds'][q] = (A.build_cold(m_['bank'], head, codes, cfg['cold_axis_points'])[0], jax.jit(jax.vmap(head)))
        return m_['colds'][q]

    # ------------------------------------------------------------- arms ----
    built, subjects, kept = {}, [], {}

    def chunks_for(d):
        return next(k for k in range(1, d + 1) if d % k == 0 and d // k <= cfg['dense_tangent_group'])

    def build_arm(s):
        s = dict(s)
        s.setdefault('q', 0)
        s.setdefault('impl', 'parent')
        name = arm_name(s)
        v = s['variant']
        j = int(v.get('exact_steps', 0))
        kind, Rp, q, M = s['model'], int(s['Rp']), int(s['q']), int(s['M'])
        eng = s['impl'] == 'eng'
        opts = graphs_opts if s.get('graphs') else None
        m_ = model(kind, Rp)
        o = operators(M)
        cold, coef = cold_head(m_, kind, q)
        rule(s['rule'])
        lattice = 'lattice' in cfg['rules'][s['rule']]
        data = dict(A=m_['A'](M), lam=o['lam'], G=m_['G'], G5=m_['G5'](s['rule']), Pq=Pq(s['rule'], M),
                    sx=o['sx'], sy=o['sy'])
        common = dict(step_budget=int(s.get('budget') or st['step_budget']), gtol=s['gtol'], ridge=cfg['inner_damping'],
                      solver=v.get('solver', 'lu'), clip=bool(v.get('clip')), lam_carry=bool(v.get('lamcarry')),
                      predictor='quad' if v.get('pred2') else 'lin')
        engflags = {}
        if kind == 'lin':
            d = Rp
            if eng:
                engflags = dict(sep_eq=lattice, ajac_dense=bool(j), sep_dense=True)
                if lattice:
                    data['sepq'] = sepq(s['rule'], M)
                data['sepd'] = o['sepd']
                if j:
                    data['Gf'] = Gfull(Rp)
                fq = B2.make_linear_query(Rp, L, dt, m_['trust'], exact_steps=j, tangent_chunks=chunks_for(d),
                                          lm_cap=s.get('cap'), compiler_options=opts, **engflags, **common)
            elif s['impl'] == 'general':
                assert not s.get('cap') and not s.get('graphs') and not j
                fq = B2.make_linear_query_general(Rp, L, dt, m_['trust'], step_budget=st['step_budget'],
                                                  gtol=s['gtol'], ridge=cfg['inner_damping'])
                engflags = dict(path='general: varpro.make_block_lm, jacfwd, LU, reject, damping reset, 2-way guard')
            else:
                assert not s.get('cap') and not s.get('graphs'), 'parent arms run the parent text, default compile'
                fq = BK.make_linear_query(Rp, L, dt, m_['trust'], exact_steps=j, tangent_chunks=chunks_for(d),
                                          **common)
            tab = {}
        else:
            d = K + q
            C = m_['C'][:, :q]
            tab = HF.build_tables(m_['params'], C, K, data, cold)
            if eng:
                # head arms: the parent query text with b2fast's LM (identical while-loop text; `cap` knob) and the
                # compile mode. E1 is not applied to head arms (the fast head rule is a file rule, not a lattice).
                cap = s.get('cap')
                HF.make_fused_lm = (lambda *a_, cap=cap, **k_: B2.make_fused_lm(*a_, cap=cap, **k_))
                try:
                    fq0 = XF.make_query(m_['params'], C, K, q, L, dt, m_['trust'], 'eq', exact_steps=j,
                                        ic_budget=st['ic_budget'], ic_gtol=cfg['ic_gtol'],
                                        tangent_chunks=chunks_for(d), **common)
                finally:
                    HF.make_fused_lm = HF_LM_ORIGINAL
                fq = jax.jit(fq0.__wrapped__, compiler_options=opts) if opts else fq0
                engflags = dict(lm='b2fast', cap=cap)
            else:
                assert not s.get('cap') and not s.get('graphs')
                fq = XF.make_query(m_['params'], C, K, q, L, dt, m_['trust'], 'eq', exact_steps=j,
                                   ic_budget=st['ic_budget'], ic_gtol=cfg['ic_gtol'], tangent_chunks=chunks_for(d),
                                   **common)
        built[name] = dict(name=name, kind='rom', family='rom', spec=s, data=data, cold=cold, coef=coef, tab=tab,
                           query=(lambda u, nu, d_, c_, _t=tab, _f=fq: _f(u, nu, d_, c_, _t)),
                           m=int(data['G5'].shape[0]), unknowns=d, exact_steps=j, knob=knob_key(s))
        rep['arm_setup'].append(dict(arm=name, family='rom', model=kind, R_prime=Rp, q=q, M=M, rule=s['rule'],
                                     m=int(data['G5'].shape[0]), unknowns=d, gtol=s['gtol'], variant=v,
                                     exact_steps=j, cap=s.get('cap'), impl=s['impl'], graphs=bool(s.get('graphs')),
                                     engineering=engflags, trust=m_['trust'], role=s.get('role'),
                                     arm_family=s.get('family'), candidate=bool(s.get('candidate', True)),
                                     control=bool(cfg['rules'][s['rule']].get('control')),
                                     timed=not s.get('untimed', False), certify=bool(s.get('certify', True)),
                                     knob=list(knob_key(s))))
        return name

    def invoke(b, u, c):
        nu = float(physical[c, 4])
        if b['kind'] == 'fom':
            return run_fom(b['setting'], u, nu, b['graphs'])
        return b['query'](u, nu, b['data'], b['cold'])

    def rom_row(v):
        reasons = np.asarray(v[3]).tolist()
        it = np.asarray(v[1])
        gj = np.asarray(v[12], float)
        return dict(stop_reasons=reasons, iterations=it.tolist(), total_iterations=int(it.sum()),
                    max_iterations=int(it.max()), ic_iterations=int(v[5]), ic_reason=int(v[6]),
                    budget_exits=int(sum(r == 0 for r in reasons)), rejected_exits=int(sum(r == 3 for r in reasons)),
                    tiny_step_exits=int(sum(r == 2 for r in reasons)),
                    worst_joint_stationarity=float(np.max(gj)) if np.isfinite(gj).all() else None,
                    stalled_exits=int(sum(r in (0, 2, 3) for r in reasons)),
                    damping_retries_total=int(np.sum(np.asarray(v[15]))))

    def fom_row(b, v):
        fs = b['setting']
        rn = np.asarray(v[2])
        okstep = np.isfinite(rn) & (rn <= fs['ntol'] * (1 + 1e-9))
        return dict(dt=fs['dt'], ntol=fs['ntol'], ltol=fs['ltol'], mesh=fs.get('mesh', L), impl=fs.get('impl', 'audited'),
                    graphs=b['graphs'], newton_iterations_total=int(np.sum(np.asarray(v[1]))), steps=int(len(rn)),
                    stalled_steps=int(np.sum(~okstep)), nonlinear_converged=bool(okstep.all()),
                    max_relative_residual=float(np.nanmax(rn)))

    keep_fields = set(n_ for pr in cfg.get('parity_pairs', []) for n_ in pr) | (
        {fs['name'] + sfx for fs in cfg['fom_settings'] for sfx in ('', '__graphs')}
        if cfg.get('fom_mode_parity', True) and cfg.get('fom_both_modes', True) else set())

    def run_quick(names):
        for name in names:
            b = built[name]
            t_arm = time.perf_counter()
            secs = []
            for c in range(ncase):
                t0 = time.perf_counter()
                v = invoke(b, jnp.asarray(inputs_u[c]), c)
                jax.block_until_ready(v)
                secs.append(time.perf_counter() - t0)
                f = np.asarray(v[0])
                assert np.isfinite(f).all(), name
                vh = (None,) + tuple(host(v[1:]))
                row = dict(name=name, case=c, family=b['family'], field_sha256=sha_array(f),
                           field_sha256_sub=sha_array(f[:, ::16, ::16]), seconds=secs[-1], **score(f, c))
                row.update(rom_row(vh) if b['kind'] == 'rom' else fom_row(b, vh))
                rep['quick'].append(row)
                if name in keep_fields:
                    # ROM: fields, (iterations, residual norms, reasons), all 51 internal states, rejected steps
                    kept[(name, c)] = (f, vh[1:4], np.asarray(vh[7]) if b['kind'] == 'rom' else None,
                                       np.asarray(vh[15]) if b['kind'] == 'rom' else None)
                if cfg.get('save_restricted', True) and not (cfg.get('save_restricted_skip_graphs_fom')
                                                             and b['kind'] == 'fom' and b['graphs']):
                    extra = dict(internal_latents=np.asarray(v[7])) if b['kind'] == 'rom' else {}
                    np.savez_compressed(out / f'restricted_{name}_case{c}.npz', fields=f[:, ::sub, ::sub], **extra)
                if c in cfg['audit_cases'] and name in cfg['audit_arms']:
                    np.save(out / f'full_{name}_case{c}.npy', f)
            b['quick_seconds'] = float(np.median(secs[1:] if len(secs) > 1 else secs))
            subjects.append(name)
            rows = [x_ for x_ in rep['quick'] if x_['name'] == name]
            print('QUICK', name, 'evolved%', round(100 * max(x_['same_grid_evolved'] for x_ in rows), 4),
                  'median%', round(100 * float(np.median([x_['same_grid_evolved'] for x_ in rows])), 4),
                  'stalled', sum(x_.get('stalled_exits', x_.get('stalled_steps', 0)) for x_ in rows),
                  'iters', [x_.get('total_iterations') for x_ in rows], 'sec', round(b['quick_seconds'], 4),
                  round(time.perf_counter() - t_arm, 1), el(), flush=True)
            save()

    fom_names = []
    for fs in cfg['fom_settings']:
        for gr in ([False, True] if cfg.get('fom_both_modes', True) else [False]):
            nm = fs['name'] + ('__graphs' if gr else '')
            built[nm] = dict(name=nm, kind='fom', family='fom', setting=fs, graphs=gr)
            fom_names.append(nm)
    run_quick(fom_names)
    for s in cfg['arms']:
        run_quick([build_arm(s)])
    rep['phases']['quick_seconds'] = el()

    # ------------------------------------------------ parity (engineering) ----
    PBAR = cfg['parity_bar']
    for fa, fb in cfg.get('parity_pairs', []):
        per = []
        for c in range(ncase):
            (xa, ia, sa, ja), (xb, ib, sb, jb) = kept[(fa, c)], kept[(fb, c)]
            per.append(dict(case=c, relative=float(np.linalg.norm(xa - xb) / np.linalg.norm(xb)),
                            states_relative=float(np.max(np.linalg.norm(sa - sb, axis=1) /
                                                         np.maximum(np.linalg.norm(sb, axis=1), 1e-300))),
                            iterations_identical=bool(np.array_equal(ia[0], ib[0])),
                            reasons_identical=bool(np.array_equal(ia[2], ib[2])),
                            rejections_identical=bool(np.array_equal(ja, jb)),
                            iterations_eng=int(np.sum(ia[0])), iterations_parent=int(np.sum(ib[0]))))
        worst = max(max(x_['relative'], x_['states_relative']) for x_ in per)
        ints = all(x_['iterations_identical'] and x_['reasons_identical'] and x_['rejections_identical'] for x_ in per)
        rep['parity'].append(dict(engineered=fa, parent=fb, worst_relative=worst, integers_identical=ints, bar=PBAR,
                                  passed=bool(worst <= PBAR and ints), cases=per,
                                  scope='output fields and all 51 internal states; per-step iterations, exit reasons '
                                        'and rejected-step counts'))
        print('PARITY', fa, 'vs', fb, f'{worst:.3e}', 'integers', ints, flush=True)
    # FOM compile-mode identity by field SHA (every setting, every case; cheap, no fields held)
    qs = {(x_['name'], x_['case']): x_['field_sha256'] for x_ in rep['quick']}
    rep['fom_mode_sha_identical'] = {fs['name']: all(qs.get((fs['name'] + '__graphs', c)) == qs.get((fs['name'], c))
                                                     for c in range(ncase))
                                     for fs in cfg['fom_settings'] if (fs['name'] + '__graphs', 0) in qs}
    # FOM compile-mode parity: fields and per-step Newton iteration vectors, every setting
    rep['fom_mode_parity'] = []
    for fs in cfg['fom_settings']:
        a_, b_ = fs['name'] + '__graphs', fs['name']
        if (a_, 0) not in kept:
            continue
        per = [dict(case=c, relative=float(np.linalg.norm(kept[(a_, c)][0] - kept[(b_, c)][0]) /
                                           np.linalg.norm(kept[(b_, c)][0])),
                    newton_identical=bool(np.array_equal(kept[(a_, c)][1][0], kept[(b_, c)][1][0])))
               for c in range(ncase)]
        w_ = max(x_['relative'] for x_ in per)
        ok_ = all(x_['newton_identical'] for x_ in per)
        rep['fom_mode_parity'].append(dict(setting=fs['name'], worst_relative=w_, newton_identical=ok_,
                                           passed=bool(w_ <= PBAR and ok_), cases=per))
        print('FOM-PARITY', fs['name'], f'{w_:.3e}', ok_, flush=True)
    kept.clear()
    save()

    # ---------------------------------------------------------- certify ----
    steps_total = int(round(.25 / dt))
    cert_done = {}
    for name in [n_ for n_ in subjects if built[n_]['kind'] == 'rom' and built[n_]['spec'].get('certify', True)
                 and not cfg.get('skip_certificates')]:
        b = built[name]
        t0 = time.perf_counter()
        coefs, metas, stall = [], [], 0
        for d_, dr in enumerate(draws):
            for i in dr:
                ph = pop_physical[i]
                v = b['query'](jnp.asarray(e.initial(L, ph)), float(ph[4]), b['data'], b['cold'])
                wv = np.asarray(v[7])
                assert len(wv) == steps_total + 1 and np.isfinite(wv).all(), (name, i)
                stall += int(sum(r in (0, 2, 3) for r in np.asarray(v[3]).tolist()))
                coefs.append(np.asarray(b['coef'](jnp.asarray(wv))))
                metas.append(np.stack((np.full(len(wv), d_), np.full(len(wv), i), np.arange(len(wv))), 1))
        co, meta = np.concatenate(coefs), np.concatenate(metas)
        o = operators(b['spec']['M'])
        tg = H.dense_targets(b['data']['G'], co, o['sx'], o['sy'], L, chunk=cfg['target_chunk'])
        r = H.rho(b['data']['Pq'], H.sampled_advection(b['data']['G5'], co, L), tg)
        np.savez_compressed(out / f'deployed_{name}.npz', coefficients=co, meta=meta, rho=r)
        j = b['exact_steps']
        res_ = {}
        for label, kmin in (('primary', max(j, 1)), ('k_ge_j_plus_1', j + 1)):
            per = []
            for d_ in range(len(draws)):
                sel = (meta[:, 0] == d_) & (meta[:, 2] >= kmin)
                per.append(dict(draw=d_, states=int(sel.sum()), rho_max=float(r[sel].max()),
                                rho_p95=float(np.quantile(r[sel], .95)), passed=bool(r[sel].max() <= BAR)))
            npass = sum(x_['passed'] for x_ in per[:ncert])
            conf = per[ncert]['passed'] if len(per) > ncert else None
            status = ('confirmed' if npass == ncert and conf is not False else
                      'fails' if npass == 0 else f'not confirmed ({npass}/{ncert} draws, confirmation {conf})')
            res_[label] = dict(k_threshold=kmin, per_draw=per, draws_passed=npass, draws=ncert,
                               confirmation_pass=conf, status=status)
        rep['certificates'][name] = dict(**res_['primary'], sensitivity_k_ge_j_plus_1=res_['k_ge_j_plus_1'],
                                         bar=BAR, stalled_steps=stall,
                                         rho_max_all_k=float(r[meta[:, 0] < ncert].max()),
                                         seconds=time.perf_counter() - t0)
        cert_done[name] = True
        pr = res_['primary']['per_draw']
        print('CERT', name, res_['primary']['status'], 'rho_max', f"{max(x_['rho_max'] for x_ in pr[:ncert]):.4f}",
              'conf', f"{pr[ncert]['rho_max']:.4f}" if len(pr) > ncert else None, round(time.perf_counter() - t0, 1),
              el(), flush=True)
        save()

    # ------------------------------------------------------ timed: A-B-A ----
    timed_rom = [n_ for n_ in subjects if built[n_]['kind'] == 'rom' and not built[n_]['spec'].get('untimed')]
    timed_fom = [n_ for n_ in fom_names if built[n_]['setting']['name'] not in cfg.get('untimed_fom', [])]
    for n_ in timed_rom + timed_fom:
        jax.block_until_ready(invoke(built[n_], jnp.asarray(inputs_u[0]), 0))
    order_rng = np.random.default_rng(cfg['order_seed'])
    quick_sha = {(x_['name'], x_['case']): x_['field_sha256'] for x_ in rep['quick']}
    quick_sub = {(x_['name'], x_['case']): x_['field_sha256_sub'] for x_ in rep['quick']}
    sha_every = int(cfg.get('timed_full_sha_every', 1))
    seqc = [0]

    def phase_break(label):
        jax.block_until_ready(jnp.zeros(()))
        time.sleep(cfg['phase_cooldown_seconds'])
        e.burn(cfg['phase_dummy_seconds'])
        rep['phases'].setdefault('breaks', []).append(dict(before=label, at_seconds=el()))

    def run_phase(label, arms_, reps):
        prev = None
        for r_ in range(reps):
            for c in range(ncase):
                for i in order_rng.permutation(len(arms_)):
                    name = arms_[int(i)]
                    b = built[name]
                    e.burn(cfg['burn_seconds'])
                    ht = time.perf_counter()
                    u = jax.device_put(np.array(inputs_u[c], copy=True))
                    jax.block_until_ready(u)
                    g0_ = time.perf_counter()
                    v = invoke(b, u, c)
                    jax.block_until_ready(v)
                    gs = time.perf_counter() - g0_
                    f = np.asarray(v[0])
                    hs = time.perf_counter() - ht
                    same_sub = sha_array(f[:, ::16, ::16]) == quick_sub[(name, c)]
                    full = (seqc[0] % sha_every == 0)
                    same = (sha_array(f) == quick_sha[(name, c)]) if full else same_sub
                    row = dict(name=name, case=c, rep=r_, phase=label, seq=seqc[0], previous=prev,
                               family=b['family'], gpu_seconds=gs, host_seconds=hs,
                               identical_to_quick=bool(same and same_sub), full_sha_checked=full)
                    if b['kind'] == 'rom':
                        row['total_iterations'] = int(np.sum(np.asarray(v[1])))
                    rep['invocations'].append(row)
                    prev = name
                    seqc[0] += 1
            print('TIMED', label, 'rep', r_, el(), flush=True)
            save()

    phase_break('romA1')
    run_phase('romA1', timed_rom, cfg['rom_reps'])
    phase_break('fomB')
    run_phase('fomB', timed_fom, cfg['fom_reps'])
    phase_break('romA2')
    run_phase('romA2', timed_rom, cfg['rom_reps'])

    inv = rep['invocations']
    lim = cfg['gate_limit']
    med = lambda nm, ph: float(np.median([x_['gpu_seconds'] for x_ in inv if x_['name'] == nm and x_['phase'] == ph]))
    drift = [dict(name=nm, romA1_median=med(nm, 'romA1'), romA2_median=med(nm, 'romA2')) for nm in timed_rom]
    for d_ in drift:
        d_['ratio'] = d_['romA2_median'] / d_['romA1_median']
    rep['gates']['drift'] = dict(rows=drift, limit=lim, passed=bool(all(1 / lim <= d_['ratio'] <= lim for d_ in drift)))
    rows = []
    for label, subs in (('romA1', timed_rom), ('romA2', timed_rom), ('fomB', timed_fom)):
        ivp = [x_ for x_ in inv if x_['phase'] == label]
        medians = {nm: med(nm, label) for nm in subs}
        cm = {}
        for x_ in ivp:
            cm.setdefault((x_['name'], x_['case']), []).append(x_['gpu_seconds'])
        cm = {k_: float(np.median(v_)) for k_, v_ in cm.items()}
        cut = np.quantile(list(medians.values()), [1 / 3, 2 / 3])
        for nm in subs:
            mine = [x_ for x_ in ivp if x_['name'] == nm and x_['previous'] is not None]
            lo = [x_['gpu_seconds'] / cm[(nm, x_['case'])] for x_ in mine if medians[x_['previous']] <= cut[0]]
            hi = [x_['gpu_seconds'] / cm[(nm, x_['case'])] for x_ in mine if medians[x_['previous']] >= cut[1]]
            ev = len(lo) >= 3 and len(hi) >= 3
            rows.append(dict(name=nm, phase=label, n_short=len(lo), n_long=len(hi), evaluable=ev,
                             ratio=float(np.mean(hi) / np.mean(lo)) if ev else None))
    rep['gates']['neighbour'] = dict(rows=rows, limit=lim, passed=bool(all(r_['ratio'] <= lim for r_ in rows if r_['evaluable'])),
                                     not_evaluable=[(r_['name'], r_['phase']) for r_ in rows if not r_['evaluable']],
                                     rule='per subject and phase, CASE-CONTROLLED: every invocation divided by its '
                                          '(subject, case, phase) median; mean after a long predecessor (phase median '
                                          'in the top third) / mean after a short one (bottom third)')
    print('DRIFT', rep['gates']['drift']['passed'], min(d_['ratio'] for d_ in drift), max(d_['ratio'] for d_ in drift),
          'NEIGHBOUR', rep['gates']['neighbour']['passed'], max([r_['ratio'] for r_ in rows if r_['evaluable']] or [None]),
          flush=True)
    cover = {(n_, c) for n_ in timed_rom + timed_fom for c in range(ncase)}
    counts = {k: sum(1 for x_ in inv if (x_['name'], x_['case']) == k) for k in cover}
    rep['gates']['repetition_output_identical'] = dict(passed=bool(inv) and all(x_['identical_to_quick'] for x_ in inv))
    need = {n_: (2 * cfg['rom_reps'] if n_ in timed_rom else cfg['fom_reps']) for n_ in timed_rom + timed_fom}
    rep['gates']['retained_repetitions'] = dict(minimum=min(counts.values()) if counts else 0,
                                                passed=bool(counts) and all(v_ >= need[k[0]] for k, v_ in counts.items()))
    rep['checkpoint_sha256_after'] = sha_file(a.checkpoint)
    assert rep['checkpoint_sha256'] == rep['checkpoint_sha256_after']
    rep['elapsed_seconds'] = time.perf_counter() - begin
    rep['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('EQTOL COMPLETE', flush=True)


if __name__ == '__main__':
    main()
