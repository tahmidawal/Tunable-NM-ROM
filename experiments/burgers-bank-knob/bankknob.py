"""burgers-bank-knob: nested bank truncation R' of the frozen Burgers 2D model, one mesh, ONE allocation (DESIGN.md).

Derived from experiments/burgers-repanel/repanel.py @ 219feb6e (itself burgers-eqcert/eqcert.py @ 176b2a9a): same
cohort, same same-grid full-order reference, same Newton-BiCGStab candidate grid, same optimised solver path
(hfast / xfast: fused analytic Jacobian, Cholesky, clipped step, damping carry-over, quadratic predictor), same
output contract and timing scopes. What is new (bkfast.py): the rotated bank, the R' truncation of every arm, the
linear rung, the per-arm deployed-state rho certificate on a held-out population at THIS mesh, and a timed panel
split into a main phase and a slow phase with a cool-down, so the order-effect gate can pass.

Phases, each written incrementally to result.json:
  1 cohort; same-grid reference (`fft_tight`) per case
  2 rotation (verified hash), banks (parent row blocks if kept; rotated nested blocks), Phi-free operators (gated)
  3 rules (frozen: q0 `scaled` file, `lat64` lattice, `bad0` control file), models, arms
  4 QUICK: every arm once per case against the reference; parity gates (rotated R'=R vs unrotated parent)
  5 CERTIFY: every EQ arm run on the held-out population (params_draw(*population.source_draw)) at this mesh;
    rho = || Pq^T a_EQ - Phi^T a || / || Phi^T a || on every reached state, per draw
  6 TIMED: main phase (arms whose quick run took < slow_threshold s) and slow phase, each randomised, burn-in,
    cool-down after long neighbours, synchronised, `reps` repetitions x cases
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

sha_array, sha_file, host, dump = LD.sha_array, LD.sha_file, LD.host, LD.dump
TIMES = np.array([0., .05, .1, .15, .2, .25])


def gt(g):
    return f'g{g:g}'.replace('-', 'm').replace('.', 'p')


def rel_per_time(f, truth, n0):
    return [float(np.linalg.norm(a - b)) / n0 for a, b in zip(f, truth)]


def arm_name(s):
    v = s['variant']
    sfx = ''.join(f'_{k}' for k in ([v['solver']] if v.get('solver', 'lu') != 'lu' else []) +
                  [k for k in ('clip', 'lamcarry', 'pred2') if v.get(k)])
    if v.get('exact_steps'):
        sfx += f"_x{v['exact_steps']}"
    head = 'P' if s['model'] == 'parent' else f"R{s['Rp']}"
    body = f"lin_M{s['M']}" if s['model'] == 'lin' else f"q{s['q']}_M{s['M']}"
    return f"{head}_{body}_{s['rule']}_{gt(s['gtol'])}_fast{sfx}"


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

    ck = pickle.load(open(a.checkpoint, 'rb'))
    params = jax.tree_util.tree_map(jnp.asarray, host(ck['params']))
    Zold = np.asarray(ck['Z_tr'])
    K = int(Zold.shape[1])
    R = int(np.asarray(ck['params']['h_lin']).shape[1])
    L, dt = int(cfg['intervals']), cfg['dt']
    st = cfg['strict']
    BAR = cfg['rho_bar']
    edges = [0] + list(cfg['ladder'])
    assert edges[-1] == R and edges == sorted(edges)
    rep = dict(config=cfg, commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
               backend=jax.default_backend(), gpu=jax.devices()[0].device_kind, nvidia_smi=smi,
               x64=True, matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'], jax_version=jax.__version__,
               checkpoint_sha256=sha_file(a.checkpoint), K=K, R=R, intervals=L, dt=dt, weights_frozen=True,
               output_times=TIMES.tolist(),
               timing_contract=('gpu_seconds: supplied dense initial field resident on the GPU to six dense GPU '
                                'output fields, block_until_ready on both sides; host_seconds: the same invocation '
                                'plus the host upload of the input and the host copy of the six outputs; identical '
                                'for every subject'),
               rules=[], models={}, arm_setup=[], quick=[], parity=[], certificates={}, invocations=[],
               phases={}, gates={}, dropped=[], complete=False)

    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (float, np.floating)):
            return float(x) if np.isfinite(x) else None
        if isinstance(x, np.integer):
            return int(x)
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

    # ------------------------------------------------------ rotation, banks ----
    rz = np.load(a.rotation)
    T, Lrot = np.asarray(rz['T']), np.asarray(rz['L'])
    rep['rotation'] = dict(file=str(a.rotation), file_sha256=sha_file(a.rotation), T_sha256=sha_array(T),
                           L_sha256=sha_array(Lrot), expected_file_sha256=cfg['rotation_sha256'],
                           L_times_T_identity_deviation=float(np.linalg.norm(Lrot @ T - np.eye(R))))
    assert rep['rotation']['file_sha256'] == cfg['rotation_sha256'], 'rotation file differs from the committed one'
    Tj = jnp.asarray(T)
    base = A.CoordBank(params, K, R)
    keep_parent = bool(cfg.get('keep_parent'))
    t0 = time.perf_counter()
    n = (L - 1) ** 2
    nrb = int(cfg.get('bank_blocks') or np.ceil(n * R / H.MAX_GEMM_ELEMENTS))
    x = np.arange(1, L) / L
    redges = np.linspace(0, L - 1, nrb + 1).astype(int)
    rotate = jax.jit(lambda g, t: tuple((g @ t)[:, a_:b_] for a_, b_ in zip(edges[:-1], edges[1:])))
    Gpar, Grot = [], []
    for i0, i1 in zip(redges[:-1], redges[1:]):
        xy = np.stack(np.meshgrid(x[i0:i1], x, indexing='ij'), -1).reshape(-1, 2)
        g = jax.block_until_ready(base.at(xy, chunk=8192))
        Grot.append(jax.block_until_ready(rotate(g, Tj)))
        if keep_parent:
            Gpar.append(g)
        del g
    Gpar, Grot = tuple(Gpar), tuple(Grot)
    rep['phases']['bank'] = dict(rows=n, row_blocks=nrb, column_edges=edges, parent_kept=keep_parent,
                                 seconds=time.perf_counter() - t0,
                                 rotated_bytes=int(sum(c.nbytes for rb in Grot for c in rb)),
                                 parent_bytes=int(sum(g.nbytes for g in Gpar)))
    print('BANK', n, 'rows', nrb, 'row blocks, parent kept', keep_parent, el(), flush=True)

    # ---------------------------------------------------- full-order solvers ----
    foms = {}

    def fom_for(fs):
        key = (fs.get('mesh', L), fs['dt'], fs.get('impl', 'audited'))
        if key not in foms:
            if key[0] == L and key[2] == 'audited':
                foms[key] = ip.make_fom(L, fs['dt'], 'fft')
            else:
                foms[key] = (e.make_fom(key[0], fs['dt'], target=L)[0], None)
        return foms[key]

    def run_fom(fs, u, nu):
        fn, pre = fom_for(fs)
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
                     Arot=BK.project_nested(Grot, edges, R, sx, sy, L))
            if keep_parent:
                o['Apar'] = H.project_bank(Gpar, sx, sy, L)
            ops_M[M] = jax.block_until_ready(o)
        return ops_M[M]

    # gate: Phi-free operator of the rotated bank equals the parent's times T (same projection, other association)
    Mg = max(s_['M'] for s_ in cfg['arms'])
    og = operators(Mg)
    if keep_parent:
        dA = float(jnp.linalg.norm(og['Arot'] - og['Apar'] @ Tj) / jnp.linalg.norm(og['Apar'] @ Tj))
        rep['gates']['rotated_operator_matches_parent_times_T'] = dict(M=Mg, relative=dA, passed=bool(dA <= 1e-10))
        assert dA <= 1e-10, dA
    if L <= cfg.get('phi_gate_max_mesh', 1024):
        Phig, lamg, _ = e.modes(L, Mg)
        Gflat = BK.bank_apply(Grot, jnp.eye(R))
        Aref = jnp.asarray(Phig).T @ Gflat
        g0 = dict(M=Mg, lam_identical=bool(np.array_equal(lamg, np.asarray(og['lam']))),
                  A_relative=float(jnp.linalg.norm(og['Arot'] - Aref) / jnp.linalg.norm(Aref)))
        g0['passed'] = bool(g0['lam_identical'] and g0['A_relative'] <= 1e-12)
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
            ij, w = H.lattice_rule(L, int(rs['lattice']))
            src = dict(kind='lattice', s=int(rs['lattice']))
        else:
            fp = inputs / rs['file']
            assert sha_file(fp) == rs['sha256'], rs['file']
            z = np.load(fp)
            ij = H.transfer_nodes(z['nodes'].astype(int), rs['mesh'], L)
            w = np.asarray(z['weights'], float) * (L / rs['mesh']) ** 2
            src = dict(kind='transfer', file=rs['file'], sha256=rs['sha256'], source_mesh=rs['mesh'])
        keep = w > 0
        ij, w = ij[keep], w[keep]
        g5 = base.stencil(ij, L)                                   # (m, 5, R) parent
        G5[name] = dict(par=g5 if keep_parent else None, rot=jnp.einsum('msr,rp->msp', g5, Tj))
        np.savez_compressed(out / f'rule_L{L}_{name}.npz', ij=ij, weights=w)
        rule_nodes[name] = (ij, w)
        rep['rules'].append(dict(rule=name, m=int(len(ij)), source=src, control=bool(rs.get('control')),
                                 note=rs.get('note')))
        return rule_nodes[name]

    Pq_cache = {}

    def Pq(name, M):
        if (name, M) not in Pq_cache:
            ij, w = rule(name)
            o = operators(M)
            Pq_cache[(name, M)] = jnp.asarray(H.phi_rows(L, o['kx'], o['ky'], ij) * w[:, None])
        return Pq_cache[(name, M)]

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
        if kind == 'parent':
            assert keep_parent and Rp == R
            m_ = dict(params=params, C=Cfull, bank=base, G=Gpar, trust=trust0, codes=Zsub,
                      A=lambda M: operators(M)['Apar'], G5=lambda nm: G5[nm]['par'])
        elif kind == 'trunc':
            m_ = dict(params=BK.fold_head(params, Lrot, Rp), C=jnp.asarray(Lrot[:Rp]) @ Cfull,
                      bank=BK.RotBank(base, T, Rp), G=BK.prefix(Grot, edges, Rp), trust=trust0, codes=Zsub,
                      A=lambda M, Rp=Rp: operators(M)['Arot'][:, :Rp],
                      G5=lambda nm, Rp=Rp: G5[nm]['rot'][:, :, :Rp])
        else:                                                   # linear rung: codes = training coefficient vectors
            codes_all = Hall @ Lrot[:Rp].T
            trust = .01 * float(np.max(np.linalg.norm(codes_all - codes_all.mean(0), axis=1)))
            m_ = dict(params=None, C=None, bank=BK.RotBank(base, T, Rp), G=BK.prefix(Grot, edges, Rp), trust=trust,
                      codes=Hsub @ Lrot[:Rp].T, A=lambda M, Rp=Rp: operators(M)['Arot'][:, :Rp],
                      G5=lambda nm, Rp=Rp: G5[nm]['rot'][:, :, :Rp])
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
        name = arm_name(s)
        v = s['variant']
        j = int(v.get('exact_steps', 0))
        kind, Rp, q, M = s['model'], int(s.get('Rp', R)), int(s['q']), int(s['M'])
        m_ = model(kind, Rp)
        o = operators(M)
        cold, coef = cold_head(m_, kind, q)
        rule(s['rule'])
        data = dict(A=m_['A'](M), lam=o['lam'], G=m_['G'], G5=m_['G5'](s['rule']), Pq=Pq(s['rule'], M),
                    sx=o['sx'], sy=o['sy'])
        common = dict(step_budget=st['step_budget'], gtol=s['gtol'], ridge=cfg['inner_damping'],
                      solver=v.get('solver', 'lu'), clip=bool(v.get('clip')), lam_carry=bool(v.get('lamcarry')),
                      predictor='quad' if v.get('pred2') else 'lin')
        if kind == 'lin':
            d = Rp
            fq = BK.make_linear_query(Rp, L, dt, m_['trust'], exact_steps=j, tangent_chunks=chunks_for(d), **common)
            tab = {}
        else:
            d = K + q
            C = m_['C'][:, :q]
            tab = HF.build_tables(m_['params'], C, K, data, cold)
            fq = XF.make_query(m_['params'], C, K, q, L, dt, m_['trust'], 'eq', exact_steps=j,
                               ic_budget=st['ic_budget'], ic_gtol=cfg['ic_gtol'], tangent_chunks=chunks_for(d),
                               **common)
        extra = {}
        if tab.get('Rr') is not None:
            sv = np.asarray(jnp.linalg.svd(tab['Rr'], compute_uv=False))
            extra['correction_elimination_condition_number'] = float(sv[0] / sv[-1])
        built[name] = dict(name=name, kind='rom', family='rom', spec=s, data=data, cold=cold, coef=coef, tab=tab,
                           query=(lambda u, nu, d_, c_, _t=tab, _f=fq: _f(u, nu, d_, c_, _t)),
                           m=int(data['G5'].shape[0]), unknowns=d, exact_steps=j)
        rep['arm_setup'].append(dict(arm=name, family='rom', model=kind, R_prime=Rp, q=q, M=M, rule=s['rule'],
                                     m=int(data['G5'].shape[0]), unknowns=d, gtol=s['gtol'], variant=v,
                                     exact_steps=j, trust=m_['trust'], role=s.get('role'),
                                     control=bool(cfg['rules'][s['rule']].get('control')),
                                     timed=not s.get('untimed', False), certify=bool(s.get('certify', True)), **extra))
        return name

    def invoke(b, u, c):
        nu = float(physical[c, 4])
        if b['kind'] == 'fom':
            return run_fom(b['setting'], u, nu)
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
                    newton_iterations_total=int(np.sum(np.asarray(v[1]))), steps=int(len(rn)),
                    stalled_steps=int(np.sum(~okstep)), nonlinear_converged=bool(okstep.all()),
                    max_relative_residual=float(np.nanmax(rn)))

    keep_fields = set(n_ for pr in cfg.get('parity_pairs', []) for n_ in pr)

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
                row = dict(name=name, case=c, family=b['family'], field_sha256=sha_array(f), seconds=secs[-1], **score(f, c))
                row.update(rom_row(vh) if b['kind'] == 'rom' else fom_row(b, vh))
                rep['quick'].append(row)
                if name in keep_fields:
                    kept[(name, c)] = (f, vh[1:4])
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
                  'iters', [x_.get('total_iterations') for x_ in rows], 'sec', round(b['quick_seconds'], 3),
                  round(time.perf_counter() - t_arm, 1), el(), flush=True)
            save()

    for fs in cfg['fom_settings']:
        built[fs['name']] = dict(name=fs['name'], kind='fom', family='fom', setting=fs)
    run_quick([fs['name'] for fs in cfg['fom_settings']])
    for s in cfg['arms']:
        run_quick([build_arm(s)])
    rep['phases']['quick_seconds'] = el()

    # parity: rotated R' = R against the unrotated parent (same arm otherwise)
    PBAR = cfg['parity_bar']
    for fa, fb in cfg.get('parity_pairs', []):
        per = []
        for c in range(ncase):
            (xa, ia), (xb, ib) = kept[(fa, c)], kept[(fb, c)]
            per.append(dict(case=c, relative=float(np.linalg.norm(xa - xb) / np.linalg.norm(xb)),
                            iterations_identical=bool(np.array_equal(ia[0], ib[0])),
                            reasons_identical=bool(np.array_equal(ia[2], ib[2])),
                            iterations_rotated=int(np.sum(ia[0])), iterations_parent=int(np.sum(ib[0]))))
        worst = max(x_['relative'] for x_ in per)
        ints = all(x_['iterations_identical'] and x_['reasons_identical'] for x_ in per)
        rep['parity'].append(dict(rotated=fa, parent=fb, worst_relative=worst, integers_identical=ints, bar=PBAR,
                                  passed=bool(worst <= PBAR and ints), cases=per))
        print('PARITY', fa, 'vs', fb, f'{worst:.3e}', 'integers', ints, flush=True)
    kept.clear()
    save()

    # ---------------------------------------------------------- certify ----
    steps_total = int(round(.25 / dt))
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
        per = []
        for d_ in range(len(draws)):
            sel = (meta[:, 0] == d_) & (meta[:, 2] >= j)
            per.append(dict(draw=d_, states=int(sel.sum()), rho_max=float(r[sel].max()),
                            rho_p95=float(np.quantile(r[sel], .95)), passed=bool(r[sel].max() <= BAR)))
        npass = sum(x_['passed'] for x_ in per[:ncert])
        conf = per[ncert]['passed'] if len(per) > ncert else None
        status = ('confirmed' if npass == ncert and conf is not False else
                  'fails' if npass == 0 else f'not confirmed ({npass}/{ncert} draws, confirmation {conf})')
        rep['certificates'][name] = dict(k_threshold=j, per_draw=per, draws_passed=npass, draws=ncert,
                                         confirmation_pass=conf, status=status, bar=BAR, stalled_steps=stall,
                                         rho_max_all_k=float(r[meta[:, 0] < ncert].max()),
                                         seconds=time.perf_counter() - t0)
        print('CERT', name, status, 'rho_max', f"{max(x_['rho_max'] for x_ in per[:ncert]):.4f}",
              'conf', f"{per[ncert]['rho_max']:.4f}" if len(per) > ncert else None, round(time.perf_counter() - t0, 1),
              el(), flush=True)
        save()

    # ------------------------------------------------------------ timed ----
    timed = [n_ for n_ in subjects if not (built[n_]['kind'] == 'rom' and built[n_]['spec'].get('untimed'))
             and n_ not in cfg.get('untimed_fom', [])]
    for n_ in timed:
        jax.block_until_ready(invoke(built[n_], jnp.asarray(inputs_u[0]), 0))
    thr = cfg['slow_threshold_seconds']
    phases = [('main', [n_ for n_ in timed if built[n_]['quick_seconds'] < thr]),
              ('slow', [n_ for n_ in timed if built[n_]['quick_seconds'] >= thr])]
    rep['phases']['timed_phases'] = {k: v for k, v in phases}
    order_rng = np.random.default_rng(cfg['order_seed'])
    quick_sha = {(x_['name'], x_['case']): x_['field_sha256'] for x_ in rep['quick']}
    for ph_name, arms_ in phases:
        prev = 0.
        seq = 0
        for r_ in range(cfg['reps']):
            for c in range(ncase):
                for i in order_rng.permutation(len(arms_)):
                    name = arms_[int(i)]
                    b = built[name]
                    if prev >= cfg['cooldown_after_seconds']:
                        time.sleep(min(cfg['cooldown_max_seconds'], prev))
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
                    prev = gs
                    same = sha_array(f) == quick_sha[(name, c)]
                    row = dict(name=name, case=c, rep=r_, phase=ph_name, seq=seq, family=b['family'], gpu_seconds=gs,
                               host_seconds=hs, identical_to_quick=same, same_grid_evolved=score(f, c)['same_grid_evolved'])
                    vh = (None,) + tuple(host(v[1:]))
                    if b['kind'] == 'rom':
                        row['total_iterations'] = int(np.sum(np.asarray(vh[1])))
                    rep['invocations'].append(row)
                    seq += 1
            print('TIMED', ph_name, 'rep', r_, el(), flush=True)
            save()
    cover = {(n_, c) for n_ in timed for c in range(ncase)}
    counts = {k: sum(1 for x_ in rep['invocations'] if (x_['name'], x_['case']) == k) for k in cover}
    rep['gates']['repetition_output_identical'] = dict(
        passed=bool(rep['invocations']) and all(x_['identical_to_quick'] for x_ in rep['invocations']))
    rep['gates']['retained_repetitions'] = dict(passed=bool(counts) and min(counts.values()) >= cfg['required_reps'],
                                                minimum=min(counts.values()) if counts else 0)
    rep['checkpoint_sha256_after'] = sha_file(a.checkpoint)
    assert rep['checkpoint_sha256'] == rep['checkpoint_sha256_after']
    rep['elapsed_seconds'] = time.perf_counter() - begin
    rep['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('BANKKNOB COMPLETE', flush=True)


if __name__ == '__main__':
    main()
