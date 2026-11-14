"""b2d-worstcase: development driver for the linear rung of the Burgers 2D model (PROTOCOL.md sections 4-6).

For one bank (checkpoint `params` + rotation T) and one cohort it measures, per case and per output time:
  * the projection error of the reference onto the span of the first R' rotated columns (grid least squares),
  * the error of the reduced solve for each arm of the config (the iso-error lane's engineered linear query,
    burgers2d-speed/b2fast.make_linear_query, unchanged), and the distance of the reduced solution from the
    projection of the reference,
  * LM iteration counts / stop reasons, and a simple warm query time (median of `time_reps` calls per case;
    development timing only -- final timing uses the iso-error driver).

The operator / bank / rule construction is the text of experiments/iso-error/b2iso.py; what is new here is the
cohort selection by slice, the projection, the `ic: grid` option (initial coefficients by least squares on the
full interior grid instead of the 48 x 48 Gauss points) and the absence of the certificate / A-B-A phases.

    python wc.py --config cfg.json --checkpoint ck.pkl --rotation rot.npz --out output
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import engines as e
import iterative_paths as ip
import arms as A
import hops as H
import bkfast as BK            # patches hops.bank_apply for the nested rotated bank
import b2fast as B2
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / 'fastsolve'))
import qsfast as QS            # query-speed lane's lagged-operator solve (byte-identical copy)


def sha_array(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cohort(cfg):
    c = cfg['cohort']
    P = np.concatenate([e.params_draw(int(s), int(n)) for s, n in c['draws']])
    if c.get('slice'):
        P = P[int(c['slice'][0]):int(c['slice'][1])]
    if c.get('take'):
        P = P[:int(c['take'])]
    return P


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    return x


def arm_name(s):
    n = f"R{s['Rp']}_M{s['M']}_{s['rule']}_g{s['gtol']:g}"
    if s.get('exact_steps'):
        n += f"_x{s['exact_steps']}"
    if s.get('ic', 'gauss') != 'gauss':
        n += f"_ic{s['ic']}"
    if s.get('budget'):
        n += f"_b{s['budget']}"
    if s.get('tag'):
        n += f"_{s['tag']}"
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--rotation', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not cfg.get('allow_cpu_smoke'):
        assert jax.default_backend() == 'gpu', jax.default_backend()
    assert os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    begin = time.perf_counter()
    el = lambda: round(time.perf_counter() - begin, 1)
    L, dt = int(cfg['intervals']), cfg['dt']
    n = (L - 1) ** 2

    ck = pickle.load(open(a.checkpoint, 'rb'))
    params = jax.tree_util.tree_map(jnp.asarray, jax.tree_util.tree_map(np.asarray, ck['params']))
    R = int(np.asarray(ck['params']['g'][-1][0]).shape[1])
    rz = np.load(a.rotation)
    T = np.asarray(rz['T'])
    Rb = int(T.shape[1])
    edges = [0] + [int(v) for v in cfg['ladder'] if int(v) <= Rb]
    assert edges[-1] == Rb and T.shape[0] == R, (edges, T.shape, R)
    rep = dict(config=cfg, commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
               host=os.uname().nodename, backend=jax.default_backend(), gpu=jax.devices()[0].device_kind,
               checkpoint_sha256=sha_file(a.checkpoint), rotation_sha256=sha_file(a.rotation), R=R, Rb=Rb,
               intervals=L, dt=dt, projection={}, arms={}, complete=False)
    save = lambda: (out / 'result.json').write_text(json.dumps(clean(rep)))

    # ------------------------------------------------------------------ cohort --
    physical = cohort(cfg)
    ncase = len(physical)
    rep['cohort'] = dict(name=cfg['cohort']['name'], cases=ncase, sha256=sha_array(physical))
    rep['physical'] = physical.tolist()
    for seed, cnt, nm in cfg['cohort'].get('disjoint_from', []):
        other = e.params_draw(int(seed), int(cnt))
        hit = any(np.any(np.all(np.isclose(o, physical), axis=1)) for o in other)
        assert not hit or cfg['cohort'].get('allow_overlap'), f'cohort overlaps {nm}'
    want = cfg['cohort'].get('expected_sha256')
    assert want is None or want == rep['cohort']['sha256'], 'cohort differs'
    inputs_u = [e.initial(L, ph) for ph in physical]
    n0 = np.array([float(np.linalg.norm(u)) for u in inputs_u])

    # ------------------------------------------------------ rotation, bank ----
    Tj = jnp.asarray(T)
    base = A.CoordBank(params, 0, R)
    nrb = int(np.ceil(n * R / H.MAX_GEMM_ELEMENTS))
    x = np.arange(1, L) / L
    redges = np.linspace(0, L - 1, nrb + 1).astype(int)
    rotate = jax.jit(lambda g, t: tuple((g @ t)[:, a_:b_] for a_, b_ in zip(edges[:-1], edges[1:])))
    Grot = []
    for i0, i1 in zip(redges[:-1], redges[1:]):
        xy = np.stack(np.meshgrid(x[i0:i1], x, indexing='ij'), -1).reshape(-1, 2)
        g = jax.block_until_ready(base.at(xy, chunk=8192))
        Grot.append(jax.block_until_ready(rotate(g, Tj)))
        del g
    Grot = tuple(Grot)
    Gflat = jax.block_until_ready(BK.bank_apply(Grot, jnp.eye(Rb)))               # (n, Rb), grid rows i-major
    gram = np.asarray(Gflat.T @ Gflat)
    rep['bank'] = dict(rows=n, row_blocks=nrb, column_edges=edges,
                       grid_orthonormality_deviation=float(np.linalg.norm(gram - np.eye(Rb))))
    print('BANK', n, 'rows', nrb, 'blocks; orth dev', rep['bank']['grid_orthonormality_deviation'], el(), flush=True)

    # ------------------------------------------------------------- truth ----
    tight = cfg['reference']
    fom, pre = ip.make_fom(L, tight['dt'], 'fft')
    truth = np.zeros((ncase, 6, L + 1, L + 1))
    worst_res = 0.
    for c in range(ncase):
        v = fom(jnp.asarray(inputs_u[c]), float(physical[c, 4]), tight['ntol'], tight['ltol'], *pre)
        f = np.asarray(v[0])
        rn = np.asarray(v[2])
        assert np.isfinite(f).all() and rn.max() <= tight['ntol'] * (1 + 1e-9), ('truth not converged', c)
        worst_res = max(worst_res, float(rn.max()))
        truth[c] = f
    rep['truth'] = dict(solver=tight, worst_relative_residual=worst_res, seconds=el())
    print('TRUTH done', el(), flush=True)
    tin = jnp.asarray(truth[:, :, 1:-1, 1:-1].reshape(ncase, 6, n))

    # --------------------------------------------------------- projection ----
    Qs = {}

    def Qof(Rp):
        if Rp not in Qs:
            Qs[Rp] = jax.block_until_ready(jnp.linalg.qr(Gflat[:, :Rp], mode='reduced')[0])
        return Qs[Rp]

    proj_fields = {}
    for Rp in cfg['projection_widths']:
        Q = Qof(Rp)
        pr = jnp.einsum('nr,ctn->ctr', Q, tin)
        pf = jnp.einsum('nr,ctr->ctn', Q, pr)
        err = np.asarray(jnp.linalg.norm(pf - tin, axis=-1)) / n0[:, None]            # (case, 6)
        proj_fields[Rp] = pf
        ev = err[:, 1:].max(1)
        rep['projection'][str(Rp)] = dict(per_time=err.tolist(), evolved=ev.tolist(), t0=err[:, 0].tolist(),
                                          worst=float(ev.max()), median=float(np.median(ev)),
                                          p95=float(np.quantile(ev, .95)), worst_any_time=float(err.max()))
        print(f'PROJ R{Rp} evolved worst% {100*ev.max():.4f} median% {100*np.median(ev):.4f} '
              f'p95% {100*np.quantile(ev,.95):.4f} t0 worst% {100*err[:,0].max():.4f}', el(), flush=True)
    save()

    # ---------------------------------------------------------- operators ----
    ops_M = {}

    def operators(M):
        if M not in ops_M:
            kx, ky, lam = H.modes_lean(L, M)
            sx, sy = (jnp.asarray(t_) for t_ in H.sine_tables(L, kx, ky))
            o = dict(kx=kx, ky=ky, lam=jnp.asarray(lam), sx=sx, sy=sy,
                     Arot=BK.project_nested(Grot, edges, Rb, sx, sy, L), sepd=B2.dense_tables(L, kx, ky))
            ops_M[M] = jax.block_until_ready(o)
        return ops_M[M]

    rule_nodes, G5 = {}, {}

    def rule(name):
        if name not in rule_nodes:
            ij, w = H.lattice_rule(L, int(cfg['rules'][name]['lattice']))
            G5[name] = jnp.einsum('msr,rp->msp', base.stencil(ij, L), Tj)
            rule_nodes[name] = (ij, w)
        return rule_nodes[name]

    Pq_cache, sepq_cache = {}, {}

    def Pq(name, M):
        if (name, M) not in Pq_cache:
            ij, w = rule(name)
            o = operators(M)
            Pq_cache[(name, M)] = jnp.asarray(H.phi_rows(L, o['kx'], o['ky'], ij) * w[:, None])
        return Pq_cache[(name, M)]

    def sepq(name, M):
        if (name, M) not in sepq_cache:
            o = operators(M)
            sepq_cache[(name, M)] = B2.lattice_tables(L, int(cfg['rules'][name]['lattice']), o['kx'], o['ky'])
        return sepq_cache[(name, M)]

    # ------------------------------------------------ trust radius, codes ----
    if 'codes' in rz.files:
        codes_rot = np.asarray(rz['codes'])[:, :Rb]          # training coefficient vectors, rotated coordinates
        rep['codes_source'] = 'rotation file (training projections)'
    else:
        Zold = np.asarray(ck['Z_tr'])
        hv0 = jax.jit(jax.vmap(lambda z: A.sc.head(params, z)))
        Hall = np.concatenate([np.asarray(hv0(jnp.asarray(Zold[s:s + 8192]))) for s in range(0, len(Zold), 8192)])
        codes_rot = Hall @ np.asarray(rz['L'])[:Rb].T
        rep['codes_source'] = 'checkpoint head at the stored training codes (as the deployed model)'
    trust = {}

    def trust_of(Rp):
        if Rp not in trust:
            cz = codes_rot[:, :Rp]
            trust[Rp] = .01 * float(np.max(np.linalg.norm(cz - cz.mean(0), axis=1)))
        return trust[Rp]

    colds = {}

    def cold_of(Rp, ic):
        if (Rp, ic) not in colds:
            if ic == 'gauss':
                bank = BK.RotBank(base, T, Rp)
                colds[(Rp, ic)] = A.build_cold(bank, (lambda w_: w_), codes_rot[:8, :Rp], cfg['cold_axis_points'])[0]
            else:                                            # least squares on the full interior grid
                xy = jnp.asarray(e.coords(L))
                w = jnp.full((n,), 1. / np.sqrt(n))
                Q_, R_ = jnp.linalg.qr(Gflat[:, :Rp] * w[:, None], mode='reduced')
                z = jnp.zeros((1, Rp))
                colds[(Rp, ic)] = jax.block_until_ready((xy, w, Q_, R_, z, jnp.zeros((1,)), z))
        return colds[(Rp, ic)]

    Gf_cache = {}

    def Gfull(Rp):
        if Rp not in Gf_cache:
            Gf_cache[Rp] = Gflat[:, :Rp].reshape(L - 1, L - 1, Rp)
        return Gf_cache[Rp]

    def chunks_for(d):
        return next(k for k in range(1, d + 1) if d % k == 0 and d // k <= cfg.get('dense_tangent_group', 34))

    # ---------------------------------------------------------------- arms ----
    reps = int(cfg.get('time_reps', 3))
    for s in cfg['arms']:
        name = arm_name(s)
        Rp, M, j = int(s['Rp']), int(s['M']), int(s.get('exact_steps', 0))
        nc = min(ncase, int(s.get('cases') or ncase))
        t_arm = time.perf_counter()
        o = operators(M)
        rule(s['rule'])
        data = dict(A=o['Arot'][:, :Rp], lam=o['lam'], G=BK.prefix(Grot, edges, Rp), G5=G5[s['rule']][:, :, :Rp],
                    Pq=Pq(s['rule'], M), sx=o['sx'], sy=o['sy'], sepq=sepq(s['rule'], M), sepd=o['sepd'])
        if j:
            data['Gf'] = Gfull(Rp)
        opts = B2.GRAPHS if (s.get('graphs') and jax.default_backend() == 'gpu') else None
        if s.get('qs'):                                      # fast solve of the query-speed lane, same data dict
            assert not j
            if s['qs'].get('stencil32'):
                data['G5d32'] = QS.stencil_diff32(data['G5'])
            fq = QS.make_query(Rp, L, dt, compiler_options=opts, **s['qs'])
        else:
            fq = B2.make_linear_query(Rp, L, dt, trust_of(Rp) * float(s.get('trust_factor', 1.)), exact_steps=j,
                                      tangent_chunks=chunks_for(Rp), lm_cap=s.get('cap'), compiler_options=opts,
                                      sep_eq=True, ajac_dense=bool(j), sep_dense=True,
                                      step_budget=int(s.get('budget') or cfg['step_budget']), gtol=s['gtol'],
                                      ridge=cfg['inner_damping'], solver='chol', clip=True, lam_carry=True,
                                      predictor='quad')
        cold = cold_of(Rp, s.get('ic', 'gauss'))
        pf = proj_fields.get(Rp)
        rows = []
        for c in range(nc):
            u0 = jnp.asarray(inputs_u[c])
            nu = float(physical[c, 4])
            v = jax.block_until_ready(fq(u0, nu, data, cold, {}))
            ts = []
            for _ in range(reps if c < int(cfg.get('time_cases', 16)) else 0):
                t0 = time.perf_counter()
                jax.block_until_ready(fq(u0, nu, data, cold, {}))
                ts.append(time.perf_counter() - t0)
            f = np.asarray(v[0])
            ok = bool(np.isfinite(f).all())
            err = [float(np.linalg.norm(f[t] - truth[c, t])) / n0[c] for t in range(6)] if ok else [None] * 6
            row = dict(case=c, ok=ok, per_time=err, evolved=(max(err[1:]) if ok else None),
                       iterations=int(np.asarray(v[1]).sum()), max_iterations=int(np.asarray(v[1]).max()),
                       reasons=np.bincount(np.asarray(v[3]).astype(int), minlength=6).tolist(),
                       seconds=(float(np.median(ts)) if ts else None))
            if pf is not None and ok:
                fi = jnp.asarray(f[:, 1:-1, 1:-1].reshape(6, n))
                row['to_projection'] = (np.asarray(jnp.linalg.norm(fi - pf[c], axis=-1)) / n0[c]).tolist()
            rows.append(row)
        ev = np.array([r_['evolved'] if r_['ok'] else np.inf for r_ in rows])
        tt = [r_['seconds'] for r_ in rows if r_['seconds'] is not None]
        summ = dict(spec=s, cases=nc, trust=trust_of(Rp), failed=int(np.sum(~np.isfinite(ev))),
                    worst=float(ev.max()), median=float(np.median(ev)), p95=float(np.quantile(ev, .95)),
                    mean=float(ev.mean()), worst_case_index=int(np.argmax(ev)),
                    ms=(1e3 * float(np.median(tt)) if tt else None),
                    ms_mean=(1e3 * float(np.mean(tt)) if tt else None), ms_max=(1e3 * float(np.max(tt)) if tt else None),
                    mean_iterations=float(np.mean([r_['iterations'] for r_ in rows])),
                    setup_and_run_seconds=time.perf_counter() - t_arm, rows=rows)
        rep['arms'][name] = summ
        print(f"ARM {name} cases {nc} worst% {100*summ['worst']:.4f} median% {100*summ['median']:.4f} "
              f"p95% {100*summ['p95']:.4f} ms {summ['ms'] and round(summ['ms'],2)} iters {summ['mean_iterations']:.1f} "
              f"failed {summ['failed']}", el(), flush=True)
        save()
        del fq, data
    rep['complete'] = True
    rep['seconds'] = el()
    save()
    print('ALL-ARMS-DONE', el(), flush=True)


if __name__ == '__main__':
    main()
