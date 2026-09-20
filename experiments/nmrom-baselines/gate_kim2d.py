"""Reproduction gate: Kim et al. 2022, Section 6.2 (2D viscous Burgers, Re=1e4, 60x60, nt=1500).

Published targets (arXiv 2009.11990v2): NM-LSPG at n_s=5 reaches "less than 1 % maximum relative
error" with n_train=4 (Fig. 14); NM-LSPG-HR 0.93-0.98 % (Table 3, 55/58 basis/samples best);
LS-LSPG-HR 34-38 %. Error = max_n ||x~(t_n)-x(t_n)||_2/||x(t_n)||_2, larger of u and v, mu=1.

Everything is written to --out; fields are saved so the errors can be recomputed with NumPy.
"""
from __future__ import annotations
import argparse, json, os, sys, time, hashlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--nx', type=int, default=60)
    ap.add_argument('--nt', type=int, default=1500)
    ap.add_argument('--ns', type=int, default=5)
    ap.add_argument('--b', type=int, default=100)
    ap.add_argument('--db', type=int, default=10)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--max-epochs', type=int, default=10000)
    ap.add_argument('--train-wall', type=float, default=3000., help='seconds per autoencoder')
    ap.add_argument('--act', default='swish')
    ap.add_argument('--scale', default='feature', choices=['feature', 'global'])
    ap.add_argument('--train-dtype', default='float32')
    ap.add_argument('--hr', type=int, nargs='*', default=[55, 58, 51, 54, 44, 47, 40, 40, 60, 60],
                    help='pairs: residual basis, samples')
    ap.add_argument('--allow-cpu', action='store_true')
    a = ap.parse_args()

    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    backend = jax.default_backend()
    print(f'jax_backend={backend}', flush=True)
    if backend != 'gpu' and not a.allow_cpu:
        sys.exit(42)
    sys.path.insert(0, str(HERE))
    import kimae, lspg
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    act = kimae.ACT[a.act]
    nx = a.nx; m = nx - 2; n = m * m; dx = 1. / (nx - 1); dt = 2. / a.nt; Re = 1e4
    tdt = jnp.float32 if a.train_dtype == 'float32' else jnp.float64

    # ------------------------------------------------------------ FOM
    def f(x):
        u, v = x[:n].reshape(m, m), x[n:].reshape(m, m)
        def rhs(w):
            p = jnp.pad(w, 1)
            c, xm, xp, ym, yp = p[1:-1, 1:-1], p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]
            return -u * (c - xm) / dx - v * (c - ym) / dx + (xm + xp + ym + yp - 4 * c) / (Re * dx * dx)
        return jnp.concatenate((rhs(u).ravel(), rhs(v).ravel()))

    def res(x, prev):
        return x - prev - dt * f(x)

    def fom_step(prev, _):
        def cond(s): return (s[2] > 1e-11 * jnp.maximum(jnp.linalg.norm(prev), 1e-30)) & (s[1] < 30)
        def body(s):
            x, it, _ = s
            r, lin = jax.linearize(lambda q: res(q, prev), x)
            d, _ = jax.scipy.sparse.linalg.gmres(lin, -r, tol=1e-12, atol=0., restart=40, maxiter=20)
            x = x + d
            return x, it + 1, jnp.linalg.norm(res(x, prev))
        x, it, rn = jax.lax.while_loop(cond, body, (prev, jnp.int32(0), jnp.linalg.norm(res(prev, prev))))
        return x, (x, it, rn / jnp.maximum(jnp.linalg.norm(prev), 1e-30))

    fom = jax.jit(lambda x0: jax.lax.scan(fom_step, x0, None, length=a.nt)[1])

    def ic(mu):
        g = np.arange(1, nx - 1) * dx
        X, Y = np.meshgrid(g, g, indexing='ij')
        w = np.where((X <= .5) & (Y <= .5), mu * np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y), 0.)
        return np.concatenate((w.ravel(), w.ravel()))

    mus_train, mu_test = [.9, .95, 1.05, 1.1], 1.0
    snaps = {}
    for mu in mus_train + [mu_test]:
        t0 = time.perf_counter()
        X, it, rn = jax.tree_util.tree_map(np.asarray, fom(jnp.asarray(ic(mu))))
        snaps[mu] = np.concatenate((ic(mu)[None], X))
        print(f'FOM mu={mu} {time.perf_counter()-t0:.1f}s newton_max={it.max()} relres_max={rn.max():.2e}', flush=True)
        assert np.isfinite(X).all() and rn.max() < 1e-9
    t0 = time.perf_counter(); fom(jnp.asarray(ic(mu_test)))[0].block_until_ready()
    fom_seconds = time.perf_counter() - t0

    def maxrel(approx, truth):   # larger of u and v, max over n>=1 (paper: n in N(Nt))
        e = [np.max(np.linalg.norm((approx - truth)[1:, s], axis=1) / np.linalg.norm(truth[1:, s], axis=1))
             for s in (slice(0, n), slice(n, 2 * n))]
        return dict(u=float(e[0]), v=float(e[1]), max=float(max(e)))

    # ------------------------------------------------------------ data, reference x_ref(mu) = x_0(mu)
    D = np.concatenate([snaps[mu] - snaps[mu][:1] for mu in mus_train])       # (6004, 2n)
    rng = np.random.default_rng(20260920)
    perm = rng.permutation(D.shape[0]); nva = D.shape[0] // 10
    va, tr = perm[:nva], perm[nva:]
    idx, valid, M2 = kimae.mask_tables(m, m, a.b, a.db)
    M1 = 2 * n
    print(f'n={n} M1={M1} M2={M2} P={idx.shape[1]} nnz={valid.sum()} train={tr.size} val={va.size}', flush=True)
    idxj = jnp.asarray(idx)
    ref = snaps[mu_test][0]
    truth = snaps[mu_test]
    results = dict(config=vars(a), n=n, M1=M1, M2=M2, fom_seconds_warm=fom_seconds,
                   published=dict(nm_lspg='<1 % (Fig. 14, n_s=5, n_train=4)', nm_lspg_hr_percent=[.93, .94, .95, .97, .97, .98],
                                  ls_lspg_hr_percent=[34.38, 37.73, 37.84, 37.95, 37.96, 37.97]), seeds=[])

    # ------------------------------------------------------------ LS-LSPG control (POD n_s per component)
    def pod(Dc):
        w, V = np.linalg.eigh(Dc.T @ Dc)
        return V[:, ::-1][:, :a.ns]
    Phi = [pod(D[:, :n]), pod(D[:, n:])]
    Phij = [jnp.asarray(P) for P in Phi]
    def ls_res(z, zp, args):
        refj, Pu, Pv = args
        dec = lambda q: refj + jnp.concatenate((Pu @ q[:a.ns], Pv @ q[a.ns:]))
        return res(dec(z), dec(zp))
    ls_roll = jax.jit(lspg.make_rollout(ls_res, a.nt))
    Zls, (it, dn, rn) = ls_roll(jnp.zeros(2 * a.ns), (jnp.asarray(ref), *Phij))
    Zls = np.asarray(Zls)
    ls_field = ref + np.concatenate((Zls[:, :a.ns] @ Phi[0].T, Zls[:, a.ns:] @ Phi[1].T), 1)
    results['ls_lspg'] = dict(error=maxrel(ls_field, truth), gn_iterations_max=int(np.max(it)),
                              finite=bool(np.isfinite(ls_field).all()))
    proj = ref + np.concatenate(((truth - ref)[:, :n] @ Phi[0] @ Phi[0].T, (truth - ref)[:, n:] @ Phi[1] @ Phi[1].T), 1)
    results['ls_projection'] = maxrel(proj, truth)
    print('LS-LSPG', results['ls_lspg'], 'LS projection', results['ls_projection'], flush=True)
    np.savez(out / 'fields_common.npz', truth=truth, ls_lspg=ls_field)

    # ------------------------------------------------------------ autoencoders per seed
    for seed in a.seeds:
        rec = dict(seed=seed, train={})
        P, SC = [], []
        for c, name in enumerate('uv'):
            Dc = D[:, c * n:(c + 1) * n]
            if a.scale == 'feature':
                sc = np.abs(Dc[tr]).max(0); sc = np.where(sc > 1e-12 * sc.max(), sc, 1.)
            else:
                sc = np.full(n, np.abs(Dc[tr]).max())
            Xn = Dc / sc
            p0 = kimae.init(jax.random.PRNGKey(1000 * seed + c), n, a.ns, M1, M2, idx, valid, tdt)
            p, info = kimae.train(p0, jnp.asarray(Xn[tr], tdt), jnp.asarray(Xn[va], tdt), idx, valid, act,
                                  batch=240, micro=240, max_epochs=a.max_epochs, wall_seconds=a.train_wall,
                                  seed=seed, tag=f'seed{seed}-{name}')
            np.save(out / f'history_seed{seed}_{name}.npy', info.pop('history'))
            rec['train'][name] = info
            print(f'TRAINED seed={seed} {name}', info, flush=True)
            P.append(jax.tree_util.tree_map(lambda w: jnp.asarray(w, jnp.float64), p)); SC.append(jnp.asarray(sc))
            import pickle
            with open(out / f'ae_seed{seed}_{name}.pkl', 'wb') as fh:
                pickle.dump(dict(params=jax.tree_util.tree_map(np.asarray, p), scale=sc), fh)

        def dec_full(z, args):
            refj, pu, pv, su, sv, ix = args
            return refj + jnp.concatenate((kimae.decode(pu, z[:a.ns], ix, act) * su, kimae.decode(pv, z[a.ns:], ix, act) * sv))
        args = (jnp.asarray(ref), P[0], P[1], SC[0], SC[1], idxj)
        # nonlinear projection error (their Eq. 6.2 uses g(h(.)); reported with the max-rel metric too)
        enc = jax.jit(jax.vmap(lambda x, args: jnp.concatenate((kimae.encode(args[1], x[:n] / args[3], act),
                                                                  kimae.encode(args[2], x[n:] / args[4], act))), (0, None)))
        decv = jax.jit(jax.vmap(dec_full, (0, None)))
        Zp = enc(jnp.asarray(truth - ref), args)
        rec['nm_projection'] = maxrel(np.asarray(decv(Zp, args)), truth)
        z0 = enc(jnp.zeros((1, 2 * n)), args)[0]
        nm_res = lambda z, zp, args: res(dec_full(z, args), dec_full(zp, args))
        roll = jax.jit(lspg.make_rollout(nm_res, a.nt))
        Z, (it, dn, rn) = roll(z0, args); Z.block_until_ready()
        t0 = time.perf_counter(); Z, (it, dn, rn) = roll(z0, args); Z.block_until_ready(); tq = time.perf_counter() - t0
        field = np.asarray(decv(Z, args))
        rec['nm_lspg'] = dict(error=maxrel(field, truth), seconds_warm=tq, gn_iterations_mean=float(np.mean(it)),
                              gn_iterations_max=int(np.max(it)), finite=bool(np.isfinite(field).all()),
                              ic_error=float(np.linalg.norm(field[0] - truth[0]) / np.linalg.norm(truth[0])))
        print(f'NM-LSPG seed={seed}', rec['nm_lspg'], 'projection', rec['nm_projection'], flush=True)
        np.savez(out / f'fields_seed{seed}.npz', nm_lspg=field, Z=np.asarray(Z))

        # ---------------- hyper-reduction: residual snapshots from NM-LSPG on the training parameters
        if a.hr:
            rollr = jax.jit(lspg.make_rollout(nm_res, a.nt, keep_residuals=True))
            Rs = []
            for mu in mus_train:
                am = (jnp.asarray(snaps[mu][0]),) + args[1:]
                _, o = rollr(z0, am)
                Rs.append(np.asarray(o[3]).reshape(-1, 2 * n))
            Rs = np.concatenate(Rs).T
            rec['hr'] = []
            nbr = neighbour_table(m)
            for nr, nz in zip(a.hr[::2], a.hr[1::2]):
                Phir = kimae.pod_basis(Rs, nr)
                rows = kimae.greedy_samples(Phir, nz)
                pinv = jnp.asarray(np.linalg.pinv(Phir[rows]))
                comp, node = np.divmod(rows, n)
                need = np.unique(np.concatenate((node, nbr[node][nbr[node] >= 0].ravel())))
                pos = np.full(n + 1, need.size, np.int64); pos[need] = np.arange(need.size)   # sentinel -> zero slot
                loc = jnp.asarray(np.stack([pos[node]] + [pos[np.where(nbr[node, k] >= 0, nbr[node, k], n)] for k in range(4)], 1))
                subs = [kimae.subnet(P[c], idx, valid, need) for c in range(2)]
                compj = jnp.asarray(comp)
                hargs = (jnp.asarray(ref)[jnp.asarray(np.concatenate((need, need + n)))], subs[0][0], subs[1][0],
                         SC[0][need], SC[1][need], subs[0][1], subs[1][1], loc, compj, pinv)
                def hr_raw(z, zp, h):
                    refn, pu, pv, su, sv, iu, iv, loc, comp, pinv = h
                    k = su.size
                    def nodes(q):
                        u = refn[:k] + kimae.decode(pu, q[:a.ns], iu, act) * su
                        v = refn[k:] + kimae.decode(pv, q[a.ns:], iv, act) * sv
                        return jnp.concatenate((u, jnp.zeros(1))), jnp.concatenate((v, jnp.zeros(1)))
                    u, v = nodes(z); up, vp = nodes(zp)
                    def rhs(w):
                        c, xm, xp, ym, yp = (w[loc[:, j]] for j in range(5))
                        return -u[loc[:, 0]] * (c - xm) / dx - v[loc[:, 0]] * (c - ym) / dx + (xm + xp + ym + yp - 4 * c) / (Re * dx * dx)
                    ru = u[loc[:, 0]] - up[loc[:, 0]] - dt * rhs(u)
                    rv = v[loc[:, 0]] - vp[loc[:, 0]] - dt * rhs(v)
                    return jnp.where(comp == 0, ru, rv)
                hr_res = lambda z, zp, h: h[-1] @ hr_raw(z, zp, h)
                # control: sampled rows of the full residual == sub-network residual (active paths are exact)
                kmid = a.nt // 2
                full_rows = np.asarray(nm_res(Z[kmid], Z[kmid - 1], args))[rows]
                sub_rows = np.asarray(hr_raw(Z[kmid], Z[kmid - 1], hargs))
                parity = float(np.max(np.abs(full_rows - sub_rows)) / max(np.max(np.abs(full_rows)), 1e-300))
                assert parity < 1e-9, parity
                hroll = jax.jit(lspg.make_rollout(hr_res, a.nt))
                Zh, (ith, _, _) = hroll(z0, hargs); Zh.block_until_ready()
                t0 = time.perf_counter(); Zh, _ = hroll(z0, hargs); Zh.block_until_ready(); th = time.perf_counter() - t0
                fh = np.asarray(decv(Zh, args))
                rec['hr'].append(dict(subnet_parity=parity, residual_basis=nr, samples=nz, nodes_evaluated=int(need.size),
                                      active_hidden=[subs[0][2], subs[1][2]], error=maxrel(fh, truth) if np.isfinite(fh).all() else None,
                                      seconds_warm=th, gn_iterations_mean=float(np.mean(ith))))
                print(f'NM-LSPG-HR seed={seed}', rec['hr'][-1], flush=True)
                if (nr, nz) == (a.hr[0], a.hr[1]):
                    np.savez(out / f'fields_seed{seed}_hr.npz', nm_lspg_hr=fh)
        results['seeds'].append(rec)
        (out / 'summary.json').write_text(json.dumps(results, indent=2, default=str) + '\n')

    errs = [r['nm_lspg']['error']['max'] for r in results['seeds'] if r['nm_lspg']['finite']]
    med = float(np.median(errs)) if errs else None
    results['gate'] = dict(rule='median over seeds of NM-LSPG max relative error <= 1.5 % AND LS-LSPG control >= 10 %',
                           nm_lspg_median=med, ls_lspg=results['ls_lspg']['error']['max'],
                           passed=bool(med is not None and med <= .015 and results['ls_lspg']['error']['max'] >= .10))
    results['source_sha256'] = {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in ('gate_kim2d.py', 'kimae.py', 'lspg.py')}
    results['provenance'] = dict(job_id=os.environ.get('SLURM_JOB_ID'), gpu=str(jax.devices()[0].device_kind), backend=backend,
                                 commit=os.environ.get('SOURCE_COMMIT'), jax=jax.__version__,
                                 matmul=os.environ.get('JAX_DEFAULT_MATMUL_PRECISION'))
    (out / 'summary.json').write_text(json.dumps(results, indent=2, default=str) + '\n')
    print('GATE', results['gate'], flush=True)


def neighbour_table(m):
    """(n,4) interior indices of x-1, x+1, y-1, y+1 neighbours in C order; -1 on the boundary."""
    i = np.arange(m * m); ix, iy = np.divmod(i, m)
    return np.stack((np.where(ix > 0, i - m, -1), np.where(ix < m - 1, i + m, -1),
                     np.where(iy > 0, i - 1, -1), np.where(iy < m - 1, i + 1, -1)), 1)


if __name__ == '__main__':
    main()
