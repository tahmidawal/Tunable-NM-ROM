"""One-allocation comparison panel for heat 2D at one mesh (DESIGN.md).

Arm-major blocks: every arm is built, warmed on every case (untimed; errors, solver statistics and
audit fields saved), then timed case by case (repetitions, each after a GPU burn-in). A sentinel
(the exact DST propagation on case 0) is timed between blocks; after every full-order block a
deliberately contaminated sentinel (FOM dispatched without synchronising) is timed as the
positive control of the order-effect gate.
"""
from __future__ import annotations

import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

import numpy as np
import scipy.linalg

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'hires-heat'))
import core as C   # noqa: E402  (enables jax x64)
import jax, jax.numpy as jnp   # noqa: E402
import podfac as P   # noqa: E402


def gpu_identity():
    try:
        return subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,memory.total,driver_version', '--format=csv,noheader'],
                              capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as exc:
        return f'unavailable: {exc}'


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Method:
    """prep(u0 jax) -> input (untimed); run(input) -> output, synchronised (timed);
    fields(output) -> jax f64 [6, n-1, n-1]; stats(output) -> list (JSON-able)."""
    def __init__(self, run, prep=None, fields=None, stats=None, sync=None):
        self.run_raw = run; self.prep = prep or (lambda u: u)
        self.fields = fields or (lambda o: o[0]); self.stats = stats or (lambda o: [np.asarray(s).tolist() for s in o[1:]])
        self.sync = sync or C.block

    def run(self, x):
        out = self.run_raw(x); self.sync(out); return out


# ------------------------------------------------------------------ builders
def vmap_lap(n):
    return jax.jit(jax.vmap(lambda c: C.negative_laplacian(c, n), in_axes=-1, out_axes=-1))


def build_fields(ux, uy, core_cols, n, rows=256):
    """[rho^2, k] core vectors -> [N, k] device fields (row-major, hires-heat bank ordering), built row-block by
    row-block into ONE donated buffer (peak = the bank + one block; no transposed or concatenated copy)."""
    rx, ry = ux.shape[1], uy.shape[1]; k = core_cols.shape[1]; m = jnp.asarray(core_cols.reshape(rx, ry, k)); b = jnp.asarray(uy)
    f = jax.jit(lambda a, m, b: jnp.einsum('ia,abk,jb->ijk', a, m, b).reshape(-1, k))
    put = jax.jit(lambda buf, blk, start: jax.lax.dynamic_update_slice(buf, blk, (start, 0)), donate_argnums=0)
    buf = jnp.zeros(((n - 1) ** 2, k))
    for i0 in range(0, n - 1, rows):
        a = jnp.asarray(ux[i0:min(i0 + rows, n - 1)])
        if a.shape[0] < rows:   # keep one compiled shape: pad the last block, write only its valid rows
            blk = f(jnp.pad(a, ((0, rows - a.shape[0]), (0, 0))), m, b)[:a.shape[0] * (n - 1)]
        else:
            blk = f(a, m, b)
        buf = put(buf, blk, i0 * (n - 1))
    return C.block(buf)


def gram_t(A, B, chunk=1 << 19):
    """A^T B streamed over row chunks (never materialises a transposed copy of a tall basis)."""
    out = 0.
    for s in range(0, A.shape[0], chunk):
        out = out + np.asarray(A[s:s + chunk].T @ B[s:s + chunk])
    return out


def pod_operators(V, n, nu, dt, chunk=32):
    """K = V^T L V and G2 = (L V)^T (L V) on the grid, chunk by chunk (L = -Delta_h, 5-point)."""
    lap = vmap_lap(n); r = V.shape[1]; K = np.zeros((r, r)); G2 = np.zeros((r, r))
    shape = (n - 1, n - 1)
    blocks = [(s, min(s + chunk, r)) for s in range(0, r, chunk)]
    for i, (a, b) in enumerate(blocks):
        Li = lap(V[:, a:b].reshape(shape + (b - a,))).reshape(-1, b - a)
        K[:, a:b] = gram_t(V, Li)
        for c, d in blocks[i:]:
            Lj = Li if c == a else lap(V[:, c:d].reshape(shape + (d - c,))).reshape(-1, d - c)
            G2[a:b, c:d] = gram_t(Li, Lj); G2[c:d, a:b] = G2[a:b, c:d].T
            del Lj
        del Li
    return 0.5 * (K + K.T), G2, float(np.abs(K - K.T).max() / np.abs(K).max())


def make_pod_query(n, times, S=None, maps=None, dt=0.025):
    nout = len(times) - 1; stride = int(round((times[1] - times[0]) / dt))
    if maps is not None:
        Mj = jnp.asarray(maps)
        @jax.jit
        def q(u0, V):
            a0 = V.T @ u0.reshape(-1)
            return ((Mj @ a0) @ V.T).reshape((len(times), n - 1, n - 1)),
        return q
    Sj = jnp.asarray(S)
    @jax.jit
    def q(u0, V):
        a0 = V.T @ u0.reshape(-1)
        def step(a, _):
            a = Sj @ a; return a, a
        _, traj = jax.lax.scan(step, a0, None, length=stride * nout)
        A = jnp.concatenate((a0[None], traj[stride - 1::stride]))
        return (A @ V.T).reshape((len(times), n - 1, n - 1)),
    return q


def linear_bank_cn(setup, dt=0.025):
    """LABELLED BASELINE: free bank coefficients, the NM-ROM's weak Crank-Nicolson step (M tests), moments init."""
    a = setup['a']; left = jnp.asarray(np.linalg.pinv(a)); aj = jnp.asarray(a)
    lam = jnp.asarray(setup['mode_lam']); nu = setup['nu']; times = setup['times']; n, d, modes = setup['n'], setup['d'], setup['modes']
    factor = (1 - dt * nu * lam / 2) / (1 + dt * nu * lam / 2); stride = int(round((times[1] - times[0]) / dt)); nout = len(times) - 1
    @jax.jit
    def query(u0, bank):
        c0 = left @ C.moments(u0, modes, n, d)
        def step(c, _):
            c = left @ (factor * (aj @ c)); return c, c
        _, traj = jax.lax.scan(step, c0, None, length=stride * nout)
        coefs = jnp.concatenate((c0[None], traj[stride - 1::stride]))
        return (coefs @ bank.T).reshape((len(times),) + (n - 1,) * d),
    return query


def qm_head(r):
    i, j = np.triu_indices(r)
    def head(params, z):
        one = jnp.ones(z.shape[:-1] + (1,), z.dtype)
        return jnp.concatenate((one, z, z[..., i] * z[..., j]), -1)
    return head


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True); ap.add_argument('--out', required=True)
    args = ap.parse_args(); cfg = json.loads(Path(args.config).read_text()); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    smoke = cfg.get('local_smoke', False)
    assert jax.default_backend() == 'gpu' and jax.config.jax_enable_x64
    assert os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest'
    print('jax_backend=gpu x64=True precision=highest', flush=True)
    n = cfg['mesh']; d = 2; times = cfg['times']; nu = cfg['diffusivity']; tj = jnp.asarray(times); N = (n - 1) ** 2
    sources = {p.name: sha_file(p) for p in sorted(HERE.glob('*.py'))}
    sources.update({'hires-heat/' + p.name: sha_file(p) for p in sorted((HERE.parent / 'hires-heat').glob('*.py'))})
    sources.update({'ops/' + p.name: sha_file(p) for p in sorted((HERE / 'ops').glob('*.py'))})
    sources[Path(args.config).name] = sha_file(args.config)
    result = dict(schema='heat-compare-panel-v1', config=cfg, source_sha256=sources, source_commit=os.environ.get('SOURCE_COMMIT'),
                  complete=False, mesh=n, unknowns=N, arms={}, blocks=[], sentinels=[], contaminated=[], gates={},
                  metadata=dict(job_id=os.environ.get('SLURM_JOB_ID'), node=os.environ.get('SLURMD_NODENAME'), gpu=gpu_identity(),
                                jax=jax.__version__, x64=True, precision='highest', local_smoke=smoke),
                  query_contract='Supplied full initial field (interior grid, f64) on the GPU to all six requested full fields on the GPU, '
                                 'synchronised. Offline assembly and compilation excluded and recorded per arm. Host transfers excluded for every method.')
    dump = lambda: C.dump(out / 'results.json', result)
    draws = np.concatenate([C.family('mr2d', s, c) for s, c in cfg['cohorts']])
    train = C.family('mr2d', *cfg['train']); val = C.family('mr2d', *cfg['validation'])
    assert not any(np.array_equal(a, b) for a in np.concatenate((train, val)) for b in draws), 'training/validation overlaps evaluation'
    result['draws_sha256'] = C.sha(draws); result['train_sha256'] = C.sha(train)
    prop = C.make_propagate(d); lam_d, lam_c = C.eig_grid(n, d), C.eig_grid(n, d, True)
    stride_a = max(1, n // cfg['audit_intervals']); sub = (slice(None), slice(stride_a - 1, None, stride_a), slice(stride_a - 1, None, stride_a))
    reps = cfg['repetitions']; burn = cfg['burn_seconds']
    fields_dir = out / 'fields'; fields_dir.mkdir(exist_ok=True)
    u_case0 = C.block(C.initial_grid(n, d, draws[0]))
    sentinel_fn = jax.jit(lambda u: prop(u, lam_d, tj, nu))
    msz = cfg['control_matmul_size']
    def _chain(x):
        for _ in range(8):
            x = jnp.tanh(x @ x / msz)
        return x
    heavy = jax.jit(_chain); heavy_x = C.block(jnp.eye(msz) + 1e-3); C.block(heavy(heavy_x))

    def sentinel(label):
        C.block(sentinel_fn(u_case0)); ts = []
        for _ in range(cfg['sentinel_reps']):
            C.burn(burn); t = time.perf_counter(); C.block(sentinel_fn(u_case0)); ts.append(1e3 * (time.perf_counter() - t))
        result['sentinels'].append(dict(after=label, ms=ts, median_ms=float(np.median(ts))))

    def idle():
        time.sleep(cfg['idle_seconds']); C.burn(burn)

    def run_block(name, method, meta):
        begin = time.perf_counter()
        row = dict(meta, name=name, same=[], physical=[], same_sub=[], physical_sub=[], device_ms=[], warm_seconds=[], stats=[])
        rng_seed = cfg['audit_sample_seed']
        for ci, draw in enumerate(draws):
            u0 = C.block(C.initial_grid(n, d, draw)); same = C.block(prop(u0, lam_d, tj, nu)); phys = C.block(prop(u0, lam_c, tj, nu))
            x = method.prep(u0)
            t = time.perf_counter(); o = method.run(x); row['warm_seconds'].append(time.perf_counter() - t)
            f = method.fields(o)
            assert f.dtype == jnp.float64 and f.shape == (len(times), n - 1, n - 1) and bool(jnp.isfinite(f).all()), (name, f.dtype, f.shape)
            row['same'].append(np.asarray(C.rel_errors(f, same)).tolist()); row['physical'].append(np.asarray(C.rel_errors(f, phys)).tolist())
            row['same_sub'].append(np.asarray(C.rel_errors(f[sub], same[sub])).tolist())
            row['physical_sub'].append(np.asarray(C.rel_errors(f[sub], phys[sub])).tolist())
            row['stats'].append(method.stats(o))
            saved = dict(draw=draw, stride=stride_a, sub=np.asarray(f[sub]))
            if ci < cfg['random_audit_cases']:
                idx = np.sort(np.random.default_rng(rng_seed + ci).choice(N, min(cfg['random_audit_nodes'], N), replace=False))
                saved['rand'] = np.asarray(f.reshape(len(times), -1)[:, jnp.asarray(idx)]); saved['rand_idx'] = idx
            np.savez(fields_dir / f'{name}__case{ci}.npz', **saved)
            del o
            ms, parity = [], 0.0
            for _ in range(reps):
                C.burn(burn); t = time.perf_counter(); o = method.run(x); ms.append(1e3 * (time.perf_counter() - t))
                parity = max(parity, float(jnp.max(jnp.abs(method.fields(o) - f)) / jnp.max(jnp.abs(f))))   # outside the timed region
                del o
            row['device_ms'].append(ms); row.setdefault('timed_vs_warm_max_relative', []).append(parity)
            del f
            del u0, same, phys, x
        row['block_seconds'] = time.perf_counter() - begin
        row['device_ms_median'] = float(np.median(row['device_ms']))
        row['worst_all_times'] = float(np.max(row['same']))
        try:
            row['device_bytes_in_use_after'] = int(jax.devices()[0].memory_stats().get('bytes_in_use', -1))
            row['device_peak_bytes'] = int(jax.devices()[0].memory_stats().get('peak_bytes_in_use', -1))
        except Exception:
            pass
        result['arms'][name] = row; result['blocks'].append(name)
        print('block', name, 'worst%', round(100 * row['worst_all_times'], 5), 'ms', round(row['device_ms_median'], 3), flush=True)
        dump()

    sentinel('start')
    tests = C.mode_list(cfg['tests'], n, d)

    # ---------------- NM-ROM + linear-bank baselines (frozen wide2d)
    if cfg.get('nmrom'):
        t0 = time.perf_counter()
        model = C.load_model(cfg['nmrom']['model'], HERE.parent / 'hires-heat' / 'inputs')
        bank = C.bank_at(model, n); rtri = C.tsqr_r(bank); a = C.weak_matrix(bank, tests, n, d)
        setup = dict(n=n, d=d, times=times, nu=nu, modes=tests, a=a, rtri=rtri, mode_lam=C.mode_eigs(n, tests), directions=model['directions'])
        result['nmrom_setup'] = dict(seconds=time.perf_counter() - t0, model_sha256=model['sha256'], bank_rank=int(bank.shape[1]),
                                     bank_condition=float(np.linalg.cond(rtri)))
        for arm in cfg['nmrom']['arms']:
            t0 = time.perf_counter(); opt = {**cfg['nmrom']['defaults'], **arm.get('opt', {})}
            st = C.make_stages(model, setup, arm['q'], opt)
            run_block(arm['name'], Method(lambda u, st=st: st['query'](u, bank)),
                      dict(family='nmrom', unknowns=int(model['codes'].shape[1] + arm['q']), q=arm['q'], opt=opt, setup_seconds=time.perf_counter() - t0))
            idle(); sentinel(arm['name'])
        for init in cfg['nmrom'].get('linear_bank_inits', []):
            lb = C.linear_bank(setup, init)
            run_block(f'linear_bank_{init}_BASELINE', Method(lambda u, lb=lb: (lb(u, bank),)),
                      dict(family='linear_bank', unknowns=int(bank.shape[1]), init=init, stepping='exact reduced weak evolution'))
            idle(); sentinel(f'linear_bank_{init}_BASELINE')
        if cfg['nmrom'].get('linear_bank_cn'):
            lbc = linear_bank_cn(setup)
            run_block('linear_bank_moments_cn_BASELINE', Method(lambda u: lbc(u, bank)),
                      dict(family='linear_bank', unknowns=int(bank.shape[1]), init='moments', stepping='weak CN dt 0.025 (M tests)'))
            idle(); sentinel('linear_bank_moments_cn_BASELINE')
        del bank, model, setup; jax.clear_caches()

    # ---------------- factor model of the training snapshots at this mesh
    if cfg.get('pod') or cfg.get('qm'):
        t0 = time.perf_counter(); fm = P.factor_model(n, train)
        # gate: factor representation reproduces directly generated training snapshots
        gate = []
        for j in cfg['factor_gate_draws']:
            u = np.asarray(prop(C.initial_grid(n, d, train[j]), lam_d, tj, nu))
            rep = P.field(fm, fm['C'][:, j * len(times):(j + 1) * len(times)])
            gate.append(float(np.linalg.norm(u - rep) / np.linalg.norm(u)))
        result['factor_model'] = dict(rho=list(fm['rho']), factor_residual=fm['factor_residual'], snapshot_gate_max=max(gate),
                                      seconds=time.perf_counter() - t0, snapshots=int(fm['C'].shape[1]))
        assert max(gate) < 1e-12 and fm['factor_residual'] < 1e-12, result['factor_model']
        dump()

    # ---------------- POD-Galerkin / POD-LSPG (uncentred POD of the training snapshots at this mesh)
    if cfg.get('pod'):
        rmax = max(cfg['pod']['ranks']); Ufull, sv = P.pod(fm['C'], rmax)
        result['pod'] = dict(singular_values=sv[:rmax + 1].tolist(), ranks={})
        for r in cfg['pod']['ranks']:
            t0 = time.perf_counter(); V = build_fields(fm['Ux'], fm['Uy'], Ufull[:, :r], n)
            orth = float(np.max(np.abs(gram_t(V, V) - np.eye(r))))
            K, G2, asym = pod_operators(V, n, nu, cfg['pod']['dt'])
            c = cfg['pod']['dt'] * nu / 2; I = np.eye(r)
            SG = np.linalg.solve(I + c * K, I - c * K); SL = np.linalg.solve(I + 2 * c * K + c * c * G2, I - c * c * G2)
            maps = np.stack([scipy.linalg.expm(-nu * t * K) for t in times])
            setup_s = time.perf_counter() - t0
            result['pod']['ranks'][str(r)] = dict(orthonormality=orth, K_asymmetry=asym, setup_seconds=setup_s,
                                                  spectral_radius_galerkin=float(np.max(np.abs(np.linalg.eigvals(SG)))),
                                                  spectral_radius_lspg=float(np.max(np.abs(np.linalg.eigvals(SL)))))
            assert orth < 1e-10, (r, orth)
            for kind, q in (('galerkin_cn', make_pod_query(n, times, S=SG)), ('lspg_cn', make_pod_query(n, times, S=SL)),
                            ('galerkin_exact', make_pod_query(n, times, maps=maps))):
                if kind not in cfg['pod']['methods']:
                    continue
                run_block(f'pod{r}_{kind}', Method(lambda u, q=q: q(u, V)), dict(family='pod', unknowns=r, method=kind, setup_seconds=setup_s))
                idle(); sentinel(f'pod{r}_{kind}')
            del V; jax.clear_caches()

    # ---------------- quadratic manifold
    if cfg.get('qm'):
        result['qm'] = {}
        for r in cfg['qm']['ranks']:
            t0 = time.perf_counter()
            gamma, sel = P.qm_select(fm['C'], fm['traj'], r, seed=cfg['qm']['split_seed'], fraction=cfg['qm']['holdout_fraction'])
            uref, Vc, W, A, st = P.qm_fit(fm['C'], r, gamma)
            core_cols = np.concatenate((uref[:, None], Vc, W), 1)
            B = build_fields(fm['Ux'], fm['Uy'], core_cols, n)
            rt = np.linalg.qr(core_cols, mode='r'); rt = rt * np.where(np.diag(rt) < 0, -1., 1.)[:, None]
            gram = gram_t(B, B); rgate = float(np.linalg.norm(gram - rt.T @ rt) / np.linalg.norm(gram)); del gram
            a = C.weak_matrix(B, tests, n, d)
            head = qm_head(r)
            model = dict(d=d, head_fn=head, head_params={}, codes=jnp.asarray(A.T), directions=np.zeros((core_cols.shape[1], 0)))
            setup = dict(n=n, d=d, times=times, nu=nu, modes=tests, a=a, rtri=rt, mode_lam=C.mode_eigs(n, tests), directions=model['directions'])
            setup_s = time.perf_counter() - t0
            result['qm'][str(r)] = dict(gamma=gamma, selection=sel, fit=st, rtri_gram_gate=rgate, columns=int(core_cols.shape[1]),
                                        setup_seconds=setup_s, bank_bytes=int(B.size * 8))
            assert rgate < 1e-10, rgate
            dump()
            for arm in cfg['qm']['arms']:
                opt = {**cfg['nmrom']['defaults'], **arm.get('opt', {})}
                stq = C.make_stages(model, setup, 0, opt)
                run_block(f'qm{r}_{arm["name"]}', Method(lambda u, stq=stq: stq['query'](u, B)),
                          dict(family='qm', unknowns=r, opt=opt, gamma=gamma, setup_seconds=setup_s))
                idle(); sentinel(f'qm{r}_{arm["name"]}')
            del B, model, setup; jax.clear_caches()

    # ---------------- labelled controls (not FOM candidates)
    run_block('dst_exact_CONTROL', Method(lambda u: (prop(u, lam_d, tj, nu),)), dict(family='control', unknowns=N))
    idle(); sentinel('dst_exact_CONTROL')
    for nc in cfg.get('coarse_intervals', []):
        cq = C.make_coarse(n, nc, d, times, cfg['cg_arms'][cfg['coarse_cg']], nu)
        run_block(f'coarse{nc}_{cfg["coarse_cg"]}_CONTROL', Method(cq), dict(family='control', unknowns=(nc - 1) ** 2))
        idle(); sentinel(f'coarse{nc}')

    # ---------------- neural operators (PyTorch, same process, same GPU)
    if cfg.get('operators'):
        import torch
        sys.path.insert(0, str(HERE / 'ops'))
        import heatops as H
        result['torch_environment'] = H.configure()
        for name, spec in cfg['operators'].items():
            path = Path(spec['checkpoint'])
            if not path.exists():
                result.setdefault('operators_missing', []).append(dict(name=name, checkpoint=str(path))); continue
            net, norm, ck = H.load(path)
            prov = json.loads((path.parent.parent / 'provenance.json').read_text())
            assert ck['mesh'] == n and prov['mesh'] == n, ('operator trained at another mesh', ck['mesh'], n)
            assert list(prov['train']) == list(cfg['train']) and list(prov['validation']) == list(cfg['validation']), prov
            assert ck['family'] == spec['family'], (ck['family'], spec['family'])
            sync = lambda o: torch.cuda.synchronize()
            def run(x, net=net, norm=norm):
                with torch.no_grad():
                    return H.query_interior(net, x, *norm)
            m = Method(run, prep=lambda u: torch.from_dlpack(u), fields=lambda o: jnp.from_dlpack(o), stats=lambda o: [], sync=sync)
            run_block(name, m, dict(family='operator', unknowns=None, checkpoint=str(path), checkpoint_sha256=sha_file(path),
                                    training_provenance=dict(job_id=prov.get('job_id'), train=prov['train'], validation=prov['validation'],
                                                             wall_seconds=prov['wall_seconds']),
                                    operator_family=ck['family'], best_epoch=int(ck['epoch']), best_step=int(ck['step']),
                                    parameters=int(sum(p.numel() * (2 if p.is_complex() else 1) for p in net.parameters())),
                                    parameter_dtype=str(getattr(net, 'parameter_dtype', torch.float64))))
            del net; torch.cuda.empty_cache(); idle(); sentinel(name)

    # ---------------- full-order candidate grid, fastest first
    for name in cfg['fom_order']:
        spec = cfg['cg_arms'][name]; q = C.make_cg(n, times, spec, nu)
        run_block(name, Method(q), dict(family='fom', unknowns=N, dt=spec['dt'], rtol=spec['tolerance'], named=name.endswith('_NAMED')))
        # positive control of the order gate: the sentinel timed behind deliberately queued, unsynchronised GPU work
        # (a jitted chain of dense matmuls, which dispatches asynchronously; a while-loop FOM call does not).
        C.burn(burn); pending = heavy(heavy_x); t = time.perf_counter(); C.block(sentinel_fn(u_case0))
        tc = 1e3 * (time.perf_counter() - t); C.block(pending)
        result['contaminated'].append(dict(after=name, ms=tc))
        idle(); sentinel(name)

    # ---------------- A2: re-time real arms AFTER the slow full-order phase (direct carry-over test)
    rt = cfg.get('retime', {})
    if rt.get('nmrom'):
        model = C.load_model(cfg['nmrom']['model'], HERE.parent / 'hires-heat' / 'inputs')
        bank = C.bank_at(model, n); rtri = C.tsqr_r(bank); a = C.weak_matrix(bank, tests, n, d)
        setup = dict(n=n, d=d, times=times, nu=nu, modes=tests, a=a, rtri=rtri, mode_lam=C.mode_eigs(n, tests), directions=model['directions'])
        arm = next(x for x in cfg['nmrom']['arms'] if x['name'] == rt['nmrom'])
        st = C.make_stages(model, setup, arm['q'], {**cfg['nmrom']['defaults'], **arm.get('opt', {})})
        run_block(rt['nmrom'] + '__RETIME', Method(lambda u: st['query'](u, bank)), dict(family='nmrom', retime_of=rt['nmrom'], unknowns=None))
        idle(); sentinel(rt['nmrom'] + '__RETIME')
        del bank, model, setup, st; jax.clear_caches()
    if rt.get('pod'):
        r = rt['pod']; V = build_fields(fm['Ux'], fm['Uy'], Ufull[:, :r], n)
        K, G2, _ = pod_operators(V, n, nu, cfg['pod']['dt']); c = cfg['pod']['dt'] * nu / 2; I = np.eye(r)
        q = make_pod_query(n, times, S=np.linalg.solve(I + c * K, I - c * K))
        run_block(f'pod{r}_galerkin_cn__RETIME', Method(lambda u: q(u, V)), dict(family='pod', retime_of=f'pod{r}_galerkin_cn', unknowns=None))
        idle(); sentinel(f'pod{r}_galerkin_cn__RETIME'); del V; jax.clear_caches()
    if rt.get('operator') and rt['operator'] in cfg.get('operators', {}) and Path(cfg['operators'][rt['operator']]['checkpoint']).exists():
        net, norm, ck = H.load(Path(cfg['operators'][rt['operator']]['checkpoint']))
        def run(x):
            with torch.no_grad():
                return H.query_interior(net, x, *norm)
        run_block(rt['operator'] + '__RETIME', Method(run, prep=lambda u: torch.from_dlpack(u), fields=lambda o: jnp.from_dlpack(o), stats=lambda o: [],
                                                      sync=lambda o: torch.cuda.synchronize()), dict(family='operator', retime_of=rt['operator'], unknowns=None))
        del net; torch.cuda.empty_cache(); idle(); sentinel(rt['operator'] + '__RETIME')

    # ---------------- order-effect gate
    med = np.array([s['median_ms'] for s in result['sentinels']]); ref = float(np.median(med))
    dev = np.abs(med / ref - 1)
    cont = np.array([c['ms'] for c in result['contaminated']]) / ref - 1 if result['contaminated'] else np.array([])
    result['gates']['order_effect'] = dict(reference_ms=ref, max_relative_deviation=float(dev.max()), tolerance=cfg['order_tolerance'],
                                           passed=bool(dev.max() <= cfg['order_tolerance']),
                                           positive_control_min_relative_deviation=float(cont.min()) if cont.size else None,
                                           positive_control_fails_as_required=bool(cont.size and cont.min() > cfg['order_tolerance']))
    result['complete'] = True; dump(); print('HEAT COMPARE PANEL COMPLETE', json.dumps(result['gates']), flush=True)


if __name__ == '__main__':
    main()
