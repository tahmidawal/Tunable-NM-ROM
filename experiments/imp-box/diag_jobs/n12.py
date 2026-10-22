"""imp-box diagnostic (no results used scientifically): why the real-size smokes (smk2-A2 and smk2-A4, H200, JAX 0.10.2)
flagged every fixed-step F-SPEC row at N = 12 as non-finite (stats[0] = nsteps - 1) while every other N and the adaptive
N = 12 rows were valid, and the first smoke (vmapped cohorts) had valid N = 12 rows.
  python n12.py nocache | cache <cachedir>
1) production path: spec.make_fom(N, n) unbatched over the 32 val cases (as ladder.run_unbatched_cohort), N in 8..24,
   n in 5, 40; per-case bad counts;
2) instrumented copy of make_fom's q (same nested scan / fori_loop structure, coefficients computed inside) returning,
   per step, the finiteness of every stage and the update, for the first failing (case, n);
3) independent nonlinear reference at the first failing stage input: dense NumPy sine/cosine evaluation on the same
   M = 3N/2 padded grid with trapezoid analysis (no FFT), compared with pscore.nonlinear on the device.
Cache evidence: number of files in the cache directory before / after (writes); a replay that writes nothing hit."""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, jax.numpy as jnp
mode = sys.argv[1]
cdir = sys.argv[2] if len(sys.argv) > 2 else None
if mode == 'cache':
    os.makedirs(cdir, exist_ok=True)
    jax.config.update('jax_compilation_cache_dir', cdir)
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 0.0)
    jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
else:
    jax.config.update('jax_enable_compilation_cache', False)
import geom as G, spec as SP
nfiles = lambda: len(os.listdir(cdir)) if cdir else None
res = dict(mode=mode, backend=jax.default_backend(), jax=jax.__version__, cache_files_before=nfiles())
Dv = G.cohorts('W40', 'A2')['val']
fail = None
for N in (8, 10, 12, 14, 16, 24):
    for n in (5, 40):
        q = SP.make_fom(N, n)
        bad = [float(q(jnp.asarray(d))[1][0]) for d in Dv]
        res[f'{N}|{n}'] = dict(cases_bad=int(np.sum(np.array(bad) > 0)), max_bad=max(bad))
        if fail is None and max(bad) > 0:
            fail = (N, n, int(np.argmax(np.array(bad) > 0)))
        print(N, n, res[f'{N}|{n}'], flush=True)
res['cache_files_after'] = nfiles()
if fail is None and os.environ.get('N12_FORCE'):
    fail = (12, 5, 0)                            # path test only
res['first_failure'] = fail
if fail:
    N, n, ci = fail
    d = jnp.asarray(Dv[ci])
    keep, dt = n // 5, G.T_END / n
    tab, lam, NL = SP.ICTable(N), SP.lam_grid(N), SP.nl_fun(N)

    def q(d):                                    # instrumented copy of spec.make_fom's q (same loop structure)
        c = SP._coefs(lam, d[4], dt)
        b0 = tab.coefs(d)

        def st(i, s):
            v, fl = s
            vn, stg = SP._etd(v, c, NL)
            f = jnp.stack([jnp.all(jnp.isfinite(x)) for x in stg + (vn,)])
            return vn, fl.at[i].set(f)

        def block(s, k):
            s = jax.lax.fori_loop(k * keep, (k + 1) * keep, st, s)
            return s, None
        s, _ = jax.lax.scan(block, (b0, jnp.ones((n, 4), bool)), jnp.arange(5))
        return s[1]
    fl = np.asarray(jax.jit(q)(d))
    res['stage_finite_per_step'] = fl.tolist()
    # independent NL at the input of the first failing step (states from an eager step loop, same coefficients)
    c = SP._coefs(lam, d[4], dt)
    v = tab.coefs(d)
    k0 = int(np.argmin(fl.all(axis=1))) if not fl.all() else 0
    for i in range(k0):
        v = SP._etd(v, c, NL)[0]
    M = 3 * N // 2
    x = np.arange(1, M) / M
    kk = np.arange(1, N)
    S = np.sin(np.pi * np.outer(x, kk)); C = np.cos(np.pi * np.outer(x, kk)) * np.pi * kk
    b = np.asarray(v)
    u = S @ b @ S.T
    g = C @ b @ S.T + S @ b @ C.T
    Sa = (2.0 / M) * np.sin(np.pi * np.outer(kk, x))
    ref = -(Sa @ (u * g) @ Sa.T)
    dev = np.asarray(jax.jit(NL)(jnp.asarray(b)))
    res['first_failing_step'] = k0
    res['NL_device_vs_independent'] = float(np.max(np.abs(dev - ref)) / (np.max(np.abs(ref)) + 1e-300))
    res['NL_device_finite'] = bool(np.all(np.isfinite(dev)))
    res['eager_step_finite'] = [bool(jnp.all(jnp.isfinite(x))) for x in SP._etd(v, c, NL)[1]]
print('RESULT', json.dumps(res), flush=True)
