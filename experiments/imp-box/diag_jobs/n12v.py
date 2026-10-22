"""imp-box diagnostic (no results used scientifically): loop-structure variants of the fixed-step F-SPEC query on H200,
after n12.py (job 6188970) reproduced the N = 12 failure without any cache (fused loop non-finite from step 1, eager
steps finite, nonlinear term equal to an independent dense evaluation to 2.5e-15). For each variant and N, the 32 val cases
at n = 5, 40: bad counts and the max difference from an eager reference (Python loop of a jitted single step)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, jax.numpy as jnp
jax.config.update('jax_enable_compilation_cache', False)
import geom as G, spec as SP
Dv = G.cohorts('W40', 'A2')['val']


def variant(N, n, kind):
    keep, dt = n // 5, G.T_END / n
    tab, lam, NL, syn = SP.ICTable(N), SP.lam_grid(N), SP.nl_fun(N), SP.synth_fn(N)

    def q(d):
        c = SP._coefs(lam, d[4], dt)
        b0 = tab.coefs(d)

        def st(_, s):
            v, bad = s
            vn, stg = SP._etd(v, c, NL)
            return vn, bad + (~SP._finite_all(stg + (vn,))).astype(jnp.float64)
        if kind == 'static_fori':
            def block(s, _):
                s = jax.lax.fori_loop(0, keep, st, s)
                return s, syn(s[0])
            s, outs = jax.lax.scan(block, (b0, 0.0), None, length=5)
        elif kind == 'scan_steps':
            def block(s, _):
                s, _ = jax.lax.scan(lambda s_, __: (st(0, s_), None), s, None, length=keep)
                return s, syn(s[0])
            s, outs = jax.lax.scan(block, (b0, 0.0), None, length=5)
        elif kind == 'unroll2':
            def block(s, _):
                s = jax.lax.fori_loop(0, keep, st, s, unroll=2)
                return s, syn(s[0])
            s, outs = jax.lax.scan(block, (b0, 0.0), None, length=5)
        elif kind == 'barrier':
            def stb(i, s):
                v, bad = s
                vn, stg = SP._etd(v, c, NL)
                vn = jax.lax.optimization_barrier(vn)
                return vn, bad + (~SP._finite_all(stg + (vn,))).astype(jnp.float64)

            def block(s, _):
                s = jax.lax.fori_loop(0, keep, stb, s)
                return s, syn(s[0])
            s, outs = jax.lax.scan(block, (b0, 0.0), None, length=5)
        elif kind == 'unrolled_blocks':
            s, outl = (b0, 0.0), []
            for k in range(5):
                s = jax.lax.fori_loop(0, keep, st, s)
                outl.append(syn(s[0]))
            outs = jnp.stack(outl)
        return jnp.concatenate([syn(b0)[None], outs]), s[1]
    return jax.jit(q)


def eager(N, n, d):
    keep, dt = n // 5, G.T_END / n
    lam, NL, syn = SP.lam_grid(N), SP.nl_fun(N), SP.synth_fn(N)
    c = SP._coefs(lam, d[4], dt)
    stp = jax.jit(lambda v: SP._etd(v, c, NL)[0])
    v = SP.ICTable(N).coefs(d)
    outs = [syn(v)]
    for k in range(5):
        for i in range(keep):
            v = stp(v)
        outs.append(syn(v))
    return np.asarray(jnp.stack(outs))


res = dict(device=str(jax.devices()[0]), jax=jax.__version__)
for kind in ('production', 'unroll2', 'barrier', 'M2N'):
    for N in (10, 12, 14, 16, 24, 48):
        for n in (5, 40):
            q = SP.make_fom(N, n) if kind == 'production' else (SP.make_fom(N, n, M=2 * N) if kind == 'M2N' else variant(N, n, kind))
            bad, diff = 0, 0.0
            for j, d in enumerate(Dv[:8]):
                O, b = q(jnp.asarray(d))
                b = float(b[0]) if kind in ('production', 'M2N') else float(b)
                bad += int(b > 0)
                if j < 2:
                    E = eager(N, n, jnp.asarray(d))
                    dd = float(np.max(np.abs(np.asarray(O) - E)) / np.max(np.abs(E)))
                    diff = dd if not np.isfinite(dd) else max(diff, dd)
            res[f'{kind}|{N}|{n}'] = dict(bad=bad, diff_vs_eager=diff)
            print(kind, N, n, res[f'{kind}|{N}|{n}'], flush=True)
print('RESULT', json.dumps(res), flush=True)
