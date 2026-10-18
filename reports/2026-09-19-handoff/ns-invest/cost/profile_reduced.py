"""Small-scale profile of ONE reduced NS query's building blocks (local GB10, read-only).

Validates the STRUCTURE of the cost model in report.md, not A100 absolute numbers:
  (a) residual cost is independent of q and scales with M*R^2 (one streaming pass over Q);
  (b) jax.jacfwd of the step residual scales with M*R^2*(K+q) (the tensor is re-streamed once
      per tangent batch and the contraction is a batched (R,R)@(R,T) matmul);
  (c) an analytic Jacobian that reuses X = Q@cm costs one pass + M*R*T;
  (d) f32 tensor halves the streaming cost;
  (e) a pre-contracted (M,d,d) tensor at d = K+q is cheap.
Random data throughout (timing only). Every large array is a jit ARGUMENT.
"""
import json, os, sys, time
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

WT = '/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-17-ns2d/experiments/ns2d'
sys.path.insert(0, WT)
import ns2d_rom as RM  # noqa: E402  (read-only import; PYTHONDONTWRITEBYTECODE=1)

OUT = os.path.dirname(os.path.abspath(__file__))
print('backend', jax.default_backend(), jax.devices()[0].device_kind, flush=True)
assert jax.default_backend() == 'gpu'

DT, NU = 2e-3, 3e-3
REPS = 12


def timeit(f, *args, reps=REPS):
    out = f(*args); jax.block_until_ready(out)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); out = f(*args); jax.block_until_ready(out)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1e3, float(np.min(ts)) * 1e3


def make_case(M, R, K, q, seed=0, dtype=jnp.float64):
    rng = np.random.default_rng(seed)
    T = rng.standard_normal((M, R, R)) / R
    Q = jnp.asarray(T + T.swapaxes(1, 2), dtype)
    A = jnp.asarray(rng.standard_normal((M, R)) / np.sqrt(R), dtype)
    lam = jnp.asarray(np.sort(rng.uniform(40, 4e4, M)), dtype)
    W1 = jnp.asarray(rng.standard_normal((128, K)) / np.sqrt(K), dtype)
    W2 = jnp.asarray(rng.standard_normal((R, 128)) / np.sqrt(128), dtype)
    S = jnp.asarray(rng.standard_normal((R, K)) / np.sqrt(K), dtype)
    Cq = jnp.asarray(np.linalg.qr(rng.standard_normal((R, max(q, 1))))[0][:, :q], dtype)
    w = jnp.asarray(rng.standard_normal(K + q) * 0.1, dtype)
    c_prev = jnp.asarray(rng.standard_normal(R) * 0.1, dtype)
    return dict(Q=Q, A=A, lam=lam, W1=W1, W2=W2, S=S, Cq=Cq, w=w, c_prev=c_prev, K=K, q=q, M=M, R=R)


def head_fn(K):
    def head(w, W1, W2, S, Cq):
        z = w[:K]
        return W2 @ jnp.tanh(W1 @ z) + S @ z + Cq @ w[K:]
    return head


results = []


def record(**kw):
    results.append(kw)
    print(' '.join(f'{k}={v}' for k, v in kw.items()), flush=True)


# ---------------------------------------------------------------- bandwidth reference
for R, M in ((512, 544), (256, 544), (512, 128)):
    cs = make_case(M, R, 32, 0)
    Q = cs['Q']
    t_sum, _ = timeit(jax.jit(lambda Q: jnp.sum(Q)), Q)
    gb = Q.nbytes / 1e9
    record(kind='stream_sum', M=M, R=R, ms=round(t_sum, 3), GB=round(gb, 3), GBps=round(gb / t_sum * 1e3, 1))

# ---------------------------------------------------------------- residual vs q, R, M, dtype
for (M, R, K, q, dtype) in ((544, 512, 32, 0, jnp.float64), (544, 512, 32, 128, jnp.float64), (544, 512, 32, 512, jnp.float64),
                            (544, 256, 32, 0, jnp.float64), (128, 512, 32, 0, jnp.float64), (544, 512, 32, 0, jnp.float32)):
    cs = make_case(M, R, K, q, dtype=dtype)
    head = head_fn(K)
    weak = RM.make_weak(cs['A'], cs['Q'], cs['lam'], DT)   # THE lane's residual, verbatim

    def fun_step(w, c_prev, nu, A, Q, lam, W1, W2, S, Cq):
        return RM.make_weak(A, Q, lam, DT)(head(w, W1, W2, S, Cq), c_prev, nu)

    args = (cs['w'], cs['c_prev'], NU, cs['A'], cs['Q'], cs['lam'], cs['W1'], cs['W2'], cs['S'], cs['Cq'])
    t_res, _ = timeit(jax.jit(fun_step), *args)
    record(kind='residual', M=M, R=R, K=K, q=q, T=K + q, dtype=str(dtype.__name__ if hasattr(dtype, '__name__') else dtype),
           ms=round(t_res, 3), GB=round(cs['Q'].nbytes / 1e9, 3))
    if dtype is jnp.float64 and M == 544 and R == 512:
        # jacfwd exactly as make_lm does it
        jf = jax.jit(lambda *a: jax.jacfwd(fun_step)(*a))
        t_jf, _ = timeit(jf, *args)
        record(kind='jacfwd', M=M, R=R, K=K, q=q, T=K + q, ms=round(t_jf, 3),
               flop_GF=round(2 * M * R * R * (K + q) / 1e9, 2))

        # analytic Jacobian reusing X = Q @ cm (Q symmetric): dq/dc = 0.5 * X
        def res_and_jac(w, c_prev, nu, A, Q, lam, W1, W2, S, Cq):
            c = head(w, W1, W2, S, Cq)
            cm = 0.5 * (c + c_prev)
            X = Q @ cm                                            # (M, R): the ONE tensor pass
            qm = 0.5 * (X @ cm)
            am = A @ cm
            den = 1.0 + 0.5 * DT * nu * lam
            r = (A @ c - A @ c_prev - DT * (qm - nu * lam * am)) / den
            dc = jax.jacfwd(lambda ww: head(ww, W1, W2, S, Cq))(w)   # (R, T), tiny
            Jc = (A - DT * (0.5 * X - 0.5 * nu * lam[:, None] * A)) / den[:, None]   # (M, R)
            return r, Jc @ dc
        t_an, _ = timeit(jax.jit(res_and_jac), *args)
        # parity check vs jacfwd
        r1 = jax.jit(fun_step)(*args); J1 = jf(*args); r2, J2 = jax.jit(res_and_jac)(*args)
        rel = float(jnp.linalg.norm(J1 - J2) / jnp.linalg.norm(J1))
        record(kind='analytic_res+jac', M=M, R=R, K=K, q=q, T=K + q, ms=round(t_an, 3), rel_vs_jacfwd=rel)

        # what one LM iteration costs as written (trial residual + evaluate) vs analytic
        record(kind='lm_iter_as_written', M=M, R=R, T=K + q, ms=round(2 * t_res + t_jf, 3),
               note='fun(zn) + evaluate(zn)=[fun + jacfwd] (XLA may CSE the primal inside jacfwd)')
        record(kind='lm_iter_analytic', M=M, R=R, T=K + q, ms=round(t_res + t_an, 3),
               note='trial residual (1 pass) + fused residual+Jacobian (1 pass)')

# ---------------------------------------------------------------- pre-contracted (M, d, d) at d = K+q
for d in (32, 96, 160, 288):
    M = 544
    cs = make_case(M, d, d, 0)
    Qd, Ad, lamd = cs['Q'], cs['A'], cs['lam']
    w = cs['w']; cp = cs['c_prev']

    def fun_d(w, c_prev, nu, A, Q, lam):
        return RM.make_weak(A, Q, lam, DT)(w, c_prev, nu)
    a = (w, cp, NU, Ad, Qd, lamd)
    t_r, _ = timeit(jax.jit(fun_d), *a)
    t_j, _ = timeit(jax.jit(lambda *x: jax.jacfwd(fun_d)(*x)), *a)
    record(kind='precontracted', M=M, d=d, res_ms=round(t_r, 3), jacfwd_ms=round(t_j, 3), GB=round(Qd.nbytes / 1e9, 4))

# ---------------------------------------------------------------- sampled (EQ-style) residual at m nodes, R = 512
# r_s(c) = sum over 24 stencil products of (Psi_S c)(G_S c) at m nodes, projected by Pq (M, m).
for m in (1024, 2048, 4096):
    M, R, K, q = 2176, 512, 32, 0
    rng = np.random.default_rng(1)
    PS = jnp.asarray(rng.standard_normal((9, m, R)) / np.sqrt(R))     # 9 shifted psi samples
    GS = jnp.asarray(rng.standard_normal((9, m, R)) / np.sqrt(R))     # 9 shifted omega samples
    Pq = jnp.asarray(rng.standard_normal((M, m)) / np.sqrt(m))
    A = jnp.asarray(rng.standard_normal((M, R)) / np.sqrt(R)); lam = jnp.asarray(np.sort(rng.uniform(40, 4e4, M)))
    cs = make_case(8, R, K, q); head = head_fn(K)
    pairs = [(i, j) for i in range(9) for j in range(9)][:24]          # 24 bilinear stencil products (Arakawa)

    def fun_eq(w, c_prev, nu, A, PS, GS, Pq, lam, W1, W2, S, Cq):
        c = head(w, W1, W2, S, Cq); cm = 0.5 * (c + c_prev)
        ps = PS @ cm; gs = GS @ cm                                       # (9, m) each
        adv = sum(ps[i] * gs[j] for i, j in pairs) * (256 * 256 / 12.0)
        am = A @ cm
        return (A @ c - A @ c_prev - DT * (Pq @ adv - nu * lam * am)) / (1.0 + 0.5 * DT * nu * lam)
    a = (cs['w'], cs['c_prev'], NU, A, PS, GS, Pq, lam, cs['W1'], cs['W2'], cs['S'], cs['Cq'])
    t_r, _ = timeit(jax.jit(fun_eq), *a)
    t_j, _ = timeit(jax.jit(lambda *x: jax.jacfwd(fun_eq)(*x)), *a)
    record(kind='eq_sampled', M=M, R=R, m=m, T=K + q, res_ms=round(t_r, 3), jacfwd_ms=round(t_j, 3),
           MB=round((PS.nbytes + GS.nbytes + Pq.nbytes) / 1e6, 1))

json.dump(dict(device=jax.devices()[0].device_kind, results=results), open(os.path.join(OUT, 'profile.json'), 'w'), indent=1)
print('WROTE', os.path.join(OUT, 'profile.json'))
