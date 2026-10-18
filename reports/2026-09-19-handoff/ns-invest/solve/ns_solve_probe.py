"""ns_solve_probe.py -- discriminating arms for the NS2D solve-loss investigation (read-only on
the repo; writes only to the scratchpad).  The weak residual is re-implemented DENSELY through
FFT projection (no advection tensor), which is the same exact quantity (gate R-TQ: tensor vs
dense oracle 2.4e-14), so any arm here is a formulation change on the production residual.

Arms (q = 0 head-only manifold, K = 32; and q = 512 = free bank coefficients):
  weakM      production twin: M = 2176 Fourier tests, Helmholtz row scaling, gtol 1e-6, dt 2e-3
  weakM_g10  same, LM stationarity gtol 1e-10            -> loose-stationarity hypothesis
  full       ALL Fourier modes (= full-grid residual, Helmholtz-scaled)  -> test-space hypothesis
  weakM_dt   dt = 1e-3 (1000 steps)                       -> ROM time-discretisation hypothesis
  galerkin   tangent-space (Galerkin) projection, square system  -> formulation change
  window     restart from the oracle fit of the truth every 100 steps  -> accumulation hypothesis
  onestep    from the oracle fit at step n, ONE weak step; compare with the oracle fit at n+1
             and record the weak residual at both points          -> basin vs projection loss
"""
from __future__ import annotations
import os, sys, json, time, pickle
import numpy as np
NS = '/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-17-ns2d/experiments/ns2d'
sys.path[:0] = [NS, os.path.join(os.path.dirname(NS), 'separable-decoder')]
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import ns2d_fom as F
import ns2d_decoder as D
import ns2d_rom as RM

E = os.environ.get
SP = E('SP', os.path.dirname(os.path.abspath(__file__)))
N = int(E('N', '256'))
CASES = [int(v) for v in E('CASES', '1,0').split(',')]
PLAN = [(int(a.split(':')[0]), a.split(':')[1]) for a in E('PLAN', '0:weakM,0:onestep,0:window,0:full,0:weakM_g10,512:weakM,512:onestep,0:weakM_dt,0:galerkin,512:window,512:full').split(',')]
ARMS = [a for _, a in PLAN]; QS = sorted(set(q for q, _ in PLAN))
DEADLINE = float(E('DEADLINE', '780'))          # seconds of wall clock after which no new arm starts
DT, T, OUT_EVERY = 2e-3, 1.0, 20
NSTEPS = int(round(T / DT)); NOUT = NSTEPS // OUT_EVERY + 1
EVAL_STEPS = [0, 100, 200, 300, 400, 500]
M = int(E('M', '2176'))
GTOL, BUDGET = 1e-6, 200
OUT = f'{SP}/probe_N{N}' + E('TAG', '') + '.json'
t_begin = time.time()
report = dict(N=N, cases=CASES, arms=ARMS, qs=QS, M=M, backend=jax.default_backend(),
              device=str(jax.devices()[0]), results={}, notes=[])


def log(*a):
    print(f'[{time.time()-t_begin:7.1f}s]', *a, flush=True)


def dump():
    json.dump(report, open(OUT, 'w'), indent=1)


def left():
    return DEADLINE - (time.time() - t_begin)


# ------------------------------------------------------------------ model ---
ck = pickle.load(open(f'{NS}/checkpoints/ckpt_K32_R512.pkl', 'rb'))
params = jax.tree_util.tree_map(jnp.asarray, ck['params'])
Ztr = np.asarray(ck['Z_tr']); K, R = 32, 512
G = D.bank_on_grid(params, N)
Qb, Rb = jnp.linalg.qr(G, mode='reduced')
jax.block_until_ready(Rb)
stride = max(1, len(Ztr) // 8192)
Zcand = jnp.asarray(Ztr[::stride])
head_fn = lambda z: D.head(params, z)
Hc = jax.jit(jax.vmap(head_fn))(Zcand)
Hrot = Hc @ Rb.T
Hn = jnp.sum(Hrot * Hrot, 1)
log('bank', G.shape, 'cond Rb', float(jnp.abs(jnp.diag(Rb)).max() / jnp.abs(jnp.diag(Rb)).min()))

# test modes: same ordering as RM.fourier_modes, only the ids (no dense Phi)
cands = []
kmax = N // 2 - 1
for kx in range(-kmax, kmax + 1):
    for ky in range(-kmax, kmax + 1):
        if (kx, ky) == (0, 0) or kx < 0 or (kx == 0 and ky < 0):
            continue
        cands.append((kx * kx + ky * ky, kx, ky))
cands.sort()
ids = []
for k2, kx, ky in cands:
    for name in ('c', 's'):
        ids.append((kx, ky, name))
        if len(ids) == M:
            break
    if len(ids) == M:
        break
kxs = jnp.asarray([i[0] % N for i in ids]); kys = jnp.asarray([i[1] % N for i in ids])
is_cos = jnp.asarray([1.0 if i[2] == 'c' else 0.0 for i in ids])
nrm = N / np.sqrt(2.0)                      # ||cos(2 pi k.x)|| on the grid for k != 0, non-Nyquist
lamg = F.lam_grid(N)
lamM = lamg[kxs, kys]
report['test_space'] = dict(M=M, kmax_sq=int(cands[(M - 1) // 2][0]))
# check nrm against one explicit mode
X, Y = F.coords(N)
v = np.cos(2 * np.pi * (ids[5][0] * X + ids[5][1] * Y)).ravel()
assert abs(np.linalg.norm(v) - nrm) < 1e-8 * nrm, (np.linalg.norm(v), nrm)


def resid_grid(w1, w0, nu, dt):
    wm = 0.5 * (w1 + w0)
    return w1 - w0 - dt * (F.arakawa(F.poisson(wm, N), wm, N) + nu * F.laplacian(wm, N))


def proj_M(Rg, nu, dt):
    Fk = jnp.fft.fft2(Rg)
    vals = Fk[kxs, kys]
    r = (is_cos * jnp.real(vals) - (1.0 - is_cos) * jnp.imag(vals)) / nrm
    return r / (1.0 + 0.5 * dt * nu * lamM)


def proj_all(Rg, nu, dt):
    return jnp.real(jnp.fft.ifft2(jnp.fft.fft2(Rg) / (1.0 + 0.5 * dt * nu * lamg))).ravel()


def make_lm(fun, budget, trust=np.inf):
    """RM.make_lm with gtol traced (passed in args[-1])."""
    def lm(z0, args, tol, gtol):
        def evaluate(z):
            r = fun(z, *args); J = jax.jacfwd(fun)(z, *args)
            return r, J, jnp.linalg.norm(r)

        def grad(r, J):
            return jnp.linalg.norm(J.T @ r) / (jnp.linalg.norm(J) * jnp.linalg.norm(r) + 1e-300)

        r, J, rn = evaluate(z0)
        reason = jnp.where(jnp.isfinite(rn), jnp.where(grad(r, J) <= gtol, 4,
                                                       jnp.where(rn <= tol, 1, 0)), 3).astype(jnp.int32)

        def body(s):
            z, r, J, rn, lam, it, reason = s
            H = J.T @ J; g = J.T @ r
            dz = jnp.linalg.solve(H + lam * jnp.diag(jnp.diag(H) + 1e-30), -g)
            ok = jnp.all(jnp.isfinite(dz)) & (jnp.linalg.norm(dz) <= trust)
            zn = z + jnp.where(ok, dz, 0.)
            rn2 = jnp.linalg.norm(fun(zn, *args))
            accept = ok & jnp.isfinite(rn2) & (rn2 < rn)
            r2, J2, rn2 = jax.lax.cond(accept, lambda: evaluate(zn), lambda: (r, J, rn))
            gn = grad(r2, J2)
            tiny = ok & (jnp.linalg.norm(dz) <= 1e-14 * (1 + jnp.linalg.norm(z)))
            reason = jnp.where(gn <= gtol, 4, jnp.where(rn2 <= tol, 1, jnp.where(
                tiny, 2, jnp.where((~accept) & (lam >= 1e14), 3, 0)))).astype(jnp.int32)
            return (jnp.where(accept, zn, z), r2, J2, rn2,
                    jnp.where(accept, jnp.maximum(lam / 3, 1e-12), jnp.minimum(lam * 10, 1e14)),
                    it + 1, reason)

        z, r, J, rn, lam, it, reason = jax.lax.while_loop(
            lambda s: (s[5] < budget) & (s[6] == 0), body,
            (z0, r, J, rn, jnp.asarray(1e-6), jnp.int32(0), reason))
        return z, rn, it, reason, grad(r, J)
    return lm


def build(q, mode):
    """mode in {'M', 'all', 'galerkin'}; q in {0, 512}.  Returns (fun_step, coef, step_scan(window))."""
    if q == 0:
        coef = head_fn
    else:
        coef = lambda w: w

    if mode == 'galerkin':
        # tangent basis at the PREVIOUS state: T = G dh/dw (n, K+q); square system T^T R = 0,
        # solved as a least-squares problem with the same LM (Jacobian is square).
        def fun_step(w, c_prev, nu, dt, Gd, Tt):
            w1 = (Gd @ coef(w)).reshape(N, N); w0 = (Gd @ c_prev).reshape(N, N)
            Rg = resid_grid(w1, w0, nu, dt)
            return Tt.T @ Rg.ravel()
    else:
        proj = proj_M if mode == 'M' else proj_all

        def fun_step(w, c_prev, nu, dt, Gd):
            w1 = (Gd @ coef(w)).reshape(N, N); w0 = (Gd @ c_prev).reshape(N, N)
            return proj(resid_grid(w1, w0, nu, dt), nu, dt)
    lm = make_lm(fun_step, BUDGET)

    def one_step(w, wprev, nu, dt, Gd, scale, gtol):
        c_prev = coef(w)
        if mode == 'galerkin':
            Tt = jax.jacfwd(lambda ww: Gd @ coef(ww))(w)
            args = (c_prev, nu, dt, Gd, Tt)
        else:
            args = (c_prev, nu, dt, Gd)
        we = w + (w - wprev)
        r0 = jnp.linalg.norm(fun_step(w, *args)); re = jnp.linalg.norm(fun_step(we, *args))
        wi = jnp.where(jnp.isfinite(re) & (re < r0), we, w)
        w2, rn, it, reason, gn = lm(wi, args, 1e-9 * scale, gtol)
        return w2, rn, it, reason, gn

    @jax.jit
    def window(w, wprev, nu, dt, Gd, scale, gtol):
        def step(carry, _):
            w, wprev = carry
            w2, rn, it, reason, gn = one_step(w, wprev, nu, dt, Gd, scale, gtol)
            return (w2, w), (rn, it, reason, gn)
        (w, wprev), (rn, it, reason, gn) = jax.lax.scan(step, (w, wprev), None, length=100)
        return w, wprev, rn, it, reason, gn

    @jax.jit
    def single(w, wprev, nu, dt, Gd, scale, gtol):
        return one_step(w, wprev, nu, dt, Gd, scale, gtol)

    @jax.jit
    def resid_at(w, c_prev, nu, dt, Gd):
        if mode == 'galerkin':
            Tt = jax.jacfwd(lambda ww: Gd @ coef(ww))(w)
            return jnp.linalg.norm(fun_step(w, c_prev, nu, dt, Gd, Tt))
        return jnp.linalg.norm(fun_step(w, c_prev, nu, dt, Gd))

    return coef, window, single, resid_at


ic_lm = make_lm(lambda w, y, Rbm: Rbm @ head_fn(w) - y, 400)


@jax.jit
def ic_fit_q0(u0, Qb, Rb):
    y = Qb.T @ u0
    idx = jnp.argmin(Hn - 2 * Hrot @ y)
    w, icrn, icit, icreason, icgn = ic_lm(Zcand[idx], (y, Rb), 0., GTOL)
    return w, icrn, icit


@jax.jit
def bank_coef(u, Qb, Rb):
    return jnp.linalg.solve(Rb, Qb.T @ u)


field = jax.jit(lambda c, Gd: Gd @ c)

# ------------------------------------------------------------------ truth ---
phys_dev = F.params_draw(20260918, 64)
ref = np.load(f'{SP}/ns304/experiments/ns2d/output/reference_N256.npz') if N == 256 else None
run_fom, _ = F.make_fom(N, DT, NSTEPS, 1)
truth = {}
for c in CASES:
    t0 = time.time()
    w0 = jnp.asarray(F.initial(N, phys_dev[c])); nu = float(phys_dev[c][-1])
    st, it, rn = run_fom(w0, nu, 1e-11, 1e-9)
    jax.block_until_ready(st)
    st_h = np.asarray(st).reshape(NSTEPS + 1, -1)
    chk = None
    if ref is not None:
        chk = max(float(np.linalg.norm(st_h[s] - ref['U'][c, j]) / np.linalg.norm(ref['U'][c, j]))
                  for j, s in enumerate(EVAL_STEPS))
    truth[c] = dict(states=st_h, nu=nu, n0=float(np.linalg.norm(st_h[0])), value_check=chk,
                    newton_max=int(np.asarray(it).max()), worst_rel_res=float(np.asarray(rn).max()))
    log(f'FOM case {c} nu={nu:.4g} Re={1/nu:.0f} in {time.time()-t0:.1f}s, value check vs ns304 reference {chk}, newton max {truth[c]["newton_max"]}')
    report['results'][f'fom_case{c}'] = {k: v for k, v in truth[c].items() if k != 'states'}
dump()

Gd = G


def errs(fields, c):
    """fields (6, n) at EVAL_STEPS -> fixed-initial relative errors."""
    tr = truth[c]['states'][EVAL_STEPS]
    return (np.linalg.norm(fields - tr, axis=1) / truth[c]['n0']).tolist()


def oracle_targets(c, steps):
    U = jnp.asarray(truth[c]['states'][steps])
    Ct = jax.vmap(lambda u: bank_coef(u, Qb, Rb))(U)
    Z, rn, its, reasons = D.oracle_fit(head_fn, Rb, Ct, Zcand, n_starts=4, budget=200, gtol=GTOL)
    return Z, Ct, its, reasons


built = {}


def get(q, mode):
    if (q, mode) not in built:
        built[(q, mode)] = build(q, mode)
    return built[(q, mode)]


def rollout(q, mode, c, dt, gtol, restart=None):
    """Full horizon by chained 100-step windows; restart=(Z at window starts) for the window arm."""
    coef, window, single, resid_at = get(q, mode)
    nu = truth[c]['nu']; scale = truth[c]['n0']
    nwin = int(round(T / (100 * dt)))
    fields, its, reasons, gns, rns = [], [], [], [], []
    if q == 0:
        w, icrn, icit = ic_fit_q0(jnp.asarray(truth[c]['states'][0]), Qb, Rb)
    else:
        w = bank_coef(jnp.asarray(truth[c]['states'][0]), Qb, Rb)
    wprev = w
    out_w = [w]
    per_window = []
    for wi in range(nwin):
        if restart is not None and wi > 0:
            w = restart[wi]; wprev = w
        w, wprev, rn, it, reason, gn = window(w, wprev, nu, dt, Gd, scale, gtol)
        jax.block_until_ready(w)
        out_w.append(w)
        its.append(int(np.asarray(it).sum())); reasons.append({str(k): int(v) for k, v in zip(*np.unique(np.asarray(reason), return_counts=True))})
        gns.append(float(np.asarray(gn).max())); rns.append(float(np.asarray(rn).max()))
    W = jnp.stack(out_w)
    if dt == DT:
        idx = list(range(0, nwin + 1, 1))            # 6 windows of 100 steps -> eval steps
    else:
        idx = list(range(0, nwin + 1, int(round(DT / dt))))
    fl = np.asarray(jax.vmap(lambda w: field(coef(w), Gd))(W[jnp.asarray(idx)]))
    e = errs(fl, c)
    return dict(fixed_per_time=e, worst_evolved=float(max(e[1:])), iterations_per_window=its,
                reasons_per_window=reasons, worst_gradient=max(gns), worst_resid=max(rns))


for q, arm in PLAN:
    if True:
        for c in CASES:
            key = f'q{q}_{arm}_case{c}'
            if left() < 0:
                report['notes'].append(f'skipped {key}: deadline'); dump(); continue
            t0 = time.time()
            try:
                if arm == 'weakM':
                    res = rollout(q, 'M', c, DT, GTOL)
                elif arm == 'weakM_g10':
                    res = rollout(q, 'M', c, DT, 1e-10)
                elif arm == 'full':
                    res = rollout(q, 'all', c, DT, GTOL)
                elif arm == 'weakM_dt':
                    res = rollout(q, 'M', c, 1e-3, GTOL)
                elif arm == 'galerkin':
                    res = rollout(q, 'galerkin', c, DT, GTOL)
                elif arm == 'window':
                    if q == 0:
                        Z, Ct, its, reasons = oracle_targets(c, EVAL_STEPS)
                        Zr = [jnp.asarray(Z[i]) for i in range(6)]
                        man = [float(np.linalg.norm(np.asarray(field(head_fn(Zr[i]), Gd)) - truth[c]['states'][EVAL_STEPS[i]]) / truth[c]['n0']) for i in range(6)]
                    else:
                        Zr = [bank_coef(jnp.asarray(truth[c]['states'][s]), Qb, Rb) for s in EVAL_STEPS]
                        man = [float(np.linalg.norm(np.asarray(field(Zr[i], Gd)) - truth[c]['states'][EVAL_STEPS[i]]) / truth[c]['n0']) for i in range(6)]
                    res = rollout(q, 'M', c, DT, GTOL, restart=Zr)
                    res['manifold_at_restarts'] = man
                    res['window_excess'] = [res['fixed_per_time'][i] - man[i] for i in range(6)]
                elif arm == 'onestep':
                    coef, window, single, resid_at = get(q, 'M')
                    nu = truth[c]['nu']; scale = truth[c]['n0']
                    steps = list(range(0, 500, 50))
                    if q == 0:
                        Za, _, _, _ = oracle_targets(c, steps)
                        Zb, _, _, _ = oracle_targets(c, [s + 1 for s in steps])
                        Za, Zb = jnp.asarray(Za), jnp.asarray(Zb)
                    else:
                        Za = jnp.stack([bank_coef(jnp.asarray(truth[c]['states'][s]), Qb, Rb) for s in steps])
                        Zb = jnp.stack([bank_coef(jnp.asarray(truth[c]['states'][s + 1]), Qb, Rb) for s in steps])
                    rows = []
                    for i, s in enumerate(steps):
                        w2, rn, it, reason, gn = single(Za[i], Za[i], nu, DT, Gd, scale, GTOL)
                        c_prev = coef(Za[i])
                        r_rom = float(rn); r_orc = float(resid_at(Zb[i], c_prev, nu, DT, Gd))
                        f_rom = np.asarray(field(coef(w2), Gd)); f_orc = np.asarray(field(coef(Zb[i]), Gd))
                        tr1 = truth[c]['states'][s + 1]
                        rows.append(dict(step=s, resid_rom=r_rom, resid_oracle=r_orc,
                                         err_rom=float(np.linalg.norm(f_rom - tr1) / scale),
                                         err_oracle=float(np.linalg.norm(f_orc - tr1) / scale),
                                         gap_rom_vs_oracle=float(np.linalg.norm(f_rom - f_orc) / scale),
                                         err_prev_oracle=float(np.linalg.norm(np.asarray(field(c_prev, Gd)) - truth[c]['states'][s]) / scale),
                                         it=int(it), reason=int(reason), gn=float(gn)))
                    res = dict(rows=rows,
                               mean_gap=float(np.mean([r['gap_rom_vs_oracle'] for r in rows])),
                               mean_gap_x500=float(500 * np.mean([r['gap_rom_vs_oracle'] for r in rows])))
                else:
                    raise ValueError(arm)
                res['seconds'] = time.time() - t0
                report['results'][key] = res
                log(key, json.dumps({k: v for k, v in res.items() if k in ('fixed_per_time', 'worst_evolved', 'mean_gap', 'mean_gap_x500', 'window_excess', 'manifold_at_restarts', 'seconds', 'worst_gradient')}))
            except Exception as ex:                                         # noqa: BLE001
                report['results'][key] = dict(error=repr(ex)[:500], seconds=time.time() - t0)
                log(key, 'ERROR', repr(ex)[:300])
            dump()
report['seconds'] = time.time() - t_begin
dump()
log('done')
