"""Batched throughput: frozen ROM arms vs their accuracy-matched pseudo-spectral solver
configurations (PROTOCOL.md). One job per mesh; everything timed inside that job on one GPU.

Reads (read-only) the source lane's frozen files, banks, heads, fresh cohort and ladders.
Writes only <out>/output/batched.json.
"""
from __future__ import annotations
import argparse, gc, json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ns3d-hard"))
import pipeline as P                     # noqa: E402  (sets x64, matmul precision, sys.path)
import rank as RK                        # noqa: E402
from pipeline import jax, jnp, H, F, SR, HR, log, HEAD, jdump, ints   # noqa: E402

AXES = (-3, -2, -1)


# ----------------------------------------------------------------- ROM, shared operator --

def shift_field_r(field, offset_samples):
    """SR.shift_field with real-to-complex FFTs (same phases; the last axis is half-spectrum)."""
    n = field.shape[-1]
    spec = jnp.fft.rfftn(field, axes=AXES)
    modes = jnp.fft.fftfreq(n) * n; rmodes = jnp.fft.rfftfreq(n) * n
    for axis in range(3):
        mm = rmodes if axis == 2 else modes
        shape = [1, 1, 1]; shape[axis] = mm.shape[0]
        spec = spec * jnp.exp(-2j * jnp.pi * mm * (offset_samples[axis] / n)).reshape((1, *shape))
    return jnp.fft.irfftn(spec, s=(n, n, n), axes=AXES)


def make_rom_parts(dt, nsteps, out_every, n, R, k, iters, damping, ic_iters, rfft_shift=False):
    """The unchanged head query (head_rom.make_head_run, q = 0, frame free, extrapolate),
    split into encode / latent / decode, with ONE rearrangement: the viscosity-dependent
    preconditioner division of the quadratic operator is applied after the contraction,
    so the operator (T + T^T) is shared by all queries of a batch."""
    code_fit = HR.make_code_fit(k, ic_iters)
    C0 = jnp.zeros((R, 0))
    shift = shift_field_r if rfft_shift else SR.shift_field

    def prepare(nu, A, lam, Dd):                 # SR.make_frozen_step.prepare without `sym`
        pre = 1.0 + 0.5 * dt * nu * lam
        Apre = A / pre[:, None]
        lamApre = (nu * lam / pre)[:, None] * A
        Ddpre = Dd / pre[None, :, None]
        return Apre, pre, Apre + 0.5 * dt * lamApre, lamApre, Ddpre

    def residual_and_jacobian(w, prev, ops, symS):
        Apre, pre, jconst, lamApre, Ddpre = ops
        a, delta = w[:R], w[R:]
        mid = 0.5 * (a + prev)
        symmid = jnp.einsum("msk,k->ms", symS, mid) / pre[:, None]
        shift_cols = jnp.einsum("dmr,r->md", Ddpre, mid)
        res = (Apre @ (a - prev) - 0.5 * dt * (symmid @ mid) + dt * (lamApre @ mid)
               - shift_cols @ delta)
        jac_a = jconst - 0.5 * dt * symmid - 0.5 * jnp.einsum("dmr,d->mr", Ddpre, delta)
        return res, jnp.concatenate((jac_a, -shift_cols), axis=1)

    def step_solve(w0, a_prev, ops, p, symS):
        def body(w, _):
            a = HR.head_apply(p, w[:k])
            res, jfull = residual_and_jacobian(jnp.concatenate((a, w[k:])), a_prev, ops, symS)
            jac = jnp.concatenate((jfull[:, :R] @ HR.head_jacobian(p, w[:k]), jfull[:, R:]), axis=1)
            hess = jac.T @ jac
            L = jnp.linalg.cholesky(hess + jnp.diag(damping * jnp.diagonal(hess) + 1e-300))
            return w + jax.scipy.linalg.cho_solve((L, True), -(jac.T @ res)), None
        w, _ = jax.lax.scan(body, w0, None, length=iters, unroll=iters)
        return w

    def encode(u0, basis, p, Hcand, Hnorm, Zcand):
        c0 = SR.grid_centroid(u0)
        a0 = basis.T @ shift(u0, -c0 * n).ravel()
        idx = jnp.argmin(Hnorm - 2 * (Hcand @ a0))
        z0, y0 = code_fit(p, Zcand[idx], a0, C0)
        return jnp.concatenate((z0, y0, jnp.zeros(3))), c0

    def latent(w0, c0, nu, A, symS, lam, Dd, p):
        ops = prepare(nu, A, lam, Dd)

        def step(carry, _):
            w, previous, c = carry
            new = step_solve(w + (w - previous), HR.head_apply(p, w[:k]), ops, p, symS)
            return (new, w, (c + new[k:]) % 1.0), None

        def block(carry, _):
            carry, _ = jax.lax.scan(step, carry, None, length=out_every)
            return carry, (carry[0], carry[2])
        _, (W, cs) = jax.lax.scan(block, (w0, w0, c0), None, length=nsteps // out_every)
        return W, cs

    def decode(w0, c0, W, cs, basis, p):
        Wall = jnp.concatenate((w0[None], W)); call = jnp.concatenate((c0[None], cs))
        coef = HR.head_apply(p, Wall[:, :k])                         # (6, R)
        fields = (basis @ coef.T).T.reshape(-1, 3, n, n, n)
        return jax.vmap(lambda f, c: shift(f, c * n))(fields, call)

    sym = lambda T: T + jnp.swapaxes(T, 1, 2)

    def e2e_b(u0s, nus, basis, A, T, lam, Dd, p, Hcand, Hnorm, Zcand):
        symS = sym(T)

        def one(u0, nu):
            w0, c0 = encode(u0, basis, p, Hcand, Hnorm, Zcand)
            W, cs = latent(w0, c0, nu, A, symS, lam, Dd, p)
            return decode(w0, c0, W, cs, basis, p)
        return jax.vmap(one)(u0s, nus)

    def enc_b(u0s, basis, p, Hcand, Hnorm, Zcand):
        return jax.vmap(lambda u: encode(u, basis, p, Hcand, Hnorm, Zcand))(u0s)

    def lat_b(w0s, c0s, nus, A, T, lam, Dd, p):
        symS = sym(T)
        return jax.vmap(lambda w, c, v: latent(w, c, v, A, symS, lam, Dd, p))(w0s, c0s, nus)

    def dec_b(w0s, c0s, Ws, css, basis, p):
        return jax.vmap(lambda a, b, c, d: decode(a, b, c, d, basis, p))(w0s, c0s, Ws, css)
    return e2e_b, enc_b, lat_b, dec_b


# ------------------------------------------------------------------- solver, real FFT --

def geometry_r(n):
    kk = np.fft.fftfreq(n, d=1.0 / n); kr = np.fft.rfftfreq(n, d=1.0 / n)
    modes = np.stack(np.meshgrid(kk, kk, kr, indexing="ij"))
    k = 2 * np.pi * modes
    mask = np.all(np.abs(modes) < n / 3, axis=0)
    mask[0, 0, 0] = False
    return tuple(jnp.asarray(a) for a in (k, np.sum(k * k, axis=0), mask))


def make_solver_r(dt, nsteps, out_every, scheme):
    """ns3d_fom.make_solver (forcing None, adv_sign 1) with rfftn/irfftn; same scheme."""
    assert nsteps % out_every == 0 and scheme in ("cnab2", "rk4")

    def run(u0, nu, geom):
        n = u0.shape[-1]
        fft = lambda u: jnp.fft.rfftn(u, axes=AXES, norm="forward")
        ifft = lambda uh: jnp.fft.irfftn(uh, s=(n, n, n), axes=AXES, norm="forward")

        def project(uh):
            k, k2, mask = geom
            dot = jnp.sum(k * uh, axis=0)
            return (uh - k * (dot / jnp.where(k2 > 0, k2, 1.0))) * mask

        def nonlinear(uh):
            uh = project(uh)
            return project(fft(F.cross(ifft(uh), ifft(1j * F.cross(geom[0], uh)))))
        k2 = geom[1]
        uh0 = project(fft(u0))

        def step(carry, index):
            uh, old_e = carry
            if scheme == "rk4":
                fun = lambda u: nonlinear(u) - nu * k2 * u
                a = fun(uh); b = fun(uh + dt * a / 2); c = fun(uh + dt * b / 2); d = fun(uh + dt * c)
                unew = uh + dt * (a + 2 * b + 2 * c + d) / 6; enew = old_e
            else:
                e = nonlinear(uh); den = 1 + 0.5 * dt * nu * k2

                def startup():
                    pred = project((uh + dt * e) / (1 + dt * nu * k2))
                    return ((1 - 0.5 * dt * nu * k2) * uh + 0.5 * dt * (e + nonlinear(pred))) / den

                def normal():
                    return ((1 - 0.5 * dt * nu * k2) * uh + dt * (1.5 * e - 0.5 * old_e)) / den
                unew = jax.lax.cond(index == 0, startup, normal); enew = e
            return (project(unew), enew), None

        def block(carry, b):
            state, _ = jax.lax.scan(step, carry, b * out_every + jnp.arange(out_every))
            return state, ifft(state[0])
        _, out = jax.lax.scan(block, (uh0, jnp.zeros_like(uh0)), jnp.arange(nsteps // out_every))
        return jnp.concatenate((ifft(uh0)[None], out))
    return run


# --------------------------------------------------------------------------- helpers --

def sds(x):
    return jax.ShapeDtypeStruct(x.shape, x.dtype)


def nbytes(tree):
    return int(sum(np.prod(x.shape) * np.dtype(x.dtype).itemsize for x in jax.tree_util.tree_leaves(tree)))


def free_bytes():
    st = jax.devices()[0].memory_stats() or {}
    return int(st.get("bytes_limit", 80e9) - st.get("bytes_in_use", 0))


def build(calls, rep, B, name, *args, **kw):
    try:
        calls[name] = Callable(name, *args, **kw)
        return calls[name]
    except Exception as e:
        rep["failures"].append(f"B={B}: {name} not built: {type(e).__name__}: {str(e)[:200]}")
        log(rep["failures"][-1]); gc.collect()
        return None


class Callable:
    """A batched function at batch B: AOT compile at chunk c (largest power of two <= B
    whose buffers fit), executed as ceil(B / c) consecutive calls when c < B (the chunk
    slices are taken inside the timed call)."""

    def __init__(self, name, fn, batch_args, shared_args, B, max_chunk=None):
        self.name, self.fn, self.B = name, fn, B
        self.batch_args, self.shared = batch_args, shared_args      # arrays with leading dim B
        self.jit = jax.jit(fn)
        self.chunk, self.mem, self.chunks_tried = None, None, []
        self._compile(B if max_chunk is None else min(B, max_chunk))

    def _compile(self, c):
        self.chunk = None
        while c >= 1:
            spec = [jax.ShapeDtypeStruct((c,) + x.shape[1:], x.dtype) for x in self.batch_args]
            try:
                comp = self.jit.lower(*spec, *self.shared).compile()
            except Exception as e:   # e.g. GEMM autotuning runs out of device memory at this chunk
                self.chunks_tried.append(dict(chunk=c, compile_error=f"{type(e).__name__}: {str(e)[:160]}"))
                gc.collect(); c //= 2
                continue
            ma = comp.memory_analysis()
            need_batch = int(ma.output_size_in_bytes + ma.temp_size_in_bytes - ma.alias_size_in_bytes
                             + (sum(nbytes(s) for s in spec) if c < self.B else 0))
            info = dict(chunk=c, args_gb=ma.argument_size_in_bytes / 1e9, out_gb=ma.output_size_in_bytes / 1e9,
                        temp_gb=ma.temp_size_in_bytes / 1e9, batch_need_gb=need_batch / 1e9,
                        total_gb=(ma.argument_size_in_bytes + ma.output_size_in_bytes + ma.temp_size_in_bytes
                                  - ma.alias_size_in_bytes) / 1e9, free_gb=free_bytes() / 1e9)
            self.chunks_tried.append(info)
            if need_batch * 1.15 < free_bytes():
                self.comp, self.chunk, self.mem = comp, c, info
                return
            del comp; c //= 2
        raise RuntimeError(f"{self.name}: nothing fits")

    def shrink(self):
        if self.chunk <= 1:
            raise RuntimeError(f"{self.name}: cannot shrink below 1")
        self.chunks_tried.append(dict(oom_at_runtime=self.chunk))
        self._compile(self.chunk // 2)

    @property
    def chunked(self):
        return self.chunk < self.B

    def piece(self, q):
        return [x[q * self.chunk:(q + 1) * self.chunk] for x in self.batch_args]

    def __call__(self):
        if not self.chunked:
            return self.comp(*self.batch_args, *self.shared)
        for q in range(-(-self.B // self.chunk)):
            out = self.comp(*self.piece(q), *self.shared); jax.block_until_ready(out); del out
        return None

    def slot(self, s):
        """Output of batch slot s (runs only the chunk holding it)."""
        if not self.chunked:
            o = self.comp(*self.batch_args, *self.shared)
        else:
            o = self.comp(*self.piece(s // self.chunk), *self.shared)
        return jax.tree_util.tree_map(lambda x: x[s % self.chunk if self.chunked else s], o)

    def full_output(self):
        """Concatenated output over the whole batch (small outputs only)."""
        outs = ([self.comp(*self.batch_args, *self.shared)] if not self.chunked else
                [self.comp(*self.piece(q), *self.shared) for q in range(-(-self.B // self.chunk))])
        return jax.tree_util.tree_map(lambda *x: jnp.concatenate(x), *outs) if len(outs) > 1 else outs[0]


def safe_first_call(c):
    """Untimed first call; on a runtime out-of-memory, halve the chunk and retry."""
    while True:
        try:
            out = c(); jax.block_until_ready(out); del out
            return
        except Exception as e:
            if "RESOURCE_EXHAUSTED" not in str(e) and "out of memory" not in str(e).lower():
                raise
            gc.collect(); c.shrink()


DROPPED = []


def burn(seconds=2.0):
    warm = jnp.ones((1024, 1024)) * 1e-3; t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        warm = warm @ warm * 1e-3 + 1e-3
    jax.block_until_ready(warm)


def time_block(calls, reps, seed):
    burn()
    for name in list(calls):          # one untimed call each (after AOT compile); drop what cannot run
        try:
            safe_first_call(calls[name])
        except Exception as e:
            log(f"dropped from timing block: {name}: {type(e).__name__}: {str(e)[:200]}")
            DROPPED.append(f"{name}: {type(e).__name__}: {str(e)[:200]}")
            del calls[name]; gc.collect()
    rng = np.random.default_rng(seed); names = list(calls); samples = {k: [] for k in names}
    for _ in range(reps):
        for name in rng.permutation(names):
            t0 = time.perf_counter(); out = calls[name](); jax.block_until_ready(out)
            samples[name].append((time.perf_counter() - t0) * 1e3); del out
    return samples


def relmax(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


# ------------------------------------------------------------------------------ main --

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--mesh", type=int, required=True)
    ap.add_argument("--family", default="B")
    ap.add_argument("--src", default=str(HERE.parent / "ns3d-hard"))
    ap.add_argument("--compare", default=None, help="fresh_s0b compare.json (default: source results)")
    ap.add_argument("--ns", default="/cluster/tufts/paralab/tawal01/rebuttal-ns3d-hard-rank")
    ap.add_argument("--arms", default="r96_head,r128_head_fast,r192_head_fast")
    ap.add_argument("--batches", default="1,8,32,128"); ap.add_argument("--rom-extra", default="512")
    ap.add_argument("--reps-small", type=int, default=5); ap.add_argument("--reps-large", type=int, default=3)
    ap.add_argument("--cases", type=int, default=128)
    ap.add_argument("--no-rematch", action="store_true"); ap.add_argument("--no-cohort", action="store_true")
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--rom-rfft", action="store_true")
    ap.add_argument("--rematch-only", action="store_true")
    ap.add_argument("--lean", action="store_true", help="headline callables only: ROM shared (+r), solver rfft solve-only")
    a = ap.parse_args()
    if jax.default_backend() != "gpu" and not a.smoke:
        raise RuntimeError("need a GPU backend")
    n = a.mesh; fam = H.FAMILIES[a.family]; T = fam["horizon"]; NS = a.ns
    out = Path(a.out) / "output"; out.mkdir(parents=True, exist_ok=True)
    t_job = time.time()
    cmp = json.loads(Path(a.compare or (Path(a.src) / "results/fresh_s0b/compare.json")).read_text())
    mc = cmp["meshes"][str(n)]
    arms = a.arms.split(",")
    rep = dict(schema="ns3d-batched-v1", mesh=n, args=vars(a), job_id=os.environ.get("SLURM_JOB_ID"),
               device=str(jax.devices()[0]), backend=jax.default_backend(),
               matmul_precision=str(jax.config.jax_default_matmul_precision), x64=bool(jax.config.jax_enable_x64),
               source_compare_job=cmp.get("job_id"), source_commit=cmp.get("source_commit"),
               arms={}, solvers={}, batches={}, checks={}, cohort={}, rematch={}, failures=[])
    try:
        import subprocess
        rep["gpu"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,memory.total",
                                              "--format=csv,noheader"], text=True).strip()
    except Exception:
        rep["gpu"] = "unavailable"
    cf = HERE.parents[1] / "COMMIT.txt"; rep["commit"] = cf.read_text().strip() if cf.exists() else None

    def dump():
        rep["elapsed_seconds"] = time.time() - t_job
        st = jax.devices()[0].memory_stats() or {}
        rep["process_peak_gb"] = st.get("peak_bytes_in_use", 0) / 1e9
        jdump(out / "batched.json", rep)

    # ---- cohort (fresh, seed 2026111151), truth on mesh n on the host
    cohort = f"{NS}/genF_0,{NS}/genF_1"
    arrs, par = P.load_truth(cohort.split(","))
    ncases = min(a.cases, sum(x.shape[0] for x in arrs)); nus = par[:ncases, -1]
    truth = []
    for c, fr in enumerate(P.iter_cases(arrs)):
        if c >= ncases: break
        truth.append(np.asarray(P.to_mesh(np.asarray(fr), n)))
    uS = [np.asarray(fr[0]) for c, fr in zip(range(ncases), P.iter_cases(arrs))]
    log(f"mesh {n}: {ncases} cases loaded ({time.time() - t_job:.0f}s)")

    # ---- ROM arms (frozen settings), banks and heads as in fresh_s0b
    fz = json.loads((Path(a.src) / "frozen_rank" / f"frozen_{n}.json").read_text())["arms"]
    G, ops = P.load_rom(f"{NS}/bank128_s0b", n); lo = RK.RankArms(n, T, G, ops); rlo = G.shape[1]; del G, ops
    need_hi = any(fz[x]["rank"] > rlo for x in arms)
    hi = None
    if need_hi:
        G, ops = P.load_rom(f"{NS}/bank192_s0", n); hi = RK.RankArms(n, T, G, ops); del G, ops
    rom = {}
    for name in arms:
        cfg = fz[name]; r = cfg["rank"]; assert cfg["kind"] == "head"
        ar, hdir = (lo, f"{NS}/rdev_s0_{n}") if r <= rlo else (hi, f"{NS}/rdev192_s0_{n}")
        tag = (r, cfg["k"], cfg["width"])
        if tag not in ar.heads:
            ar.add_head(tag, r, RK.load_head(Path(hdir) / "output" / RK.head_file(*tag)))
        _, k, p, Z, Hc, Hn = ar.heads[tag]
        steps, it = cfg["steps"], cfg["iters"]
        run = HR.make_head_run(T / steps, steps, steps // 5, n, r, k, 0, iters=it,
                               damping=HEAD["damping"], ic_iters=HEAD["ic_iters"])
        Gp = ar.prefix(r); A_, T_, lam_, Dd_ = ar.sliced(r, cfg["M"]); C = jnp.zeros((r, 0))
        parts = make_rom_parts(T / steps, steps, steps // 5, n, r, k, it, HEAD["damping"], HEAD["ic_iters"])
        parts_r = make_rom_parts(T / steps, steps, steps // 5, n, r, k, it, HEAD["damping"], HEAD["ic_iters"],
                                 rfft_shift=True)
        rom[name] = dict(cfg=cfg, run=run, full=(Gp, A_, T_, lam_, Dd_, p, C, Hc, Hn, Z),
                         parts=parts, parts_r=parts_r, G=Gp, A=A_, T=T_, lam=lam_, Dd=Dd_, p=p, Hc=Hc, Hn=Hn, Z=Z)
        ent = mc["arms"][name]
        rep["arms"][name] = dict(config=cfg, fresh_worst=ent["stats"]["evolved_worst"],
                                 fresh_median=ent["stats"]["evolved_median"], fresh_ms_batch1=ent["ms"]["median_ms"],
                                 matched=dict((kk, ent["any_mesh_solve"][kk]) for kk in
                                              ("scheme", "n", "steps", "cohort_worst", "cohort_median", "ms", "ratio")),
                                 matched_e2e_ms=ent["any_mesh_e2e"]["ms"],
                                 shortlist=ent["shortlist"])
    solver_keys = sorted({(mc["arms"][x]["any_mesh_solve"]["scheme"], mc["arms"][x]["any_mesh_solve"]["n"],
                           mc["arms"][x]["any_mesh_solve"]["steps"]) for x in arms})
    rematch_keys = sorted({tuple(s) for x in arms for s in mc["arms"][x]["shortlist"]})
    rep["solver_keys"] = [list(x) for x in solver_keys]; rep["rematch_keys"] = [list(x) for x in rematch_keys]
    gc.collect(); dump()

    # ---- solvers: unchanged (complex FFT) and real-FFT, solve-only and end-to-end
    sol = {}

    def solver(key, var):
        if (key, var) not in sol:
            scheme, m, s = key
            if var == "vmap":
                fn = F.make_solver(T / s, s, s // 5, scheme=scheme); geom = F.geometry(m)
            else:
                fn = make_solver_r(T / s, s, s // 5, scheme); geom = geometry_r(m)
            sol[(key, var)] = (fn, geom)
        return sol[(key, var)]

    ref_solver = {}

    def solve_ref(key, c):                 # unchanged batch-1 solver, solve-only input
        scheme, m, s = key
        if key not in ref_solver:
            ref_solver[key] = F.make_solver(T / s, s, s // 5, scheme=scheme)
        return ref_solver[key](P.spec_resample(jnp.asarray(uS[c]), m), float(nus[c]), F.geometry(m))

    def e2e_ref(key, c):
        return P.spec_resample(solve_ref(key, c), n)

    def rom_ref(name, c):                  # unchanged batch-1 ROM query (the fresh_s0b callable)
        R_ = rom[name]
        return R_["run"](jnp.asarray(truth[c][0]), float(nus[c]), *R_["full"])

    # ---- batch loop
    Bs = ints(a.batches); extra = ints(a.rom_extra) if a.rom_extra else []
    for B in Bs + [b for b in extra if b not in Bs]:
        rom_only = B not in Bs
        if rom_only and B * 6 * 3 * n ** 3 * 8 > 0.5 * free_bytes():
            rep["batches"][str(B)] = dict(B=B, skipped="ROM outputs exceed half the free device memory", callables={})
            log(f"B={B}: skipped (memory)"); continue
        idx = np.arange(B) % ncases
        u0n = jnp.asarray(np.stack([truth[c][0] for c in idx])); nuB = jnp.asarray(nus[idx])
        calls = {}; brep = dict(B=B, callables={}); rep["batches"][str(B)] = brep
        u0S, u0ms = None, {}
        if not rom_only:      # every input of this B on the device before anything is compiled
            u0S = jnp.asarray(np.stack([uS[c] for c in idx]))
            for key in solver_keys:
                if key[1] not in u0ms:
                    u0ms[key[1]] = jnp.concatenate([P.spec_resample(u0S[q:q + 16], key[1]) for q in range(0, B, 16)])
        try:
            for name in (arms if not a.rematch_only else []):
                R_ = rom[name]; e2e_b, enc_b, lat_b, dec_b = R_["parts"]
                vm = jax.vmap(R_["run"], in_axes=(0, 0) + (None,) * 10)
                if not a.lean:
                    build(calls, rep, B, f"ROM_VMAP|{name}", vm, [u0n, nuB], list(R_["full"]), B)
                sh = (R_["G"], R_["A"], R_["T"], R_["lam"], R_["Dd"], R_["p"], R_["Hc"], R_["Hn"], R_["Z"])
                build(calls, rep, B, f"ROM_SHARED|{name}", e2e_b, [u0n, nuB], list(sh), B)
                if a.rom_rfft:     # amendment A2 (exploratory): shift transforms with real FFTs
                    build(calls, rep, B, f"ROM_SHAREDR|{name}", R_["parts_r"][0], [u0n, nuB], list(sh), B)
                ce = None if a.lean else build(calls, rep, B, f"ROM_ENC|{name}", enc_b, [u0n],
                                                   [R_["G"], R_["p"], R_["Hc"], R_["Hn"], R_["Z"]], B)
                if ce is not None:
                    w0s, c0s = ce.full_output()
                    cl = build(calls, rep, B, f"ROM_LAT|{name}", lat_b, [w0s, c0s, nuB],
                               [R_["A"], R_["T"], R_["lam"], R_["Dd"], R_["p"]], B)
                    if cl is not None:
                        Ws, css = cl.full_output()
                        build(calls, rep, B, f"ROM_DEC|{name}", dec_b, [w0s, c0s, Ws, css], [R_["G"], R_["p"]], B)
                if rom_only:      # informational large batch: never chunked
                    for kk in [x for x in calls if x.endswith(name) and calls[x].chunked]:
                        del calls[kk]
            if not rom_only and not a.rematch_only:
                for key in solver_keys:
                    scheme, m, s = key; tag = f"{scheme}|{m}|{s}"
                    u0m = u0ms[m]
                    for var in (("rfft",) if a.lean else ("vmap", "rfft")):
                        fn, geom = solver(key, var)
                        vs = jax.vmap(fn, in_axes=(0, 0, None))
                        ve = jax.vmap(lambda u, v, g, fn=fn, m=m: P.spec_resample(fn(P.spec_resample(u, m), v, g), n),
                                      in_axes=(0, 0, None))
                        build(calls, rep, B, f"SOLVE_{var.upper()}|{tag}", vs, [u0m, nuB], [geom], B)
                        if not a.lean:
                            build(calls, rep, B, f"E2E_{var.upper()}|{tag}", ve, [u0S, nuB], [geom], B)
                    del u0m
        except Exception as e:  # report and continue with what compiled
            rep["failures"].append(f"B={B}: build failed: {type(e).__name__}: {str(e)[:300]}")
            log(rep["failures"][-1])
        for nm, c in calls.items():
            brep["callables"][nm] = dict(chunk=c.chunk, chunked=c.chunked, mem=c.mem, tried=c.chunks_tried)
        log(f"B={B}: {len(calls)} callables compiled ({time.time() - t_job:.0f}s); chunked: "
            + ", ".join(f"{k}:{c.chunk}" for k, c in calls.items() if c.chunked))
        dump()
        # ---- timing
        reps = a.reps_small if B <= 32 else a.reps_large
        try:
            samples = time_block(calls, reps, seed=1000 + B)
        except Exception as e:
            rep["failures"].append(f"B={B}: timing failed: {type(e).__name__}: {str(e)[:300]}")
            log(rep["failures"][-1]); samples = {}
        for nm, v in samples.items():
            v = np.asarray(v); med = float(np.median(v))
            brep["callables"][nm].update(reps_ms=v.tolist(), median_ms=med, min_ms=float(v.min()), max_ms=float(v.max()),
                                         samples_per_s=B / (med / 1e3), ms_per_1000=1000.0 * med / B)
            log(f"B={B} {nm}: {med:.2f} ms ({B / (med / 1e3):.1f}/s, {1000 * med / B / 1e3:.2f} s per 1000)"
                + (f" chunk {calls[nm].chunk}" if calls[nm].chunked else ""))
        dump()
        # ---- field agreement against the unchanged batch-1 code
        if B in (8, max(Bs)) or (B == 1):
            slots = sorted({0, B - 1, B // 2, min(1, B - 1)})
            chk = {}
            for nm, c in calls.items():
                kind, which = nm.split("|", 1)
                if kind in ("ROM_ENC", "ROM_LAT", "ROM_DEC"): continue
                try:
                    worst = 0.0
                    for s_ in slots:
                        cc = int(idx[s_])
                        if kind.startswith("ROM"):
                            ref = rom_ref(which, cc)
                        else:
                            sc, m_, st_ = which.split("|"); key = (sc, int(m_), int(st_))
                            ref = solve_ref(key, cc) if kind.startswith("SOLVE") else e2e_ref(key, cc)
                        worst = max(worst, relmax(c.slot(s_), ref)); del ref
                    chk[nm] = dict(slots=slots, max_rel_diff=worst)
                except Exception as e:
                    chk[nm] = dict(error=f"{type(e).__name__}: {str(e)[:200]}")
                gc.collect()
            rep["checks"][str(B)] = chk
            log(f"B={B} agreement: " + ", ".join(f"{k}:{v.get('max_rel_diff', v.get('error'))}" for k, v in chk.items()))
            dump()
        # ---- cohort agreement (B = 32 callables over all cases)
        if B == 32 and not a.no_cohort:
            ladder, _ = RK.merge_ladders([f"{NS}/fomF_0", f"{NS}/fomF_64"], cohort,
                                         sum(x.shape[0] for x in arrs), [n])
            rowof = {(r["scheme"], r["n"], r["steps"]): r for r in ladder}
            for nm, c in calls.items():
                kind, which = nm.split("|", 1)
                if kind not in ("ROM_VMAP", "ROM_SHARED", "ROM_SHAREDR", "E2E_VMAP", "E2E_RFFT"): continue
                errs = np.zeros((ncases, 6))
                for s0 in range(0, ncases, 32):
                    ids = np.arange(s0, min(s0 + 32, ncases))
                    xs = [jnp.asarray(np.stack([truth[i][0] if kind.startswith("ROM") else uS[i] for i in ids])),
                          jnp.asarray(nus[ids])]
                    if len(ids) < 32:
                        xs = [jnp.concatenate([x, x[:32 - len(ids)]]) for x in xs]
                    o = c.comp(*xs, *c.shared) if not c.chunked else jnp.concatenate(
                        [c.comp(*[x[q:q + c.chunk] for x in xs], *c.shared) for q in range(0, 32, c.chunk)])
                    for j, i in enumerate(ids):
                        errs[i] = np.asarray(P.rel_err(o[j], jnp.asarray(truth[i])))
                    del o, xs
                if kind.startswith("ROM"):
                    ref = np.asarray(mc["arms"][which]["errors"])[:ncases]
                    d = float(np.max(np.abs(errs - ref)))
                    rep["cohort"][nm] = dict(max_abs_err_diff=d, worst=float(errs[:, 1:].max()),
                                             fresh_worst=float(ref[:, 1:].max()), ok=d <= 1e-8)
                else:
                    sc, m_, st_ = which.split("|")
                    ref = np.asarray(rowof[(sc, int(m_), int(st_))]["stats"][str(n)]["per_case_worst"])[:ncases]
                    d = float(np.max(np.abs(errs[:, 1:].max(axis=1) - ref)))
                    rep["cohort"][nm] = dict(max_abs_err_diff=d, worst=float(errs[:, 1:].max()),
                                             ladder_worst=float(ref.max()), ok=d <= 1e-7)
                if not rep["cohort"][nm]["ok"]:
                    rep["failures"].append(f"cohort agreement {nm}: {d:.2e}")
                log(f"cohort {nm}: {rep['cohort'][nm]}")
                dump()
        del calls, u0ms; gc.collect()
        # ---- re-match sensitivity at B = 32
        if B == 32 and not a.no_rematch:
            rc = {}
            rin = {m: jnp.concatenate([P.spec_resample(u0S[q:q + 8], m) for q in range(0, B, 8)])
                   for m in sorted({k_[1] for k_ in rematch_keys})}       # all inputs before any compile
            for key in rematch_keys:
                scheme, m, s = key; tag = f"{scheme}|{m}|{s}"
                for var in ("vmap", "rfft"):
                    fn, geom = solver(key, var)
                    build(rc, rep, B, f"SOLVE_{var.upper()}|{tag}", jax.vmap(fn, in_axes=(0, 0, None)), [rin[m], nuB], [geom], B)
            try:
                smp = time_block(rc, a.reps_large, seed=77)
                for nm, v in smp.items():
                    med = float(np.median(v))
                    rep["rematch"][nm] = dict(reps_ms=list(map(float, v)), median_ms=med, chunk=rc[nm].chunk,
                                              samples_per_s=B / (med / 1e3), mem=rc[nm].mem)
                    log(f"rematch B=32 {nm}: {med:.2f} ms")
            except Exception as e:
                rep["failures"].append(f"rematch failed: {type(e).__name__}: {str(e)[:300]}"); log(rep["failures"][-1])
            del rc, rin; gc.collect(); dump()
        del u0n, nuB, u0S; gc.collect()
    rep["dropped_from_timing"] = DROPPED
    rep["status"] = "final" if not rep["failures"] and not DROPPED else "with-failures"
    dump()
    log(f"DONE status {rep['status']} failures {rep['failures']}")


if __name__ == "__main__":
    main()
