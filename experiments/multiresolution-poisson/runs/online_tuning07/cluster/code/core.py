"""Frozen current separable Poisson decoder, exact weak operators and FD-DST.

No Gaussian descriptors reach a neural model. Source and output are full host
nodal arrays; only interior source values affect the zero-Dirichlet equation.
The old ms_parametric module is used only for its generic Poisson family/operator.
"""
from pathlib import Path
import sys
import time
import hashlib
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
if (HERE / 'sep_common.py').exists():
    sys.path.insert(0, str(HERE))
else:
    ROOT = HERE.parents[1]
    for p in ('separable-decoder', 'cost-to-tolerance',
              'wave2d-rom-latent-stepping/deps/multistage-precision'):
        sys.path.insert(0, str(ROOT / 'experiments' / p))
import sep_common as sc
import ctol_tol
import ms_parametric as mp


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def relative(a, b):
    return float(np.linalg.norm(np.asarray(a)-np.asarray(b))/np.linalg.norm(b))


def dst1(x, axis=-1):
    """Orthonormal DST-I by odd extension; self inverse."""
    x = jnp.moveaxis(x, axis, -1)
    zero = jnp.zeros(x.shape[:-1]+(1,), x.dtype)
    ext = jnp.concatenate((zero, x, zero, -x[..., ::-1]), axis=-1)
    y = -jnp.fft.rfft(ext, axis=-1).imag[..., 1:x.shape[-1]+1]
    y = y / jnp.sqrt(2.0*(x.shape[-1]+1))
    return jnp.moveaxis(y, -1, axis)


def dst2(x):
    return dst1(dst1(x, 0), 1)


def eigenvalues(intervals):
    p = np.arange(1, intervals)
    l = 4.0*intervals**2*np.sin(np.pi*p/(2*intervals))**2
    return l[:, None]+l[None, :]


@jax.jit
def dst_solve(source, lam):
    return jnp.pad(dst2(dst2(source[1:-1, 1:-1])/lam), 1)


def full_source(intervals, param):
    # Match inherited source_interior exactly; boundary RHS values are unused.
    return np.pad(mp.source_interior(intervals+1, *param), 1)


def observation(field, intervals, observation_intervals):
    assert intervals % observation_intervals == 0
    return np.asarray(field)[::intervals//observation_intervals,
                             ::intervals//observation_intervals]


def source_params(seed, count):
    cx, cy, width, amp, _unused_descriptor = mp.sample_params(seed, count)
    return np.column_stack((cx, cy, width, amp))


def burn(seconds):
    a = jnp.ones((384, 384), dtype=jnp.float64)*0.001
    fn = jax.jit(lambda x: x@x+0.0001)
    fn(a).block_until_ready()
    end = time.perf_counter()+seconds
    while time.perf_counter() < end:
        fn(a).block_until_ready()


def assemble(params, ztrain, intervals, modes, budget):
    start = time.perf_counter()
    p = np.arange(1, intervals)
    x = p/intervals
    xx, yy = np.meshgrid(x, x, indexing='ij')
    coords = np.column_stack((xx.ravel(), yy.ravel()))
    # Network arrays are explicit JIT arguments: no captured large constants.
    feat = jax.jit(sc.features)
    chunks = []
    for s in range(0, len(coords), 16384):
        chunk = feat(params, jnp.asarray(coords[s:s+16384]))
        chunk.block_until_ready()
        chunks.append(chunk)
    bank = jnp.concatenate(chunks, axis=0)
    bank.block_until_ready()
    bank_seconds = time.perf_counter()-start
    start = time.perf_counter()
    lam = eigenvalues(intervals)
    I, J = np.nonzero(lam <= np.sort(lam.ravel())[modes-1])
    maxmode = int(max(I.max(), J.max()))+1
    S = np.sqrt(2.0/intervals)*np.sin(np.pi*np.outer(p, np.arange(1,maxmode+1))/intervals)
    Sj, Ij, Jj = jnp.asarray(S), jnp.asarray(I), jnp.asarray(J)
    W = jnp.asarray(lam[I, J]**-1)

    @jax.jit
    def project(F, S, ii, jj, weight):
        c = S.T@F[1:-1,1:-1]@S
        return c[ii,jj]*weight

    @jax.jit
    def build(G, S, ii, jj):
        cubes = G.reshape((intervals-1, intervals-1, G.shape[-1]))
        c = jnp.einsum('xa,xyr,yb->abr', S, cubes, S)
        return c[ii,jj]

    B = build(bank, Sj, Ij, Jj)
    B.block_until_ready()
    assembly_seconds = time.perf_counter()-start
    z0 = jnp.asarray(ztrain.mean(0))
    trust = float(np.linalg.norm(ztrain-ztrain.mean(0), axis=1).max())
    # B is small; only small reduced arrays enter the incumbent solver closure.
    h = lambda z: sc.head(params, z)
    solve, _ = ctol_tol.lm_tau_poisson(lambda z, xy: h(z), len(z0),
        np.zeros((B.shape[1],2)), np.ones(B.shape[1]), B,
        np.ones(B.shape[0]), budget, trust)
    @jax.jit
    def decode(z, G, weights):
        return jnp.pad((G@sc.head(weights,z)).reshape((intervals-1,intervals-1)), 1)
    @jax.jit
    def stationary(z, fm, B, weights):
        fun = lambda zz: B@sc.head(weights,zz)-fm
        r = fun(z)
        jac = jax.jacfwd(fun)(z)
        return jnp.linalg.norm(jac.T@r)/(jnp.linalg.norm(jac)*jnp.linalg.norm(r)+1e-300)
    # Gram singular values alone square the condition number: compute direct SVD
    # on the tall matrix, which is small in feature dimension and setup-only.
    singular = np.asarray(jnp.linalg.svd(bank, full_matrices=False, compute_uv=False))
    threshold = singular[0]*max(bank.shape)*np.finfo(float).eps
    info = dict(intervals=intervals, nodes_per_axis=intervals+1,
        interior_unknowns=(intervals-1)**2, boundary_nodes=4*intervals,
        requested_modes=modes, retained_modes=len(I), mode_indices=np.column_stack((I+1,J+1)).tolist(),
        stored_features=int(bank.shape[1]), retained_bank_rank=int((singular>threshold).sum()),
        singular_values=singular.tolist(), rank_threshold=float(threshold),
        bank_bytes=int(bank.size*bank.dtype.itemsize), reduced_matrix_bytes=int(B.size*B.dtype.itemsize),
        source_projection_bytes=int(S.nbytes+I.nbytes+J.nbytes+W.size*8),
        output_bytes=int((intervals+1)**2*8), bank_build_seconds=bank_seconds,
        weak_assembly_seconds=assembly_seconds, trust_delta=trust,
        operator_sha256=sha(B), bank_sha256=sha(bank))
    return dict(bank=bank, B=B, S=Sj, I=Ij, J=Jj, W=W, z0=z0,
        project=project, solve=solve, decode=decode, stationary=stationary,
        params=params, info=info, intervals=intervals)


def rom_query(host_source, ops, tau):
    """Synchronized component accounting inside the same total timed invocation."""
    start = time.perf_counter()
    src = jax.device_put(host_source)
    src.block_until_ready()
    input_end = time.perf_counter()
    fm = ops['project'](src, ops['S'], ops['I'], ops['J'], ops['W'])
    fm.block_until_ready()
    project_end = time.perf_counter()
    ans = ops['solve'](ops['z0'], fm, jnp.asarray(tau))
    jax.block_until_ready(ans)
    solve_end = time.perf_counter()
    field = ops['decode'](ans[0], ops['bank'], ops['params'])
    # Copy all requested nodal output and solve metadata to host before stop.
    field, host_ans = jax.device_get((field, ans))
    end = time.perf_counter()
    # Stationarity is validation-only and explicitly excluded from query cost.
    stationary = float(ops['stationary'](ans[0],fm,ops['B'],ops['params']))
    z, residual, initial, njac, accepted, attempts, reason = host_ans
    return np.asarray(field), dict(total_seconds=end-start,
        input_seconds=input_end-start, projection_init_seconds=project_end-input_end,
        solver_seconds=solve_end-project_end, output_seconds=end-solve_end,
        reason=int(reason), attempts=int(attempts), accepted=int(accepted), jacobians=int(njac),
        residual=float(residual), initial_residual=float(initial),
        stationarity=stationary, latent=np.asarray(z).tolist())


def fom_query(host_source, lam):
    start = time.perf_counter()
    source = jax.device_put(host_source)
    source.block_until_ready()
    input_end = time.perf_counter()
    field = dst_solve(source, lam)
    field.block_until_ready()
    solve_end = time.perf_counter()
    field = np.asarray(jax.device_get(field))
    end = time.perf_counter()
    return field, dict(total_seconds=end-start, input_seconds=input_end-start,
        projection_init_seconds=0., solver_seconds=solve_end-input_end,
        output_seconds=end-solve_end, reason=0, attempts=1, stationarity=None)


@jax.jit
def dst_coarse_solve(source, lam):
    coarse=lam.shape[0]+1
    requested=source.shape[0]-1
    factor=requested//coarse
    u=dst_solve(source[::factor,::factor],lam)
    old=jnp.linspace(0.,1.,coarse+1)
    new=jnp.linspace(0.,1.,requested+1)
    along_x=jax.vmap(lambda col:jnp.interp(new,old,col),in_axes=1,out_axes=1)(u)
    return jax.vmap(lambda row:jnp.interp(new,old,row))(along_x)


def coarse_query(host_source, lam):
    start=time.perf_counter()
    source=jax.device_put(host_source)
    source.block_until_ready()
    input_end=time.perf_counter()
    field=dst_coarse_solve(source,lam)
    field.block_until_ready()
    solve_end=time.perf_counter()
    field=np.asarray(jax.device_get(field))
    end=time.perf_counter()
    return field,dict(total_seconds=end-start,input_seconds=input_end-start,
        projection_init_seconds=0.,solver_seconds=solve_end-input_end,
        output_seconds=end-solve_end,reason=0,attempts=1,stationarity=None,
        solve_intervals=int(lam.shape[0]+1),interpolation='bilinear nodal, charged in solver_seconds')
