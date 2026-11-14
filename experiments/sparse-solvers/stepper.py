"""Measured implicit Burgers time-stepper on cuDSS (PROTOCOL.md amendment 3, Task A).

Backward Euler for the paper's Burgers full-order discretisation (L intervals per axis, (L-1)^dim interior unknowns,
zero Dirichlet walls, sign-upwind advection switching on the centre value, standard Laplacian):

    r(u) = u - u_prev + dt * ( u * L * sum_a upw_a(u)  -  nu * L^2 * sum_a (u_- - 2u + u_+) )

Newton with the sparse Jacobian factorised by cuDSS (general LU).  Variants: full, k1, k5, k10, frozen, lin1.
One process = one (dim, L); the JSON is rewritten after every setting.

  python stepper.py --dim 2 --L 256 --cases 64 --refs refs --out results/b2d_L256.json --budget 3000
"""
import argparse
import ctypes
import glob
import json
import os
import time
from pathlib import Path

import numpy as np

from bench import (CFG_HOST_NTHREADS, CUDA_R_32I, CUDA_R_64F, CUDSS_STATUS, DATA_LU_NNZ, DATA_MEMORY_ESTIMATES, MTYPE_GENERAL,
                   MVIEW_FULL, PH_ANALYSIS, PH_FACT, PH_REFACT, PH_SOLVE, MemPoll, Rec, stats)
import matrices

KERNELS = r'''
extern "C" __global__
void neg_residual(const double* u, const double* prev, double* b, double nu, double dt, double L, int m, int dim, long long N) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    double c = u[i], adv = 0.0, lap = 0.0;
    long long s = 1, idx = i;
    for (int a = dim - 1; a >= 0; --a) {
        int k = (int)(idx % m); idx /= m;
        double um = k > 0 ? u[i - s] : 0.0;
        double up = k < m - 1 ? u[i + s] : 0.0;
        adv += c > 0 ? c - um : up - c;
        lap += um + up - 2.0 * c;
        s *= m;
    }
    b[i] = -(c - prev[i] + dt * (c * L * adv - nu * L * L * lap));
}

extern "C" __global__
void jacobian(const double* u, const int* indptr, double* data, double nu, double dt, double L, int m, int dim, long long N) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    double c = u[i], g = 0.0;
    int k[3]; long long st[3];
    long long s = 1, idx = i;
    for (int a = dim - 1; a >= 0; --a) { k[a] = (int)(idx % m); idx /= m; st[a] = s; s *= m; }
    double dif = dt * nu * L * L;
    double lo = -dif - (c > 0 ? dt * L * c : 0.0);
    double hi = -dif + (c > 0 ? 0.0 : dt * L * c);
    int p = indptr[i];
    for (int a = 0; a < dim; ++a) {            // columns i - st[a], increasing column order
        double um = k[a] > 0 ? u[i - st[a]] : 0.0;
        double up = k[a] < m - 1 ? u[i + st[a]] : 0.0;
        g += c > 0 ? c - um : up - c;
        if (k[a] > 0) data[p++] = lo;
    }
    data[p++] = 1.0 + dt * L * (g + dim * fabs(c)) + 2.0 * dim * dif;
    for (int a = dim - 1; a >= 0; --a) if (k[a] < m - 1) data[p++] = hi;
}
'''

VARIANTS = ('full', 'k1', 'k5', 'k10', 'frozen', 'lin1')
CAP = {'full': 20}          # paper's Newton cap; the factorisation-reusing variants get 100


class Lib:
    def __init__(self):
        libdir = Path(glob.glob(os.path.join(os.environ['SS_PYDEPS'], 'nvidia', 'cu12', 'lib'))[0])
        lib = ctypes.CDLL(str(libdir / 'libcudss.so.1'), mode=ctypes.RTLD_GLOBAL)
        vp, i64, ci = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int
        lib.cudssMatrixCreateCsr.argtypes = [ctypes.POINTER(vp), i64, i64, i64, vp, vp, vp, vp, ci, ci, ci, ci, ci, ci]
        lib.cudssMatrixCreateDn.argtypes = [ctypes.POINTER(vp), i64, i64, i64, vp, ci, ci]
        lib.cudssExecute.argtypes = [vp, ci, vp, vp, vp, vp, vp]
        lib.cudssDataGet.argtypes = [vp, vp, ci, vp, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
        lib.cudssConfigSet.argtypes = [vp, ci, vp, ctypes.c_size_t]
        lib.cudssDataCreate.argtypes = [vp, ctypes.POINTER(vp)]
        lib.cudssDataDestroy.argtypes = [vp, vp]
        lib.cudssMatrixDestroy.argtypes = [vp]
        lib.cudssSetThreadingLayer.argtypes = [vp, ctypes.c_char_p]
        lib.cudssMatrixSetValues.argtypes = [vp, vp]
        self.lib, self.vp = lib, vp
        ver = [ctypes.c_int() for _ in range(3)]
        for k, v in enumerate(ver):
            lib.cudssGetProperty(k, ctypes.byref(v))
        self.version = '.'.join(str(v.value) for v in ver)
        self.nthreads = int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))
        self.handle, self.config = vp(), vp()
        self.chk(lib.cudssCreate(ctypes.byref(self.handle)), 'create')
        self.chk(lib.cudssSetThreadingLayer(self.handle, str(libdir / 'libcudss_mtlayer_gomp.so.1').encode()), 'threading layer')
        self.chk(lib.cudssConfigCreate(ctypes.byref(self.config)), 'config')
        nt = ctypes.c_int(self.nthreads)
        self.chk(lib.cudssConfigSet(self.config, CFG_HOST_NTHREADS, ctypes.byref(nt), ctypes.sizeof(nt)), 'host threads')

    @staticmethod
    def chk(st, what):
        if st != 0:
            raise RuntimeError(f'{what}: cudss status {st} ({CUDSS_STATUS[st] if st < len(CUDSS_STATUS) else "?"})')


class Mesh:
    """One mesh level: pattern, kernels, cuDSS analysis and first factorisation (one-off setup), then steppers."""

    def __init__(self, cp, L, dim, lib, poll):
        self.cp, self.L, self.dim, self.lib = cp, L, dim, lib
        self.m = m = L - 1
        self.N = N = m ** dim
        dev = self.dev = cp.cuda.Device(0)
        setup = {}
        t = time.perf_counter()
        A = matrices._assemble(m, dim, 1.0, [1.0] * dim, [1.0] * dim)
        setup['pattern_build_cpu_s'] = time.perf_counter() - t
        self.nnz = int(A.nnz)
        self.indptr, self.indices = cp.asarray(A.indptr), cp.asarray(A.indices)
        self.data = cp.zeros(self.nnz, dtype=cp.float64)
        self.indptr_host, self.indices_host = A.indptr, A.indices
        del A
        self.u = cp.zeros(N); self.prev = cp.zeros(N); self.b = cp.zeros(N); self.x = cp.zeros(N)
        t = time.perf_counter()
        mod = cp.RawModule(code=KERNELS)
        self.k_res, self.k_jac = mod.get_function('neg_residual'), mod.get_function('jacobian')
        self.grid = ((N + 255) // 256,)
        self.assemble(self.u, 0.01, 0.005)
        self.neg_residual(0.01, 0.005)
        float(cp.linalg.norm(self.b)); self.u += self.x; cp.copyto(self.u, self.prev)
        dev.synchronize()
        setup['kernel_compile_s'] = time.perf_counter() - t
        l, vp = lib.lib, lib.vp
        self.mA, self.mx, self.mb, self.d = vp(), vp(), vp(), vp()
        lib.chk(l.cudssMatrixCreateCsr(ctypes.byref(self.mA), N, N, self.nnz, self.indptr.data.ptr, None, self.indices.data.ptr,
                                       self.data.data.ptr, CUDA_R_32I, CUDA_R_32I, CUDA_R_64F, MTYPE_GENERAL, MVIEW_FULL, 0),
                'matrix csr')
        lib.chk(l.cudssMatrixCreateDn(ctypes.byref(self.mx), N, 1, N, self.x.data.ptr, CUDA_R_64F, 0), 'dense x')
        lib.chk(l.cudssMatrixCreateDn(ctypes.byref(self.mb), N, 1, N, self.b.data.ptr, CUDA_R_64F, 0), 'dense b')
        lib.chk(l.cudssDataCreate(lib.handle, ctypes.byref(self.d)), 'data create')
        base = poll.used(); poll.reset()
        setup['analysis_s'] = self._phase(PH_ANALYSIS, 'analysis')
        est = (ctypes.c_int64 * 16)(); w = ctypes.c_size_t(); lunnz = ctypes.c_int64()
        l.cudssDataGet(lib.handle, self.d, DATA_MEMORY_ESTIMATES, est, ctypes.sizeof(est), ctypes.byref(w))
        l.cudssDataGet(lib.handle, self.d, DATA_LU_NNZ, ctypes.byref(lunnz), 8, ctypes.byref(w))
        setup['first_factor_s'] = self._phase(PH_FACT, 'factor')
        setup.update(factor_nnz=int(lunnz.value), cudss_memory_estimates_bytes=list(est)[:6],
                     peak_device_bytes_setup=int(poll.peak - base), device_bytes_held_after_factor=int(poll.used() - base))
        self.setup = setup
        self.counts = dict(refactor=0, solve=0)
        self.tim = None  # instrumented pass: dict(refactor=.., solve=..)

    def _phase(self, ph, name):
        self.dev.synchronize()
        t = time.perf_counter()
        st = self.lib.lib.cudssExecute(self.lib.handle, ph, self.lib.config, self.d, self.mA, self.mx, self.mb)
        self.dev.synchronize()
        self.lib.chk(st, name)
        return time.perf_counter() - t

    def close(self):
        l = self.lib.lib
        l.cudssDataDestroy(self.lib.handle, self.d)
        for mtx in (self.mA, self.mx, self.mb):
            l.cudssMatrixDestroy(mtx)

    # -- kernels
    def neg_residual(self, nu, dt):
        self.k_res(self.grid, (256,), (self.u, self.prev, self.b, np.float64(nu), np.float64(dt), np.float64(self.L),
                                       np.int32(self.m), np.int32(self.dim), np.int64(self.N)))

    def assemble(self, u, nu, dt):
        self.k_jac(self.grid, (256,), (u, self.indptr, self.data, np.float64(nu), np.float64(dt), np.float64(self.L),
                                       np.int32(self.m), np.int32(self.dim), np.int64(self.N)))

    def refactor(self, nu, dt):
        if self.tim is not None:
            self.dev.synchronize(); t = time.perf_counter()
        self.assemble(self.u, nu, dt)
        st = self.lib.lib.cudssMatrixSetValues(self.mA, self.data.data.ptr)
        st = st or self.lib.lib.cudssExecute(self.lib.handle, PH_REFACT, self.lib.config, self.d, self.mA, self.mx, self.mb)
        if st:
            self.lib.chk(st, 'refactor')
        self.counts['refactor'] += 1
        if self.tim is not None:
            self.dev.synchronize(); self.tim['refactor'] += time.perf_counter() - t

    def solve(self):
        if self.tim is not None:
            self.dev.synchronize(); t = time.perf_counter()
        st = self.lib.lib.cudssExecute(self.lib.handle, PH_SOLVE, self.lib.config, self.d, self.mA, self.mx, self.mb)
        if st:
            self.lib.chk(st, 'solve')
        self.counts['solve'] += 1
        if self.tim is not None:
            self.dev.synchronize(); self.tim['solve'] += time.perf_counter() - t

    # -- time stepping; the state lives in self.u
    def advance(self, variant, nu, dt, ntol, nsteps, step0, rec):
        """nsteps steps from self.u; step0 = index of the first step in the trajectory (for the refactor schedule)."""
        cp = self.cp
        u, prev, x, b = self.u, self.prev, self.x, self.b
        if variant == 'lin1':
            for _ in range(nsteps):
                cp.copyto(prev, u)
                self.neg_residual(nu, dt)
                self.refactor(nu, dt)
                self.solve()
                u += x
                rec['newton'].append(1)
            return
        cap = CAP.get(variant, 100)
        every = {'full': 1, 'k1': 1, 'k5': 5, 'k10': 10, 'frozen': 1 << 30}[variant]
        for s in range(step0, step0 + nsteps):
            cp.copyto(prev, u)
            scale = max(float(cp.linalg.norm(prev)), 1e-300)
            self.neg_residual(nu, dt)
            rn = float(cp.linalg.norm(b))
            it = 0
            if s % every == 0:
                self.pending = True      # a refactorisation is due; done at the next iteration that is actually needed
            while rn > ntol * scale and it < cap and np.isfinite(rn):
                if variant == 'full' or self.pending:
                    self.refactor(nu, dt)
                    self.pending = False
                self.solve()
                u += x
                self.neg_residual(nu, dt)
                rn = float(cp.linalg.norm(b))
                it += 1
            rec['newton'].append(it)
            if not (rn <= ntol * scale):
                rec['unconverged'] += 1
            rec['max_rel_res'] = max(rec['max_rel_res'], rn / scale if np.isfinite(rn) else float('inf'))


class Runner:
    def __init__(self, cp, dim, L, lib, poll):
        self.cp, self.dim, self.L, self.lib, self.poll = cp, dim, L, lib, poll
        self.meshes, self.interp = {}, {}
        self.out = cp.zeros((6,) + (L + 1,) * dim)
        self.dev = cp.cuda.Device(0)

    def mesh(self, f):
        if f not in self.meshes:
            self.meshes[f] = Mesh(self.cp, self.L // f, self.dim, self.lib, self.poll)
            if f > 1:
                Lc = self.L // f
                xx = self.cp.arange(self.L + 1, dtype=self.cp.float64) * Lc / self.L
                lo = self.cp.minimum(xx.astype(self.cp.int32), Lc - 1)
                self.interp[f] = (lo, xx - lo, self.cp.zeros((Lc + 1,) * self.dim))
        return self.meshes[f]

    def write(self, k, M, f):
        """State of mesh M (interior, flat) -> output field k on the target mesh (aligned multilinear interpolation)."""
        cp, dim = self.cp, self.dim
        inner = (slice(1, -1),) * dim
        if f == 1:
            self.out[k][inner] = M.u.reshape((M.m,) * dim)
            return
        lo, fr, a = self.interp[f]
        a[inner] = M.u.reshape((M.m,) * dim)
        for ax in range(dim):
            sh = [1] * dim; sh[ax] = -1
            w = fr.reshape(sh)
            a = (1 - w) * cp.take(a, lo, axis=ax) + w * cp.take(a, lo + 1, axis=ax)
        self.out[k] = a

    def trajectory(self, u0, nu, variant, dt, ntol, f, steps_per_out=None, nout=5):
        """u0: initial field on the target mesh (device).  Returns (seconds, record); fields are in self.out."""
        M = self.mesh(f)
        dim = self.dim
        sub = int(round(0.05 / dt)) if steps_per_out is None else steps_per_out
        rec = dict(newton=[], unconverged=0, max_rel_res=0.0)
        M.counts = dict(refactor=0, solve=0)
        M.pending = True                 # never reuse a factorisation from an earlier trajectory
        self.dev.synchronize()
        t = time.perf_counter()
        M.u[...] = u0[(slice(None, None, f),) * dim][(slice(1, -1),) * dim].ravel()
        self.write(0, M, f)
        for k in range(nout):
            M.advance(variant, nu, dt, ntol, sub, k * sub, rec)
            self.write(k + 1, M, f)
        self.dev.synchronize()
        sec = time.perf_counter() - t
        rec.update(M.counts)
        return sec, rec


def fd_check(cp, M, nu, dt, u):
    """Relative difference between J v (assembled CSR) and a central finite difference of the residual kernel."""
    import cupyx.scipy.sparse as csp
    rng = cp.random.default_rng(1)
    v = rng.standard_normal(M.N)
    M.assemble(u, nu, dt)
    J = csp.csr_matrix((M.data.copy(), M.indices, M.indptr), shape=(M.N, M.N))
    jv = J @ v
    eps = 1e-6
    cp.copyto(M.prev, u)
    out = []
    for sgn in (1, -1):
        cp.copyto(M.u, u + sgn * eps * v)
        M.neg_residual(nu, dt)
        out.append(-M.b.copy())
    fd = (out[0] - out[1]) / (2 * eps)
    return float(cp.linalg.norm(jv - fd) / cp.linalg.norm(jv))


def settings_list(dim):
    same = [dict(variant=v, dt=0.005, ntol=nt, coarsen=1, group='same-mesh')
            for nt in (1e-6, 3e-3) for v in ('k1', 'frozen', 'k5', 'k10', 'full')]
    same.insert(0, dict(variant='lin1', dt=0.005, ntol=None, coarsen=1, group='same-mesh'))
    grid = []
    for f in (4, 2, 1):
        for dt in (0.05, 0.025, 0.01, 0.005):
            for v, nts in (('frozen', (1e-2, 1e-3)), ('lin1', (None,)), ('k1', (1e-2, 1e-3))):
                for nt in nts:
                    grid.append(dict(variant=v, dt=dt, ntol=nt, coarsen=f, group='matched-grid'))
    return same, grid


def name_of(s):
    nt = 'none' if s['ntol'] is None else f"{s['ntol']:g}"
    return f"{s['variant']}_dt{s['dt']:g}_ntol{nt}_c{s['coarsen']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dim', type=int, required=True)
    ap.add_argument('--L', type=int, required=True)
    ap.add_argument('--cases', type=int, required=True)
    ap.add_argument('--refs', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--budget', type=float, default=3600.0, help='wall seconds for the matched grid (same-mesh always runs)')
    ap.add_argument('--smoke', action='store_true', help='tiny settings list (pipeline test)')
    a = ap.parse_args()
    import cupy as cp
    t_start = time.time()
    dim, L, K = a.dim, a.L, a.cases
    refs = Path(a.refs)
    meta = json.loads((refs / f'b{dim}d_L{L}.json').read_text())
    rec = Rec(a.out, dict(task='burgers_stepper', dim=dim, L=L, unknowns=(L - 1) ** dim, cases=K, dtype='float64', solver='cudss',
                          reference=meta, cupy=cp.__version__, budget_s=a.budget))
    poll = MemPoll(cp)
    lib = Lib()
    rec.save(cudss_version=lib.version, host_threads=lib.nthreads, mtype='GENERAL', reordering='default')
    R = Runner(cp, dim, L, lib, poll)
    nus = [r['nu'] for r in meta['case_records']][:K]
    ref = [cp.asarray(np.load(refs / f'b{dim}d_L{L}_case{c}.npy')) for c in range(K)]
    shape = (L + 1,) * dim
    u0s = [r[0].reshape(shape).copy() for r in ref]
    n0 = [float(cp.linalg.norm(r[0])) for r in ref]

    def error(c):
        o = R.out.reshape(6, -1)
        return max(float(cp.linalg.norm(o[k] - ref[c][k])) / n0[c] for k in range(1, 6))

    stage = 'setup'
    try:
        M = R.mesh(1)
        rec.save(setup={'1': M.setup}, nnz=M.nnz)
        # ---- checks before any timing: Jacobian kernel vs finite differences; tight stepper vs the paper's reference
        stage = 'checks'
        u_mid = ref[0][3].reshape(shape)[(slice(1, -1),) * dim].ravel().copy()
        fd = fd_check(cp, M, nus[0], 0.005, u_mid)
        _, r0 = R.trajectory(u0s[0], nus[0], 'full', 0.005, 1e-9, 1)
        e_tight = error(0)
        # first field must equal the reference initial field exactly
        e_ic = float(cp.linalg.norm(R.out.reshape(6, -1)[0] - ref[0][0])) / n0[0]
        checks = dict(jacobian_fd_relative=fd, tight_full_newton_vs_reference=e_tight, initial_field_difference=e_ic,
                      tight_unconverged_steps=r0['unconverged'], tight_newton_total=int(sum(r0['newton'])),
                      passed=bool(fd < 1e-6 and e_tight < 2e-4 and e_ic == 0.0))
        rec.save(checks=checks)
        print('checks', checks, flush=True)
        if not checks['passed']:
            raise RuntimeError('pre-timing checks failed')

        same, grid = settings_list(dim)
        if a.smoke:
            same, grid = same[:3], [g for g in grid if g['dt'] == 0.05][:6]
        results = {}
        rec.save(settings=results)

        def run_setting(s):
            nm = name_of(s)
            v, dt, f = s['variant'], s['dt'], s['coarsen']
            nt = 0.0 if s['ntol'] is None else s['ntol']
            Mf = R.mesh(f)
            setup = rec.d['setup']
            if str(f) not in setup:
                setup[str(f)] = Mf.setup
            R.trajectory(u0s[0], nus[0], v, dt, nt, f, steps_per_out=1, nout=2)     # warm-up, untimed
            times, rows, reps = [], [], None
            for c in range(K):
                k = 0
                row = None
                while reps is None or k < reps:
                    sec, r = R.trajectory(u0s[c], nus[c], v, dt, nt, f)
                    if reps is None:
                        reps = 3 if sec <= 5.0 else 1
                    times.append(sec)
                    if row is None:
                        row = dict(case=c, error=error(c), newton_total=int(sum(r['newton'])), newton_max_step=int(max(r['newton'])),
                                   refactor=r['refactor'], solve=r['solve'], unconverged_steps=r['unconverged'],
                                   max_rel_residual=r['max_rel_res'], times=[])
                    row['times'].append(sec)
                    k += 1
                rows.append(row)
            errs = [r['error'] for r in rows]
            res = dict(s, name=nm, steps=int(round(0.25 / dt)), mesh_L=L // f, reps_per_case=reps, time_s=stats(times),
                       worst_error=float(np.max(errs)) if np.isfinite(errs).all() else float('inf'),
                       median_error=float(np.median(errs)), newton_total_median=float(np.median([r['newton_total'] for r in rows])),
                       newton_total_max=int(max(r['newton_total'] for r in rows)),
                       refactor_median=float(np.median([r['refactor'] for r in rows])),
                       solve_median=float(np.median([r['solve'] for r in rows])),
                       unconverged_steps_total=int(sum(r['unconverged_steps'] for r in rows)),
                       cases_with_unconverged=int(sum(r['unconverged_steps'] > 0 for r in rows)),
                       finite=bool(np.isfinite(errs).all()), status='ok', per_case=rows)
            if s['group'] == 'same-mesh':        # instrumented trajectory, case 0: split of the time
                Mf.tim = dict(refactor=0.0, solve=0.0)
                sec, r = R.trajectory(u0s[0], nus[0], v, dt, nt, f)
                res['instrumented_case0'] = dict(total_s=sec, refactor_s=Mf.tim['refactor'], solve_s=Mf.tim['solve'],
                                                 rest_s=sec - Mf.tim['refactor'] - Mf.tim['solve'],
                                                 note='synchronised around every refactorisation and solve; slower than the timed runs')
                Mf.tim = None
            results[nm] = res
            rec.save(settings=results, peak_device_bytes=int(poll.peak), elapsed_s=time.time() - t_start)
            ts = res['time_s']
            print(f"{time.strftime('%T')} {nm}: {1e3 * ts['median']:.2f} ms [{1e3 * ts['min']:.2f}, {1e3 * ts['max']:.2f}] "
                  f"worst {100 * res['worst_error']:.4f}% median {100 * res['median_error']:.4f}% newton {res['newton_total_median']:.0f} "
                  f"refac {res['refactor_median']:.0f} unconv {res['unconverged_steps_total']}", flush=True)

        stage = 'same-mesh'
        for s in same:
            stage = name_of(s)
            run_setting(s)
        t_grid = time.time()
        for s in grid:
            stage = name_of(s)
            if stage in results:
                continue
            if time.time() - t_grid > a.budget:
                results[stage] = dict(s, name=stage, status='not run (budget)')
                continue
            run_setting(s)
        rec.save(settings=results, status='ok', peak_device_bytes=int(poll.peak), elapsed_s=time.time() - t_start)
    except Exception as e:  # noqa: BLE001
        rec.save(status='failed', failed_stage=stage, error=repr(e), device_used_bytes_at_failure=int(poll.used()),
                 peak_device_bytes_at_failure=int(poll.peak), device_total_bytes=int(poll.total))
        print('FAILED', stage, repr(e), flush=True)
    poll.stop = True


if __name__ == '__main__':
    main()
