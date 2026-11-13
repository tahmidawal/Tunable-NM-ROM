"""Solve-only GPU benchmark workers: cuDSS (ctypes on libcudss), AmgX (ctypes on libamgxsh), CuPy Jacobi-PCG.

One process = one (problem, dim, n, solver).  The JSON is rewritten after every stage, so a crash
(e.g. device out-of-memory abort) leaves the stages finished so far on disk.

  python bench.py worker --solver cudss|amgx|cg --problem poisson|heat|burgers --dim 2 --n 256 --out file.json
  python bench.py driver --tasks tasks.txt --outdir results/<tag>
"""
import argparse
import ctypes
import glob
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPS = 5
TOLS = (1e-6, 1e-10)
AMGX_SPD = ['PCG_CLASSICAL_V_JACOBI', 'PCG_AGGREGATION_JACOBI', 'FGMRES_CLASSICAL_AGGRESSIVE_PMIS']
AMGX_GEN = ['FGMRES_AGGREGATION_JACOBI', 'PBICGSTAB_CLASSICAL_JACOBI', 'FGMRES_CLASSICAL_AGGRESSIVE_PMIS',
            'PBICGSTAB_AGGREGATION_W_JACOBI']


def nreps(first_seconds):
    """Protocol repeat rule: 5 repeats; 3 if one timed unit exceeds 60 s; 1 if it exceeds 600 s."""
    return REPS if first_seconds <= 60 else (3 if first_seconds <= 600 else 1)


def stats(ts):
    ts = [float(t) for t in ts]
    return {'median': float(np.median(ts)), 'min': min(ts), 'max': max(ts), 'n': len(ts), 'all': ts} if ts else None


def gpu_info():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001
        out = f'nvidia-smi failed: {e}'
    return out


class Rec:
    def __init__(self, path, meta):
        self.path, self.d = Path(path), dict(meta)
        self.d.update(status='started', host=os.uname().nodename, gpu=gpu_info(), slurm_job=os.environ.get('SLURM_JOB_ID'),
                      cpus=os.environ.get('SLURM_CPUS_PER_TASK'), started=time.strftime('%F %T %Z'))
        self.save()

    def save(self, **kw):
        self.d.update(kw)
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.d, indent=1))
        tmp.replace(self.path)


class MemPoll:
    """Peak device memory in use (all processes on the device) sampled every 5 ms via cudaMemGetInfo."""

    def __init__(self, cp):
        self.cp, self.peak, self.stop = cp, 0, False
        free, total = cp.cuda.runtime.memGetInfo()
        self.base, self.total = total - free, total
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        self.cp.cuda.Device(0).use()
        while not self.stop:
            free, total = self.cp.cuda.runtime.memGetInfo()
            self.peak = max(self.peak, total - free)
            time.sleep(0.005)

    def reset(self):
        self.peak = 0

    def used(self):
        free, total = self.cp.cuda.runtime.memGetInfo()
        return total - free


def load_matrix(a):
    import matrices
    t = time.perf_counter()
    A = matrices.build(a.problem, a.n, a.dim)
    return A, matrices.SPD[a.problem], time.perf_counter() - t


def meta(a, A, spd, tb):
    import scipy
    return dict(problem=a.problem, dim=a.dim, n=a.n, unknowns=int(A.shape[0]), nnz=int(A.nnz), spd=bool(spd), solver=a.solver,
                rhs='ones', dtype='float64', build_seconds_cpu=tb, numpy=np.__version__, scipy=scipy.__version__)


# ----------------------------------------------------------------------------------------------- cuDSS
PH_ANALYSIS, PH_FACT, PH_REFACT, PH_SOLVE = 3, 4, 8, 1008
CUDA_R_64F, CUDA_R_32I = 1, 10
MTYPE_GENERAL, MTYPE_SPD, MVIEW_FULL, MVIEW_UPPER = 0, 3, 0, 2
CFG_HOST_NTHREADS, DATA_LU_NNZ, DATA_MEMORY_ESTIMATES = 14, 1, 13
CUDSS_STATUS = ['SUCCESS', 'NOT_INITIALIZED', 'ALLOC_FAILED', 'INVALID_VALUE', 'NOT_SUPPORTED', 'EXECUTION_FAILED',
                'INTERNAL_ERROR', 'IR_FAILED']


def run_cudss(a):
    import cupy as cp
    import cupyx.scipy.sparse as csp
    A, spd, tb = load_matrix(a)
    rec = Rec(a.out, meta(a, A, spd, tb))
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
    lib.cudssSetThreadingLayer.argtypes = [vp, ctypes.c_char_p]
    lib.cudssMatrixSetValues.argtypes = [vp, vp]
    ver = [ctypes.c_int() for _ in range(3)]
    for k, v in enumerate(ver):
        lib.cudssGetProperty(k, ctypes.byref(v))
    nthreads = int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))
    rec.save(cudss_version='.'.join(str(v.value) for v in ver), cupy=cp.__version__, mtype='SPD' if spd else 'GENERAL',
             mview='FULL', host_threads=nthreads, threading_layer='libcudss_mtlayer_gomp.so.1',
             reordering='default', refactor_values='same values, same pattern')

    def chk(st, what):
        if st != 0:
            raise RuntimeError(f'{what}: cudss status {st} ({CUDSS_STATUS[st] if st < len(CUDSS_STATUS) else "?"})')

    dev = cp.cuda.Device(0)
    poll = MemPoll(cp)
    N, nnz = A.shape[0], A.nnz
    indptr, indices, data = cp.asarray(A.indptr), cp.asarray(A.indices), cp.asarray(A.data)
    Ag = csp.csr_matrix((data, indices, indptr), shape=A.shape)
    b = cp.ones(N, dtype=cp.float64)
    x = cp.zeros(N, dtype=cp.float64)
    bn = float(cp.linalg.norm(b))
    float(cp.linalg.norm(b - Ag @ b))  # compile the residual kernels before any timing
    handle, config = vp(), vp()
    chk(lib.cudssCreate(ctypes.byref(handle)), 'create')
    chk(lib.cudssSetThreadingLayer(handle, str(libdir / 'libcudss_mtlayer_gomp.so.1').encode()), 'threading layer')
    chk(lib.cudssConfigCreate(ctypes.byref(config)), 'config')
    nt = ctypes.c_int(nthreads)
    chk(lib.cudssConfigSet(config, CFG_HOST_NTHREADS, ctypes.byref(nt), ctypes.sizeof(nt)), 'host threads')
    mA, mx, mb = vp(), vp(), vp()
    chk(lib.cudssMatrixCreateCsr(ctypes.byref(mA), N, N, nnz, indptr.data.ptr, None, indices.data.ptr, data.data.ptr,
                                 CUDA_R_32I, CUDA_R_32I, CUDA_R_64F, MTYPE_SPD if spd else MTYPE_GENERAL, MVIEW_FULL, 0),
        'matrix csr')
    chk(lib.cudssMatrixCreateDn(ctypes.byref(mx), N, 1, N, x.data.ptr, CUDA_R_64F, 0), 'dense x')
    chk(lib.cudssMatrixCreateDn(ctypes.byref(mb), N, 1, N, b.data.ptr, CUDA_R_64F, 0), 'dense b')
    base_used = poll.used()

    def phase(d, ph, name):
        dev.synchronize()
        t = time.perf_counter()
        st = lib.cudssExecute(handle, ph, config, d, mA, mx, mb)
        dev.synchronize()
        dt = time.perf_counter() - t
        chk(st, name)
        return dt

    def resid():
        return float(cp.linalg.norm(b - Ag @ x)) / bn

    T = {k: [] for k in ('analysis', 'factor', 'solve', 'refactor', 'solve_after_refactor')}
    res, first, reps, k = [], {}, None, 0
    try:
        while reps is None or k < reps + 1:
            d = vp()
            chk(lib.cudssDataCreate(handle, ctypes.byref(d)), 'data create')
            poll.reset()
            stage = 'analysis'
            ta = phase(d, PH_ANALYSIS, stage)
            est = (ctypes.c_int64 * 16)()
            w = ctypes.c_size_t()
            lib.cudssDataGet(handle, d, DATA_MEMORY_ESTIMATES, est, ctypes.sizeof(est), ctypes.byref(w))
            lunnz = ctypes.c_int64()
            lib.cudssDataGet(handle, d, DATA_LU_NNZ, ctypes.byref(lunnz), 8, ctypes.byref(w))
            if k == 0:
                rec.save(first_call={'analysis': ta}, cudss_memory_estimates_bytes=list(est)[:6], factor_nnz=int(lunnz.value))
            stage = 'factor'
            tf = phase(d, PH_FACT, stage)
            used_after_factor = poll.used() - base_used
            stage = 'solve'
            x[:] = 0
            ts = phase(d, PH_SOLVE, stage)
            r1 = resid()
            stage = 'refactor'
            chk(lib.cudssMatrixSetValues(mA, data.data.ptr), 'set values')
            tr = phase(d, PH_REFACT, stage)
            stage = 'solve_after_refactor'
            x[:] = 0
            ts2 = phase(d, PH_SOLVE, stage)
            r2 = resid()
            peak = poll.peak - base_used
            chk(lib.cudssDataDestroy(handle, d), 'data destroy')
            row = dict(analysis=ta, factor=tf, solve=ts, refactor=tr, solve_after_refactor=ts2)
            if k == 0:
                first = row
                reps = nreps(ta + tf + ts + tr + ts2)
                rec.save(first_call=first, first_call_relres=[r1, r2], reps=reps, peak_device_bytes_first_cycle=int(peak),
                         device_bytes_held_after_factor=int(used_after_factor), device_total_bytes=int(poll.total),
                         device_bytes_matrix_and_vectors=int(base_used - poll.base))
            else:
                for kk, v in row.items():
                    T[kk].append(v)
                res.append([r1, r2])
                rec.save(times_s={kk: stats(v) for kk, v in T.items()}, relres=res, peak_device_bytes=int(peak))
            k += 1
        rec.save(status='ok', relres_max=float(np.max(res)))
    except Exception as e:  # noqa: BLE001
        rec.save(status='failed', failed_stage=stage, error=repr(e), device_used_bytes_at_failure=int(poll.used()),
                 peak_device_bytes_at_failure=int(poll.peak), device_total_bytes=int(poll.total))
    poll.stop = True


# ----------------------------------------------------------------------------------------------- AmgX
AMGX_MODE_DDDI = 8193
AMGX_STATUS = ['SUCCESS', 'FAILED', 'DIVERGED', 'NOT_CONVERGED']


def _quiet(node, tol=None):
    for k in list(node):
        if k in ('print_grid_stats', 'print_solve_stats', 'obtain_timings', 'print_vis_data', 'print_config'):
            node[k] = 0
        elif isinstance(node[k], dict):
            _quiet(node[k])


def run_amgx(a):
    import cupy as cp
    A, spd, tb = load_matrix(a)
    rec = Rec(a.out, meta(a, A, spd, tb))
    root = Path(os.environ['SS_AMGX_ROOT'])
    lib = ctypes.CDLL(str(root / 'build' / 'libamgxsh.so'), mode=ctypes.RTLD_GLOBAL)
    vp, ci = ctypes.c_void_p, ctypes.c_int
    P = ctypes.POINTER
    lib.AMGX_config_create_from_file.argtypes = [P(vp), ctypes.c_char_p]
    lib.AMGX_resources_create_simple.argtypes = [P(vp), vp]
    lib.AMGX_matrix_create.argtypes = [P(vp), vp, ci]
    lib.AMGX_vector_create.argtypes = [P(vp), vp, ci]
    lib.AMGX_solver_create.argtypes = [P(vp), vp, ci, vp]
    lib.AMGX_matrix_upload_all.argtypes = [vp, ci, ci, ci, ci, vp, vp, vp, vp]
    lib.AMGX_vector_upload.argtypes = [vp, ci, ci, vp]
    lib.AMGX_vector_set_zero.argtypes = [vp, ci, ci]
    lib.AMGX_vector_download.argtypes = [vp, vp]
    lib.AMGX_solver_setup.argtypes = [vp, vp]
    lib.AMGX_solver_solve_with_0_initial_guess.argtypes = [vp, vp, vp]
    lib.AMGX_solver_get_iterations_number.argtypes = [vp, P(ci)]
    lib.AMGX_solver_get_status.argtypes = [vp, P(ci)]
    for f in ('AMGX_solver_destroy', 'AMGX_matrix_destroy', 'AMGX_vector_destroy', 'AMGX_resources_destroy',
              'AMGX_config_destroy'):
        getattr(lib, f).argtypes = [vp]
    CB = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int)
    log = []
    cb = CB(lambda msg, n: log.append(msg[:n].decode(errors='replace')))

    def chk(rc, what):
        if rc != 0:
            raise RuntimeError(f'{what}: AMGX rc {rc}; log tail: {"".join(log)[-600:]}')

    chk(lib.AMGX_initialize(), 'initialize')
    lib.AMGX_register_print_callback(cb)
    dev = cp.cuda.Device(0)
    poll = MemPoll(cp)
    N, nnz = A.shape[0], A.nnz
    b = np.ones(N)
    xh = np.zeros(N)
    bn = float(np.linalg.norm(b))
    configs = AMGX_SPD if spd else AMGX_GEN
    out = {}
    rec.save(amgx_commit=os.environ.get('SS_AMGX_COMMIT'), cupy=cp.__version__, mode='dDDI', configs=configs,
             config_changes='tolerance set; max_iters 1000; printing and internal timers off; rest as shipped',
             convergence='as shipped (RELATIVE_INI, L2), zero initial guess', results=out)
    tmp = Path(os.environ.get('TMPDIR', '/tmp'))
    for cname in configs:
        for tol in TOLS:
            key = f'{cname}|{tol:g}'
            cur = out[key] = {'config': cname + '.json', 'tol': tol, 'status': 'started'}
            stage = 'config'
            cfg, rs, mA, vb, vx, solver = vp(), vp(), vp(), vp(), vp(), vp()
            try:
                j = json.loads((root / 'src' / 'src' / 'configs' / (cname + '.json')).read_text())
                _quiet(j)
                j['solver'].update(tolerance=tol, max_iters=1000, print_solve_stats=0, obtain_timings=0, print_grid_stats=0,
                                   monitor_residual=1)
                cf = tmp / f'amgx_{os.getpid()}.json'
                cf.write_text(json.dumps(j))
                chk(lib.AMGX_config_create_from_file(ctypes.byref(cfg), str(cf).encode()), 'config')
                chk(lib.AMGX_resources_create_simple(ctypes.byref(rs), cfg), 'resources')
                base_used = poll.used()
                poll.reset()
                chk(lib.AMGX_matrix_create(ctypes.byref(mA), rs, AMGX_MODE_DDDI), 'matrix create')
                chk(lib.AMGX_vector_create(ctypes.byref(vb), rs, AMGX_MODE_DDDI), 'vector create')
                chk(lib.AMGX_vector_create(ctypes.byref(vx), rs, AMGX_MODE_DDDI), 'vector create')
                stage = 'upload'
                t = time.perf_counter()
                chk(lib.AMGX_matrix_upload_all(mA, N, nnz, 1, 1, A.indptr.ctypes.data, A.indices.ctypes.data,
                                               A.data.ctypes.data, None), 'matrix upload')
                chk(lib.AMGX_vector_upload(vb, N, 1, b.ctypes.data), 'rhs upload')
                chk(lib.AMGX_vector_set_zero(vx, N, 1), 'x zero')
                dev.synchronize()
                cur['upload_s'] = time.perf_counter() - t
                Ts, Tv, its, sts, rr, reps, k = [], [], [], [], [], None, 0
                while reps is None or k < reps + 1:
                    stage = 'setup'
                    chk(lib.AMGX_solver_create(ctypes.byref(solver), rs, AMGX_MODE_DDDI, cfg), 'solver create')
                    dev.synchronize()
                    t = time.perf_counter()
                    rc = lib.AMGX_solver_setup(solver, mA)
                    dev.synchronize()
                    tset = time.perf_counter() - t
                    chk(rc, 'setup')
                    stage = 'solve'
                    t = time.perf_counter()
                    rc = lib.AMGX_solver_solve_with_0_initial_guess(solver, vb, vx)
                    dev.synchronize()
                    tsol = time.perf_counter() - t
                    chk(rc, 'solve')
                    it, st = ci(), ci()
                    lib.AMGX_solver_get_iterations_number(solver, ctypes.byref(it))
                    lib.AMGX_solver_get_status(solver, ctypes.byref(st))
                    chk(lib.AMGX_vector_download(vx, xh.ctypes.data), 'download')
                    r = float(np.linalg.norm(b - A @ xh)) / bn
                    if k == 0:
                        cur.update(first_call={'setup': tset, 'solve': tsol}, peak_device_bytes=int(poll.peak - base_used))
                        reps = nreps(tset + tsol)
                        cur['reps'] = reps
                    else:
                        Ts.append(tset); Tv.append(tsol)
                    its.append(it.value); sts.append(AMGX_STATUS[st.value] if 0 <= st.value < 4 else str(st.value)); rr.append(r)
                    chk(lib.AMGX_solver_destroy(solver), 'solver destroy')
                    solver = vp()
                    cur.update(setup_s=stats(Ts), solve_s=stats(Tv), iterations=its, solve_status=sts, relres=rr)
                    rec.save()
                    k += 1
                cur.update(status='ok', relres_max=max(rr), reached_tol=bool(max(rr) <= tol * 1.01))
            except Exception as e:  # noqa: BLE001
                cur.update(status='failed', failed_stage=stage, error=repr(e), device_used_bytes_at_failure=int(poll.used()),
                           device_total_bytes=int(poll.total))
            for h, f in ((solver, 'AMGX_solver_destroy'), (vx, 'AMGX_vector_destroy'), (vb, 'AMGX_vector_destroy'),
                         (mA, 'AMGX_matrix_destroy'), (rs, 'AMGX_resources_destroy'), (cfg, 'AMGX_config_destroy')):
                if h.value:
                    getattr(lib, f)(h)
            rec.save()
    rec.save(status='ok' if any(v['status'] == 'ok' for v in out.values()) else 'failed', amgx_log_tail=''.join(log)[-1500:])
    poll.stop = True


# ----------------------------------------------------------------------------------------------- CG
def run_cg(a):
    import cupy as cp
    import cupyx.scipy.sparse as csp
    A, spd, tb = load_matrix(a)
    rec = Rec(a.out, meta(a, A, spd, tb))
    assert spd
    dev = cp.cuda.Device(0)
    Ag = csp.csr_matrix(A)
    N = A.shape[0]
    b = cp.ones(N, dtype=cp.float64)
    dinv = 1.0 / cp.asarray(A.diagonal())
    bn = float(cp.linalg.norm(b))
    CHECK, MAXIT = 10, 400000
    rec.save(cupy=cp.__version__, method='CG, Jacobi preconditioner, zero initial guess, recurrence residual checked every '
             f'{CHECK} iterations, max {MAXIT} iterations', results={})

    def pcg(tol, maxit):
        x = cp.zeros_like(b)
        r = b.copy()
        z = dinv * r
        p = z.copy()
        rz = cp.dot(r, z)
        it = 0
        while it < maxit:
            Ap = Ag @ p
            alpha = rz / cp.dot(p, Ap)
            x += alpha * p
            r -= alpha * Ap
            z = dinv * r
            rz_new = cp.dot(r, z)
            p = z + (rz_new / rz) * p
            rz = rz_new
            it += 1
            if it % CHECK == 0 and float(cp.linalg.norm(r)) <= tol * bn:
                break
        return x, it

    pcg(1e-30, 20)  # compile kernels
    dev.synchronize()
    out = rec.d['results']
    for tol in TOLS:
        cur = out[f'{tol:g}'] = {'tol': tol}
        Ts, its, rr, reps, k = [], [], [], None, 0
        while reps is None or k < reps + 1:
            dev.synchronize()
            t = time.perf_counter()
            x, it = pcg(tol, MAXIT)
            dev.synchronize()
            dt = time.perf_counter() - t
            r = float(cp.linalg.norm(b - Ag @ x)) / bn
            if k == 0:
                reps = nreps(dt)
                cur.update(first_call=dt, reps=reps)
            else:
                Ts.append(dt)
            its.append(it); rr.append(r)
            cur.update(solve_s=stats(Ts), iterations=its, relres=rr, reached_tol=bool(max(rr) <= tol * 1.01))
            rec.save()
            k += 1
    rec.save(status='ok')


# ----------------------------------------------------------------------------------------------- driver
def driver(a):
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for line in Path(a.tasks).read_text().split('\n'):
        if not line.strip() or line.startswith('#'):
            continue
        solver, problem, dim, n = line.split()
        out = outdir / f'{problem}{dim}d_n{n}_{solver}.json'
        if out.exists() and json.loads(out.read_text()).get('status') in ('ok', 'failed', 'crashed', 'timeout'):
            continue
        env = dict(os.environ)
        if solver == 'amgx':
            env['LD_LIBRARY_PATH'] = env.get('SS_AMGX_LD', '') + ':' + env.get('LD_LIBRARY_PATH', '')
        t = time.time()
        cmd = [sys.executable, __file__, 'worker', '--solver', solver, '--problem', problem, '--dim', dim, '--n', n, '--out',
               str(out)]
        try:
            p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=a.timeout)
            rc, err = p.returncode, (p.stdout[-1500:] + '\n' + p.stderr[-3000:])
        except subprocess.TimeoutExpired as e:
            rc, err = 'timeout', str(e)[-500:]
        d = json.loads(out.read_text()) if out.exists() else dict(solver=solver, problem=problem, dim=int(dim), n=int(n))
        if d.get('status') not in ('ok', 'failed'):
            d.update(status='timeout' if rc == 'timeout' else 'crashed', returncode=rc, stderr_tail=err,
                     note='worker process ended before finishing; stages already saved are kept')
            out.write_text(json.dumps(d, indent=1))
        print(f'{time.strftime("%T")} {solver} {problem} {dim}d n={n}: {d.get("status")} rc={rc} wall={time.time() - t:.0f}s',
              flush=True)
        if d.get('status') != 'ok':
            print(str(d.get('error', '')) + ' ' + str(err)[-1200:], flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    w = sub.add_parser('worker')
    w.add_argument('--solver', required=True); w.add_argument('--problem', required=True)
    w.add_argument('--dim', type=int, required=True); w.add_argument('--n', type=int, required=True)
    w.add_argument('--out', required=True)
    d = sub.add_parser('driver')
    d.add_argument('--tasks', required=True); d.add_argument('--outdir', required=True)
    d.add_argument('--timeout', type=int, default=5400)
    a = ap.parse_args()
    if a.cmd == 'driver':
        driver(a)
    else:
        {'cudss': run_cudss, 'amgx': run_amgx, 'cg': run_cg}[a.solver](a)
