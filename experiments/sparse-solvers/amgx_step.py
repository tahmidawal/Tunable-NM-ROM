"""Measured implicit Burgers time-stepper on AmgX (PROTOCOL.md amendment 3, Task A item 4; amendment 4 fixes the settings).

Same discretisation, kernels, Newton loop, timing contract and error metric as stepper.py (the Newton loop is literally
stepper.Mesh.advance); only the linear solver differs: the Jacobian is handed to AmgX (device pointers) and solved
inexactly to a relative linear tolerance `ltol` (zero initial guess).  "Refactorisation" here = replace the matrix
coefficients + AmgX solver setup.  Variants: full (setup every Newton iteration), k1 (setup once per time step), frozen
(setup once per trajectory), lin1 (one setup + one solve per step, no residual test).

  python amgx_step.py --dim 3 --L 64 --cases 16 --refs refs --out results/amgx_b3d_L64.json --budget 1200
"""
import argparse
import ctypes
import json
import os
import time
from pathlib import Path

import numpy as np

from bench import MemPoll, Rec, stats
import matrices
from stepper import KERNELS, Mesh, Runner

AMGX_MODE_DDDI = 8193
JACOBI_KRYLOV = {'config_version': 2, 'solver': {
    'scope': 'main', 'solver': 'PBICGSTAB', 'max_iters': 200, 'monitor_residual': 1, 'convergence': 'RELATIVE_INI',
    'norm': 'L2', 'tolerance': 1e-6,
    'preconditioner': {'scope': 'jacobi', 'solver': 'BLOCK_JACOBI', 'max_iters': 1, 'monitor_residual': 0}}}
CONFIGS = {2: ['PBICGSTAB_CLASSICAL_JACOBI', 'PBICGSTAB_JACOBI_NOAMG'],
           3: ['FGMRES_CLASSICAL_AGGRESSIVE_PMIS', 'PBICGSTAB_JACOBI_NOAMG']}


def _quiet(node):
    for k in list(node):
        if k in ('print_grid_stats', 'print_solve_stats', 'obtain_timings', 'print_vis_data', 'print_config', 'store_res_history'):
            node[k] = 0
        elif isinstance(node[k], dict):
            _quiet(node[k])


class Amgx:
    def __init__(self):
        self.root = Path(os.environ['SS_AMGX_ROOT'])
        lib = ctypes.CDLL(str(self.root / 'build' / 'libamgxsh.so'), mode=ctypes.RTLD_GLOBAL)
        vp, ci = ctypes.c_void_p, ctypes.c_int
        P = ctypes.POINTER
        lib.AMGX_config_create_from_file.argtypes = [P(vp), ctypes.c_char_p]
        lib.AMGX_resources_create_simple.argtypes = [P(vp), vp]
        lib.AMGX_matrix_create.argtypes = [P(vp), vp, ci]
        lib.AMGX_vector_create.argtypes = [P(vp), vp, ci]
        lib.AMGX_solver_create.argtypes = [P(vp), vp, ci, vp]
        lib.AMGX_matrix_upload_all.argtypes = [vp, ci, ci, ci, ci, vp, vp, vp, vp]
        lib.AMGX_matrix_replace_coefficients.argtypes = [vp, ci, ci, vp, vp]
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
        self.CB = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int)
        self.log = []
        self.cb = self.CB(lambda msg, n: self.log.append(msg[:n].decode(errors='replace')))
        self.lib, self.vp, self.ci = lib, vp, ci
        self.chk(lib.AMGX_initialize(), 'initialize')
        lib.AMGX_register_print_callback(self.cb)
        self.tmp = Path(os.environ.get('TMPDIR', '/tmp'))

    def chk(self, rc, what):
        if rc != 0:
            raise RuntimeError(f'{what}: AMGX rc {rc}; log tail: {"".join(self.log)[-600:]}')

    def config(self, name, ltol):
        if name == 'PBICGSTAB_JACOBI_NOAMG':
            j = json.loads(json.dumps(JACOBI_KRYLOV))
        else:
            j = json.loads((self.root / 'src' / 'src' / 'configs' / (name + '.json')).read_text())
        _quiet(j)
        j['solver'].update(tolerance=ltol, max_iters=200, monitor_residual=1)
        cf = self.tmp / f'amgx_step_{os.getpid()}.json'
        cf.write_text(json.dumps(j))
        cfg = self.vp()
        self.chk(self.lib.AMGX_config_create_from_file(ctypes.byref(cfg), str(cf).encode()), 'config')
        return cfg, j


class AmgxMesh:
    """Same state and kernels as stepper.Mesh; refactor()/solve() go to AmgX.  One (config, ltol) per object."""
    neg_residual = Mesh.neg_residual
    assemble = Mesh.assemble
    advance = Mesh.advance

    def __init__(self, cp, L, dim, ax, poll, cfgname, ltol):
        self.cp, self.L, self.dim, self.ax = cp, L, dim, ax
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
        del A
        self.u = cp.zeros(N); self.prev = cp.zeros(N); self.b = cp.zeros(N); self.x = cp.zeros(N)
        mod = cp.RawModule(code=KERNELS)
        self.k_res, self.k_jac = mod.get_function('neg_residual'), mod.get_function('jacobian')
        self.grid = ((N + 255) // 256,)
        self.assemble(self.u, 0.01, 0.005)
        self.neg_residual(0.01, 0.005)
        float(cp.linalg.norm(self.b)); self.u += self.x; cp.copyto(self.u, self.prev)
        dev.synchronize()
        l, vp = ax.lib, ax.vp
        self.cfg, self.cfg_json = ax.config(cfgname, ltol)
        self.rs, self.mA, self.vb, self.vx, self.solver = vp(), vp(), vp(), vp(), vp()
        ax.chk(l.AMGX_resources_create_simple(ctypes.byref(self.rs), self.cfg), 'resources')
        base = poll.used(); poll.reset()
        ax.chk(l.AMGX_matrix_create(ctypes.byref(self.mA), self.rs, AMGX_MODE_DDDI), 'matrix create')
        ax.chk(l.AMGX_vector_create(ctypes.byref(self.vb), self.rs, AMGX_MODE_DDDI), 'vector create')
        ax.chk(l.AMGX_vector_create(ctypes.byref(self.vx), self.rs, AMGX_MODE_DDDI), 'vector create')
        ax.chk(l.AMGX_solver_create(ctypes.byref(self.solver), self.rs, AMGX_MODE_DDDI, self.cfg), 'solver create')
        dev.synchronize(); t = time.perf_counter()
        ax.chk(l.AMGX_matrix_upload_all(self.mA, N, self.nnz, 1, 1, self.indptr.data.ptr, self.indices.data.ptr,
                                        self.data.data.ptr, None), 'matrix upload')
        dev.synchronize(); setup['matrix_upload_s'] = time.perf_counter() - t
        t = time.perf_counter()
        ax.chk(l.AMGX_solver_setup(self.solver, self.mA), 'first setup')
        dev.synchronize(); setup['first_setup_s'] = time.perf_counter() - t
        setup.update(peak_device_bytes_setup=int(poll.peak - base), device_bytes_held_after_setup=int(poll.used() - base))
        self.setup = setup
        self.counts = dict(refactor=0, solve=0, lin_its=0, lin_not_converged=0)
        self.tim = None
        self.pending = True
        self._it, self._st = ax.ci(), ax.ci()

    def close(self):
        l = self.ax.lib
        for h, f in ((self.solver, 'AMGX_solver_destroy'), (self.vx, 'AMGX_vector_destroy'), (self.vb, 'AMGX_vector_destroy'),
                     (self.mA, 'AMGX_matrix_destroy'), (self.rs, 'AMGX_resources_destroy'), (self.cfg, 'AMGX_config_destroy')):
            if h.value:
                getattr(l, f)(h)

    def refactor(self, nu, dt):
        if self.tim is not None:
            self.dev.synchronize(); t = time.perf_counter()
        self.assemble(self.u, nu, dt)
        l = self.ax.lib
        rc = l.AMGX_matrix_replace_coefficients(self.mA, self.N, self.nnz, self.data.data.ptr, None)
        rc = rc or l.AMGX_solver_setup(self.solver, self.mA)
        if rc:
            self.ax.chk(rc, 'replace coefficients + setup')
        self.counts['refactor'] += 1
        if self.tim is not None:
            self.dev.synchronize(); self.tim['refactor'] += time.perf_counter() - t

    def solve(self):
        if self.tim is not None:
            self.dev.synchronize(); t = time.perf_counter()
        l = self.ax.lib
        rc = l.AMGX_vector_upload(self.vb, self.N, 1, self.b.data.ptr)
        rc = rc or l.AMGX_vector_set_zero(self.vx, self.N, 1)
        rc = rc or l.AMGX_solver_solve_with_0_initial_guess(self.solver, self.vb, self.vx)
        rc = rc or l.AMGX_vector_download(self.vx, self.x.data.ptr)
        if rc:
            self.ax.chk(rc, 'solve')
        l.AMGX_solver_get_iterations_number(self.solver, ctypes.byref(self._it))
        l.AMGX_solver_get_status(self.solver, ctypes.byref(self._st))
        self.counts['solve'] += 1
        self.counts['lin_its'] += self._it.value
        self.counts['lin_not_converged'] += int(self._st.value != 0)
        if self.tim is not None:
            self.dev.synchronize(); self.tim['solve'] += time.perf_counter() - t


class ARunner(Runner):
    """Runner.trajectory unchanged; mesh(f) returns the current AmgX mesh (same mesh only, f = 1)."""

    def __init__(self, cp, dim, L, ax, poll):
        super().__init__(cp, dim, L, None, poll)
        self.ax, self.cur, self.key = ax, None, None

    def use(self, cfgname, ltol):
        if self.key != (cfgname, ltol):
            if self.cur is not None:
                self.cur.close()
                self.cur = None
                self.cp.get_default_memory_pool().free_all_blocks()
            self.cur = AmgxMesh(self.cp, self.L, self.dim, self.ax, self.poll, cfgname, ltol)
            self.key = (cfgname, ltol)
        return self.cur

    def mesh(self, f):
        assert f == 1
        return self.cur

    def run(self, u0, nu, variant, dt, ntol):
        M = self.cur
        dim = self.dim
        sub = int(round(0.05 / dt))
        rec = dict(newton=[], unconverged=0, max_rel_res=0.0)
        M.counts = dict(refactor=0, solve=0, lin_its=0, lin_not_converged=0)
        M.pending = True
        self.dev.synchronize()
        t = time.perf_counter()
        M.u[...] = u0[(slice(1, -1),) * dim].ravel()
        self.write(0, M, 1)
        for k in range(5):
            M.advance(variant, nu, dt, ntol, sub, k * sub, rec)
            self.write(k + 1, M, 1)
        self.dev.synchronize()
        sec = time.perf_counter() - t
        rec.update(M.counts)
        return sec, rec


def settings_list(dim):
    out = []
    for cfg in CONFIGS[dim]:
        out.append(dict(config=cfg, variant='lin1', dt=0.005, ntol=None, ltol=1e-3, group='same-mesh'))
        for nt, lt in ((1e-6, 1e-2), (3e-3, 1e-1)):
            for v in ('k1', 'frozen', 'full'):
                out.append(dict(config=cfg, variant=v, dt=0.005, ntol=nt, ltol=lt, group='same-mesh'))
    grid = []
    for cfg in CONFIGS[dim]:
        for dt in (0.01, 0.005):
            for v in ('frozen', 'k1'):
                for nt in (1e-2, 1e-3):
                    grid.append(dict(config=cfg, variant=v, dt=dt, ntol=nt, ltol=1e-1, group='matched-grid'))
        grid.append(dict(config=cfg, variant='lin1', dt=0.01, ntol=None, ltol=1e-2, group='matched-grid'))
    return out, grid


def name_of(s):
    nt = 'none' if s['ntol'] is None else f"{s['ntol']:g}"
    return f"{s['config']}|{s['variant']}_dt{s['dt']:g}_ntol{nt}_ltol{s['ltol']:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dim', type=int, required=True)
    ap.add_argument('--L', type=int, required=True)
    ap.add_argument('--cases', type=int, required=True)
    ap.add_argument('--refs', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--budget', type=float, default=1800.0, help='wall seconds for everything after the checks')
    a = ap.parse_args()
    import cupy as cp
    t_start = time.time()
    dim, L, K = a.dim, a.L, a.cases
    refs = Path(a.refs)
    meta = json.loads((refs / f'b{dim}d_L{L}.json').read_text())
    K = min(K, len(meta['case_records']))
    rec = Rec(a.out, dict(task='burgers_stepper', dim=dim, L=L, unknowns=(L - 1) ** dim, cases=K, dtype='float64', solver='amgx',
                          reference={k: v for k, v in meta.items() if k != 'case_records'}, cupy=cp.__version__,
                          budget_s=a.budget, amgx_commit=os.environ.get('SS_AMGX_COMMIT'), mode='dDDI',
                          config_changes='linear tolerance set; max_iters 200; printing and internal timers off; rest as shipped. '
                                         'PBICGSTAB_JACOBI_NOAMG is written by this lane: BiCGStab + one block-Jacobi sweep, no AMG',
                          interface='device pointers passed to the AmgX C API (matrix values, right-hand side, solution); AmgX copies them'))
    poll = MemPoll(cp)
    ax = Amgx()
    R = ARunner(cp, dim, L, ax, poll)
    nus = [r['nu'] for r in meta['case_records']][:K]
    ref = [cp.asarray(np.load(refs / f'b{dim}d_L{L}_case{c}.npy')) for c in range(K)]
    shape = (L + 1,) * dim
    u0s = [r[0].reshape(shape).copy() for r in ref]
    n0 = [float(cp.linalg.norm(r[0])) for r in ref]

    def error(c):
        o = R.out.reshape(6, -1)
        return max(float(cp.linalg.norm(o[k] - ref[c][k])) / n0[c] for k in range(1, 6))

    results, setups, checks = {}, {}, {}
    rec.save(settings=results, setup=setups, checks=checks)
    same, grid = settings_list(dim)
    stage = 'setup'
    try:
        for s in same + grid:
            nm = stage = name_of(s)
            if time.time() - t_start > a.budget:
                results[nm] = dict(s, name=nm, status='not run (budget)')
                continue
            cfg, v, dt, lt = s['config'], s['variant'], s['dt'], s['ltol']
            nt = 0.0 if s['ntol'] is None else s['ntol']
            try:
                M = R.use(cfg, lt)
                key = f'{cfg}|{lt:g}'
                if key not in setups:
                    setups[key] = M.setup
                if cfg not in checks:        # tight full Newton with this solver vs the paper's reference, before timing
                    _, r0 = R.run(u0s[0], nus[0], 'full', 0.005, 1e-8)
                    e_tight = error(0)
                    e_ic = float(cp.linalg.norm(R.out.reshape(6, -1)[0] - ref[0][0])) / n0[0]
                    checks[cfg] = dict(ltol=lt, tight_full_newton_vs_reference=e_tight, initial_field_difference=e_ic,
                                       tight_unconverged_steps=r0['unconverged'], tight_newton_total=int(sum(r0['newton'])),
                                       linear_not_converged=r0['lin_not_converged'],
                                       passed=bool(e_tight < 2e-4 and e_ic == 0.0 and r0['unconverged'] == 0))
                    rec.save()
                    print('checks', cfg, checks[cfg], flush=True)
                if not checks[cfg]['passed']:
                    results[nm] = dict(s, name=nm, status='not run (pre-timing check failed for this config)')
                    continue
                R.run(u0s[0], nus[0], v, dt, nt)                      # warm-up, untimed
                times, rows, reps = [], [], None
                for c in range(K):
                    k, row = 0, None
                    while reps is None or k < reps:
                        sec, r = R.run(u0s[c], nus[c], v, dt, nt)
                        if reps is None:
                            reps = 3 if sec <= 5.0 else 1
                        times.append(sec)
                        if row is None:
                            row = dict(case=c, error=error(c), newton_total=int(sum(r['newton'])),
                                       newton_max_step=int(max(r['newton'])), refactor=r['refactor'], solve=r['solve'],
                                       linear_iterations=r['lin_its'], linear_not_converged=r['lin_not_converged'],
                                       unconverged_steps=r['unconverged'], max_rel_residual=r['max_rel_res'], times=[])
                        row['times'].append(sec)
                        k += 1
                    rows.append(row)
                errs = [r['error'] for r in rows]
                res = dict(s, name=nm, coarsen=1, steps=int(round(0.25 / dt)), mesh_L=L, reps_per_case=reps, time_s=stats(times),
                           worst_error=float(np.max(errs)) if np.isfinite(errs).all() else float('inf'),
                           median_error=float(np.median(errs)),
                           newton_total_median=float(np.median([r['newton_total'] for r in rows])),
                           newton_total_max=int(max(r['newton_total'] for r in rows)),
                           refactor_median=float(np.median([r['refactor'] for r in rows])),
                           solve_median=float(np.median([r['solve'] for r in rows])),
                           linear_iterations_median=float(np.median([r['linear_iterations'] for r in rows])),
                           linear_not_converged_total=int(sum(r['linear_not_converged'] for r in rows)),
                           unconverged_steps_total=int(sum(r['unconverged_steps'] for r in rows)),
                           cases_with_unconverged=int(sum(r['unconverged_steps'] > 0 for r in rows)),
                           finite=bool(np.isfinite(errs).all()), status='ok', per_case=rows)
                if s['group'] == 'same-mesh':
                    M.tim = dict(refactor=0.0, solve=0.0)
                    sec, r = R.run(u0s[0], nus[0], v, dt, nt)
                    res['instrumented_case0'] = dict(total_s=sec, refactor_s=M.tim['refactor'], solve_s=M.tim['solve'],
                                                     rest_s=sec - M.tim['refactor'] - M.tim['solve'],
                                                     note='synchronised around every setup and solve; slower than the timed runs')
                    M.tim = None
                results[nm] = res
                ts = res['time_s']
                print(f"{time.strftime('%T')} {nm}: {1e3 * ts['median']:.2f} ms [{1e3 * ts['min']:.2f}, {1e3 * ts['max']:.2f}] "
                      f"worst {100 * res['worst_error']:.4f}% newton {res['newton_total_median']:.0f} "
                      f"lin {res['linear_iterations_median']:.0f} setups {res['refactor_median']:.0f} "
                      f"unconv {res['unconverged_steps_total']}", flush=True)
            except Exception as e:  # noqa: BLE001  one setting failing (e.g. AmgX setup error) must not lose the others
                results[nm] = dict(s, name=nm, status='failed', error=repr(e)[:800])
                print('SETTING FAILED', nm, repr(e)[:300], flush=True)
                if R.cur is not None and getattr(R.cur, 'tim', None) is not None:
                    R.cur.tim = None
            rec.save(peak_device_bytes=int(poll.peak), elapsed_s=time.time() - t_start)
        rec.save(status='ok' if any(v.get('status') == 'ok' for v in results.values()) else 'failed',
                 peak_device_bytes=int(poll.peak), elapsed_s=time.time() - t_start, amgx_log_tail=''.join(ax.log)[-1500:])
    except Exception as e:  # noqa: BLE001
        rec.save(status='failed', failed_stage=stage, error=repr(e), device_used_bytes_at_failure=int(poll.used()),
                 peak_device_bytes_at_failure=int(poll.peak), device_total_bytes=int(poll.total))
        print('FAILED', stage, repr(e), flush=True)
    poll.stop = True


if __name__ == '__main__':
    main()
