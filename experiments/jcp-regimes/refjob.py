"""jcp-references cluster job (DESIGN §3–§9, A1–A5).

    python refjob.py --config <cfg.json> --out <dir>

modes
  gates  all pre-run code gates (gates.run_all('all')), incl. G3b/G3c at the escalation PS sizes
  calib  floor-proxy calibration (prod / tight / tighter, every base level, both kinds) on the calibration dev cases,
         the D1 time-only / space-only ladders, X1 (upwind+BE ladder on every dev case) and X4 (doubled middle dt);
         every field and per-step evidence array saved; writes calib_status (proxy_ok) for the production refusal
  fd     R_ST / R_S references of one cohort (DESIGN §4 procedure, A1.5/A2.1/A3.2/A5): per case and kind, three levels
         -> judge; floor failure -> tight attempt at all three levels; order or Peclet failure -> refinement level,
         judged on the finest three (tight again if that triple fails the floor); solver failure -> one retry, then
         terminal solver-fail. Writes one npz + json per (case, kind); AES-encrypted and silent for sealed cohorts.
         Refuses to run unless the committed calibration status (checks/calib_<d>d.json) passed with this FLOOR_STAB.
  ps     pseudo-spectral cross-check runs for given cases (spatial triple at dt, half step at N3, one stability rerun;
         per-run IC-quadrature and contour checks); `attempt` > 0 for an escalation ladder (A1.7).
Every mode first runs the unit gates and G1 (cheap) and refuses to continue if any fails.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import cohorts as CH
import rcohorts as RG          # jcp-regimes edit E2: regime cohorts (DESIGN 3)
import gates as G
import pscore as PS
import refcore as RC
import sealio as SI
import verdict as V

HERE = Path(__file__).resolve().parent
LEVELS = {  # kind -> dim -> dict(base [(L, dt)], refine (L, dt), scheme)
    'ST': {2: dict(base=[(2048, 1.25e-3), (4096, 6.25e-4), (8192, 3.125e-4)], refine=(12288, 0.05 / 240), scheme='cnr'),
           3: dict(base=[(128, 2.5e-3), (256, 1.25e-3), (512, 6.25e-4)], refine=(768, 0.05 / 120), scheme='cnr')},
    'S': {2: dict(base=[(2048, .005), (4096, .005), (8192, .005)], refine=(12288, .005), scheme='be'),
          3: dict(base=[(128, .005), (256, .005), (512, .005)], refine=(768, .005), scheme='be')},
}
TOL = {'prod': (1e-11, 1e-9, 20), 'retry': (1e-11, 1e-10, 40), 'tight': (1e-13, 1e-12, 40),
       'tighter': (3e-14, 1e-12, 8)}         # A9.1: budget 8 (round-off floor; 60 only burned time)
PS_CFG = {2: dict(Ns=[1024, 1536, 2048], dt=0.05 / 1600, esc=[1536, 2048, 3072]),      # A8.1
          3: dict(Ns=[128, 192, 256], dt=0.05 / 320, esc=[192, 256, 384])}
CALIB_CASES = {2: ('dev6', [0, 4]), 3: ('dev4', [0, 2])}
X1_LEVELS = {2: [(512, 2.5e-3), (1024, 1.25e-3), (2048, 6.25e-4)], 3: [(64, 5e-3), (128, 2.5e-3), (256, 1.25e-3)]}

_SOLVERS = {}
_PS = {}
T_BEGIN = time.perf_counter()
PREV_MANIFEST = {}                  # hashes this job directory recorded in an earlier (interrupted) run
QUIET = [False]                     # sealed jobs: no per-case / per-attempt telemetry on stdout (A5.5)


def log(s, sealed_ok=False):
    if QUIET[0] and not sealed_ok:
        return
    print(f'[{time.perf_counter() - T_BEGIN:9.1f}s] {s}', flush=True)


def solver(d, L, adv, max_newton):
    key = (d, L, adv, max_newton)
    if key not in _SOLVERS:
        _SOLVERS[key] = RC.make_solver(d, L, adv=adv, max_newton=max_newton)
    return _SOLVERS[key]


def attempt(coh, j, L, dt, scheme, purpose, adv='central'):
    """One level run -> (record, lattice fields, per-stage residuals, per-stage Newton counts)."""
    d = coh.dim
    ntol, ltol, mn = TOL['prod' if purpose in ('prod', 'refine') else purpose]
    t = time.perf_counter()
    u0 = coh.ic(L, j)
    t_ic = time.perf_counter() - t
    run = solver(d, L, adv, mn)
    r = run(u0, coh.nu(j), dt, scheme, ntol, ltol)
    out = jax.block_until_ready(r['out'])
    t_solve = time.perf_counter() - t - t_ic
    lat = RC.to_lattice(out, L, d)
    grad = stiffness_stats(out, L, d)       # jcp-regimes edit E3: measured stiffness index inputs (DESIGN 4.4)
    ns = int(r['stages'])
    umax_running = float(r['umax'])
    res = np.asarray(r['res'])
    nwt = np.asarray(r['nwt'])
    del out, r, u0
    nexp = RC.n_stages(dt, scheme)
    rec = dict(purpose=purpose, L=L, dt=dt, scheme=scheme, adv=adv, ntol=ntol, ltol=ltol, max_newton=mn, stages=ns,
               expected_stages=nexp, steps_ok=bool(V.steps_ok(res, ns) and ns == nexp),
               max_res=float(np.nanmax(res[:ns])) if ns else None, tau=V.floor_proxy(res, ns),
               newton_total=int(nwt[:ns].sum()), newton_max=int(nwt[:ns].max()), umax_running=umax_running,
               lattice_finite=bool(np.isfinite(lat).all()), seconds=time.perf_counter() - t, ic_seconds=t_ic,
               compile_seconds=run.compile_seconds, solve_seconds=t_solve - run.compile_seconds,
               peak_bytes=RC.peak_bytes(), grad_max=grad)
    rec['steps_ok'] = rec['steps_ok'] and rec['lattice_finite']
    return rec, lat, res[:ns], nwt[:ns]


def stiffness_stats(out, L, d, lo=0.05, hi=0.95):
    """jcp-regimes edit E3 (DESIGN 4.4): per output time, max over the interior box [lo, hi]^d of the Euclidean norm of
    the central-difference gradient of the full-grid field (flat interior, x slowest; zero walls padded). 2D only;
    3D returns None (phase 2)."""
    if d != 2:
        return None
    g = []
    for k in range(6):
        u = jnp.pad(jnp.reshape(out[k], (L - 1, L - 1)), 1)
        gx = (u[2:, 1:-1] - u[:-2, 1:-1]) * (L / 2.)
        gy = (u[1:-1, 2:] - u[1:-1, :-2]) * (L / 2.)
        x = jnp.arange(1, L) / L
        box = ((x >= lo) & (x <= hi))
        m = box[:, None] & box[None, :]
        g.append(float(jnp.max(jnp.where(m, jnp.sqrt(gx * gx + gy * gy), 0.))))
    return g


def solve_case(coh, j, kind):
    """DESIGN §4 / A1.5 / A2.1 / A3.2 / A5.1 procedure for one (case, kind). Returns (arrays, meta)."""
    d = coh.dim
    cfg = LEVELS[kind][d]
    t0 = RC.ic_lattice(coh, j)
    attempts, arrays = [], {}

    def run(L, dt, purpose):
        k = len(attempts)
        rec, lat, res, nwt = attempt(coh, j, L, dt, cfg['scheme'], purpose)
        rec['id'] = f'a{k}_L{L}'
        attempts.append(rec)
        arrays[rec['id']] = lat
        arrays['res_' + rec['id']] = res
        arrays['nwt_' + rec['id']] = nwt
        log(f"  {coh.name} c{j} {kind} {rec['id']} {purpose} L={L} dt={dt:.3e} stages={rec['stages']} "
            f"ok={rec['steps_ok']} maxres={rec['max_res']:.2e} newton={rec['newton_total']} "
            f"compile={rec['compile_seconds']:.1f}s solve={rec['solve_seconds']:.1f}s peak={rec['peak_bytes']}")
        return rec, lat

    def get(L, dt, purpose):
        rec, lat = run(L, dt, purpose)
        if not rec['steps_ok'] and purpose in ('prod', 'refine'):
            rec, lat = run(L, dt, 'retry')
        return dict(L=L, lat=lat, tau=rec['tau'], steps_ok=rec['steps_ok'], umax=rec['umax_running']), rec['id']

    dts = dict(cfg['base'])
    dts[cfg['refine'][0]] = cfg['refine'][1]
    history = []

    def judge(triple):
        jd = V.judge([x[0] for x in triple], t0, coh.nu(j))
        history.append(dict(levels=[x[1] for x in triple], outcome=jd['outcome'], checks=jd['checks']))
        return jd

    triple = [get(L, dt, 'prod') for L, dt in cfg['base']]
    jd = judge(triple)
    tight_used = refined = False
    ok = lambda c: c['solver'] and c['t0_shared']
    if ok(jd['checks']) and not jd['checks']['floor']:
        triple = [get(x[0]['L'], dts[x[0]['L']], 'tight') for x in triple]
        tight_used = True
        jd = judge(triple)
    if ok(jd['checks']) and jd['checks']['floor'] and not (jd['checks']['order'] and jd['checks']['peclet']):
        L, dt = cfg['refine']
        triple = triple[1:] + [get(L, dt, 'tight' if tight_used else 'refine')]
        refined = True
        jd = judge(triple)
        if ok(jd['checks']) and not jd['checks']['floor'] and not tight_used:
            triple = [get(x[0]['L'], dts[x[0]['L']], 'tight') for x in triple]
            tight_used = True
            jd = judge(triple)
    arrays['t0'] = t0
    meta = dict(dim=d, cohort=coh.name, case=j, kind=kind, desc=coh.desc(j), cohort_sha256=coh.sha256,
                metric_id=V.METRIC_ID[d], lattice=RC.LATTICE[d], times=RC.TIMES.tolist(),
                t0_sha256=SI.sha(np.ascontiguousarray(t0).tobytes()), scheme=cfg['scheme'], attempts=attempts,
                final=[x[1] for x in triple], history=history, judge=jd, outcome=jd['outcome'], refined=refined,
                tight_used=tight_used, floor_stab=V.FLOOR_STAB)
    return arrays, meta


def job_info():
    return dict(job_id=os.environ.get('SLURM_JOB_ID'), commit=os.environ.get('SOURCE_COMMIT'),
                gpu=jax.devices()[0].device_kind, backend=jax.default_backend(), jax=jax.__version__,
                numpy=np.__version__, precision=os.environ.get('JAX_DEFAULT_MATMUL_PRECISION'))


def V_clean(x):
    if isinstance(x, dict):
        return {str(k): V_clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [V_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return V_clean(x.tolist())
    return x


def atomic_write(path, data: bytes):
    tmp = Path(str(path) + '.tmp')
    tmp.write_bytes(data)
    os.replace(tmp, path)
    os.chmod(path, 0o444)


def write_case(outdir, arrays, meta, sealed, key, manifest):
    stem = outdir / f"{meta['dim']}d" / meta['cohort'] / meta['kind'] / f"c{meta['case']:03d}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    npz = SI.npz_bytes(**arrays)
    js = json.dumps(V_clean(meta), indent=1).encode()
    if sealed:
        for suffix, data in (('.npz.enc', npz), ('.json.enc', js)):
            blob = SI.encrypt(data, key)
            assert SI.decrypt(blob, key) == data
            atomic_write(str(stem) + suffix, blob)
            manifest.append(dict(file=str(stem.relative_to(outdir)) + suffix, plain=SI.sha(data), cipher=SI.sha(blob)))
    else:
        atomic_write(str(stem) + '.npz', npz)
        atomic_write(str(stem) + '.json', js)       # json last: its presence marks a completed case
        manifest.append(dict(file=str(stem.relative_to(outdir)) + '.npz', sha256=SI.sha(npz)))
        manifest.append(dict(file=str(stem.relative_to(outdir)) + '.json', sha256=SI.sha(js)))


def qc_aggregate(metas, gpu_seconds):
    """Frozen QC schema for sealed / OOD cohorts (A1.11, A2.9, A5.5): per kind, counts and quantiles only."""
    def q(vals):
        v = [x for x in vals if x is not None and np.isfinite(x)]
        return dict(n=len(v), min=min(v) if v else None, median=float(np.median(v)) if v else None,
                    max=max(v) if v else None)
    out = dict(gpu_seconds=gpu_seconds, kinds={})
    for kind in sorted({m['kind'] for m in metas}):
        ms = [m for m in metas if m['kind'] == kind]
        jd = [m['judge'] for m in ms]
        k = dict(case_kinds=len(ms), outcomes={o: sum(m['outcome'] == o for m in ms) for o in V.OUTCOMES},
                 refined=sum(m['refined'] for m in ms), tight=sum(m['tight_used'] for m in ms),
                 attempts=sum(len(m['attempts']) for m in ms),
                 failed_attempts=sum(not a['steps_ok'] for m in ms for a in m['attempts']),
                 p=q([p for x in jd for p in x['p']]), p_undefined=sum(p is None for x in jd for p in x['p']),
                 p_values=5 * len(ms), eps_ref=q([x['eps_ref'] for x in jd]), eps_ref_undefined=sum(
                     x['eps_ref'] is None for x in jd), peclet=q([x['peclet'] for x in jd]),
                 newton_total=q([float(sum(a['newton_total'] for a in m['attempts'])) for m in ms]))
        out['kinds'][kind] = k
    return out


def calib_status_ok(d):
    """Production refusal (A5.4): the committed calibration status must have passed with this FLOOR_STAB."""
    p = HERE / 'checks' / f'calib_{d}d.json'
    if not p.exists():
        return False, 'missing checks/calib_%dd.json' % d
    st = json.loads(p.read_text())
    if st.get('floor_stab') != V.FLOOR_STAB:
        return False, f"calibrated FLOOR_STAB {st.get('floor_stab')} != {V.FLOOR_STAB}"
    if not st.get('proxy_ok_all'):
        return False, 'calibration proxy failed'
    return True, 'provisional' if st.get('provisional') else 'ok'      # A11: provisional outputs are held


def mode_fd(cfg, outdir, rep):
    d = int(cfg['dim'])
    ok, why = calib_status_ok(d)
    rep['calib_status'] = why
    rep['floor_stab'] = V.FLOOR_STAB                    # A11.5: the factor this job judges with (sealed release)
    if not ok and not cfg.get('smoke_override'):
        log(f'REFUSING fd mode: {why}', sealed_ok=True)
        raise SystemExit(5)
    sealed = CH.is_sealed(d, cfg['cohort'])
    key = cfg.get('key')
    seed = None
    # jcp-regimes edit E2: sealed / OOD cohorts are never opened by this lane (DESIGN 3.2); only regime cohorts and the
    # E0 validation cohort dev6 (R-repro control) are admissible
    if sealed or 'key' in cfg or 'seeds_file' in cfg:
        raise SystemExit('jcp-regimes never opens sealed / OOD cohorts')
    if cfg['cohort'] in RG.REGIMES_2D:
        coh = RG.Cohort(d, cfg['cohort'])
    elif (d, cfg['cohort']) == (2, 'dev6'):
        coh = CH.Cohort(d, cfg['cohort'])
    else:
        raise SystemExit(f"cohort {cfg['cohort']} not admissible in jcp-regimes")
    cases = cfg.get('cases') or list(range(coh.n))
    rep['cohort'] = dict(name=coh.name, n=coh.n, sha256=None if sealed else coh.sha256, cases=cases,
                         n_cases=len(cases))          # case indices are not secret; descriptors are
    if sealed and list(outdir.rglob('*.enc')):
        raise SystemExit('sealed jobs never resume (A5.5): ciphertext already present in the job directory')
    prev = {} if sealed else PREV_MANIFEST
    manifest, metas = [], []
    rep['cases'] = []
    done = 0
    for j in cases:
        for kind in cfg['kinds']:
            stem = outdir / f'{d}d' / coh.name / kind / f'c{j:03d}'
            if sealed:
                if any(Path(str(stem) + s).exists() for s in ('.npz.enc', '.json.enc')):
                    raise SystemExit('sealed jobs never resume (A5.5): output exists')
            elif all(Path(str(stem) + s).exists() and prev.get(str(Path(str(stem) + s).relative_to(outdir)))
                     == SI.sha(Path(str(stem) + s).read_bytes()) for s in ('.npz', '.json')):
                # verified resume: both files match the hashes this job recorded when it wrote them (A6.5)
                meta = json.loads(Path(str(stem) + '.json').read_text())
                npz = Path(str(stem) + '.npz').read_bytes()
                arr = np.load(str(stem) + '.npz')
                assert (meta['cohort'], meta['case'], meta['kind']) == (coh.name, j, kind) and meta['final'][2] in arr
                manifest.append(dict(file=str(stem.relative_to(outdir)) + '.npz', sha256=SI.sha(npz)))
                manifest.append(dict(file=str(stem.relative_to(outdir)) + '.json',
                                     sha256=SI.sha(Path(str(stem) + '.json').read_bytes())))
                rep['cases'].append(dict(case=j, kind=kind, outcome=meta['outcome'], eps_ref=meta['judge']['eps_ref'],
                                         p=meta['judge']['p'], seconds=meta.get('seconds'), resumed=True))
                log(f'resumed (verified) {stem}')
                continue
            for s in ('.npz', '.json'):                    # unverifiable leftovers are recomputed, never trusted
                if Path(str(stem) + s).exists():
                    os.chmod(str(stem) + s, 0o644)
                    os.remove(str(stem) + s)
            t = time.perf_counter()
            arrays, meta = solve_case(coh, j, kind)
            meta['job'] = job_info()
            meta['calib_status'] = rep['calib_status']
            meta['seconds'] = time.perf_counter() - t
            write_case(outdir, arrays, meta, sealed, key, manifest)
            metas.append(meta)
            done += 1
            jd = meta['judge']
            if sealed:
                log(f'sealed item {done}/{len(cases) * len(cfg["kinds"])} written', sealed_ok=True)
            else:
                rep['cases'].append(dict(case=j, kind=kind, outcome=meta['outcome'], refined=meta['refined'],
                                         tight=meta['tight_used'], eps_ref=jd['eps_ref'], p=jd['p'],
                                         seconds=meta['seconds']))
                log(f"CASE {coh.name} {j} {kind}: {meta['outcome']} eps={jd['eps_ref']} p={jd['p']} {meta['seconds']:.0f}s")
                rep['manifest'] = manifest
                save(outdir, rep)
    if sealed:
        del rep['cases']
        mbytes = json.dumps(manifest).encode()
        blob = SI.encrypt(mbytes, key)
        atomic_write(outdir / 'manifest.json.enc', blob)
        rep['sealed_manifest'] = dict(plain_sha256=SI.sha(mbytes), cipher_sha256=SI.sha(blob),
                                      ciphertexts=[dict(file=m['file'], cipher=m['cipher']) for m in manifest])
        rep['qc'] = qc_aggregate(metas, time.perf_counter() - T_BEGIN)
    else:
        rep['manifest'] = manifest


def mode_calib(cfg, outdir, rep):
    raise SystemExit('jcp-regimes: calibration is not re-run (DESIGN 4.1 reuses the committed E0 status)')
    d = int(cfg['dim'])
    name, cases = CALIB_CASES[d]
    coh = CH.Cohort(d, name)
    rep['calib'], rep['d1'], rep['x1'], rep['x4'] = [], [], [], None
    store = {}

    rep['attempt_records'] = {}

    def keep(tag, rec, lat, res, nwt):
        store[tag] = lat
        store['res_' + tag] = res
        store['nwt_' + tag] = nwt
        rep['attempt_records'][tag] = rec                # every attempt's record is kept (A6.2)
        return rec

    for j in cases:
        n0 = float(np.linalg.norm(RC.ic_lattice(coh, j).ravel()))
        for kind in ('ST', 'S'):
            cfgk = LEVELS[kind][d]
            for L, dt in cfgk['base']:
                recs = {}
                for purpose in ('prod', 'tight', 'tighter'):
                    rec, lat, res, nwt = attempt(coh, j, L, dt, cfgk['scheme'], purpose)
                    recs[purpose] = keep(f'{kind}_c{j}_L{L}_{purpose}', rec, lat, res, nwt)
                diff = lambda a, b: float(np.max(V.tnorm(store[f'{kind}_c{j}_L{L}_{a}'] -
                                                          store[f'{kind}_c{j}_L{L}_{b}'])[1:]) / n0)
                row = dict(case=j, kind=kind, L=L, dt=dt, prod_vs_tight=diff('prod', 'tight'),
                           tight_vs_tighter=diff('tight', 'tighter'), records=recs)
                solved = all(recs[q]['steps_ok'] and np.isfinite(recs[q]['tau']) for q in recs)
                row['all_solves_ok'] = bool(solved)
                row['proxy_ok'] = bool(solved and row['prod_vs_tight'] <= recs['prod']['tau'] and
                                       row['tight_vs_tighter'] <= recs['tight']['tau'])
                row['ratio_prod'] = row['prod_vs_tight'] / recs['prod']['tau']
                row['ratio_tight'] = row['tight_vs_tighter'] / recs['tight']['tau']
                rep['calib'].append(row)
                log(f"CALIB {kind} c{j} L={L}: prod-tight {row['prod_vs_tight']:.3e} (tau {recs['prod']['tau']:.3e}) "
                    f"tight-tighter {row['tight_vs_tighter']:.3e} (tau {recs['tight']['tau']:.3e}) ok={row['proxy_ok']} "
                    f"prod solve {recs['prod']['solve_seconds']:.1f}s peak {recs['prod']['peak_bytes']}")
                save(outdir, rep)
        # D1 (R_ST): time-only ladder at the finest L (dt of the three levels), space-only ladder at the finest dt
        (L1, dt1), (L2, dt2), (L3, dt3) = LEVELS['ST'][d]['base']
        tl = []
        for dt in (dt1, dt2, dt3):
            tag = f'ST_c{j}_L{L3}_prod' if dt == dt3 else f'D1t_c{j}_L{L3}_dt{dt:.3e}'
            if tag not in store:
                keep(tag, *attempt(coh, j, L3, dt, 'cnr', 'prod'))
            tl.append(store[tag])
        sl = []
        for L in (L1, L2, L3):
            tag = f'ST_c{j}_L{L3}_prod' if L == L3 else f'D1s_c{j}_L{L}_dt{dt3:.3e}'
            if tag not in store:
                keep(tag, *attempt(coh, j, L, dt3, 'cnr', 'prod'))
            sl.append(store[tag])
        for name_, lad, hs in (('time', tl, (dt1, dt2, dt3)), ('space', sl, (1 / L1, 1 / L2, 1 / L3))):
            d12, d23 = V.tnorm(lad[0] - lad[1])[1:], V.tnorm(lad[1] - lad[2])[1:]
            rep['d1'].append(dict(case=j, ladder=name_, p=[V.solve_order(d12[k], d23[k], *hs) for k in range(5)],
                                  d12=(d12 / n0).tolist(), d23=(d23 / n0).tolist()))
        log(f"D1 c{j}: {rep['d1'][-2:]}")
        save(outdir, rep)
    for j in range(coh.n):                     # X1: upwind + BE ladder on every dev case (must be rejected)
        lv = []
        for L, dt in X1_LEVELS[d]:
            rec, lat, res, nwt = attempt(coh, j, L, dt, 'be', 'prod', adv='upwind')
            keep(f'X1_c{j}_L{L}', rec, lat, res, nwt)
            lv.append(dict(L=L, lat=lat, tau=rec['tau'], steps_ok=rec['steps_ok'], umax=rec['umax_running']))
        jd = V.judge(lv, RC.ic_lattice(coh, j), coh.nu(j))
        rep['x1'].append(dict(case=j, p=jd['p'], outcome=jd['outcome']))
        log(f"X1 c{j}: p={jd['p']} {jd['outcome']}")
        save(outdir, rep)
    if d == 2:                                 # X4 probe: dev6 case 0, R_ST, middle level dt doubled
        lv = []
        for i, (L, dt) in enumerate(LEVELS['ST'][2]['base']):
            dtx = dt * (2 if i == 1 else 1)
            tag = f'ST_c0_L{L}_prod' if (i != 1 and 0 in cases) else f'X4_L{L}_dt{dtx:.3e}'
            if tag not in store:
                keep(tag, *attempt(coh, 0, L, dtx, 'cnr', 'prod'))
            r_ = rep['attempt_records'][tag]
            lv.append(dict(L=L, lat=store[tag], tau=r_['tau'], steps_ok=r_['steps_ok'], umax=r_['umax_running']))
        jd = V.judge(lv, RC.ic_lattice(coh, 0), coh.nu(0))
        rep['x4'] = dict(p=jd['p'], outcome=jd['outcome'])
        log(f"X4: {rep['x4']}")
    np.savez(outdir / f'calib_{d}d_fields.npz', **store)
    rep['calib_fields_sha256'] = SI.sha((outdir / f'calib_{d}d_fields.npz').read_bytes())
    rep['calib_status'] = dict(dim=d, floor_stab=V.FLOOR_STAB, provisional=False,
                               proxy_ok_all=all(r['proxy_ok'] for r in rep['calib']),
                               worst_ratio_prod=max(r['ratio_prod'] for r in rep['calib']),
                               worst_ratio_tight=max(r['ratio_tight'] for r in rep['calib']),
                               job=job_info(), fields_sha256=rep['calib_fields_sha256'])


def ps_case(coh, j, Ns, dt, attempt_no, purpose_tag, dt_base, ps_gates_ok):
    d = coh.dim
    m = RC.LATTICE[d]
    runs = []
    for N in Ns:
        for dtx, purpose in [(dt, purpose_tag)] + ([(dt / 2, 'half')] if N == Ns[-1] else []):
            t = time.perf_counter()
            if hasattr(coh, 'ps_coefs'):        # jcp-regimes edit E4: regime ICs (two-bump families are sums)
                b0, b0b = coh.ps_coefs(j, N), coh.ps_coefs(j, N, 512)
            else:
                b0 = PS.ic_coefs_2d(coh.D[j], N) if d == 2 else PS.ic_coefs_3d(coh.T, j, N)
                b0b = PS.ic_coefs_2d(coh.D[j], N, 512) if d == 2 else PS.ic_coefs_3d(coh.T, j, N, 512)
            ic_rel = float(np.linalg.norm(b0 - b0b) / np.linalg.norm(b0b))
            ic_max = float(np.max(np.abs(b0 - b0b)))
            Lk = jnp.asarray(PS.lin_eig(N, d, coh.nu(j)))
            c32, c64 = PS.etdrk4_coefs(Lk, dtx, 32), PS.etdrk4_coefs(Lk, dtx, 64)
            cv = [float(jnp.max(jnp.abs(c32[k] - c64[k])) / jnp.max(jnp.abs(c64[k]))) for k in ('Q', 'f1', 'f2', 'f3')]
            contour = max(cv) if all(np.isfinite(cv)) else float('inf')
            del c32, c64, Lk, b0b
            S = PS.lattice_S(N, m)
            if (N, d) not in _PS:
                _PS[(N, d)] = PS.make_ps(N, d)
            r = _PS[(N, d)](jnp.asarray(b0), coh.nu(j), dtx, S)
            kexp = int(round(np.log2(dt_base / dtx)))
            checks_ok = bool(np.isfinite(ic_rel) and ic_rel <= 1e-12 and ic_max <= 1e-13 and contour <= 1e-10
                             and ps_gates_ok)
            runs.append(dict(id=f'N{N}_dt{kexp}_a{attempt_no}', N=N, dt=dtx, attempt=attempt_no, purpose=purpose,
                             umax=r['umax'], finite=r['finite'], ic_quad_rel=ic_rel, ic_quad_max=ic_max,
                             contour_rel=contour, job_gates_ok=ps_gates_ok, checks_ok=checks_ok, lat=r['lat'],
                             seconds=time.perf_counter() - t, peak_bytes=RC.peak_bytes()))
            log(f"  PS {coh.name} c{j} {runs[-1]['id']} umax={r['umax']:.3f} finite={r['finite']} checks={checks_ok} "
                f"{runs[-1]['seconds']:.0f}s peak={runs[-1]['peak_bytes']}")
            del r
    return runs


def stability_ok(runs, d, dt, N3):
    um = [r['umax'] for r in runs]
    if not all(np.isfinite(u) for u in um):        # NaN-safe: any non-finite maximum fails (A8.2)
        return False
    return bool(dt <= 0.66 * 2.8 / (2 * d * np.pi * N3 * max(um)))


def mode_ps(cfg, outdir, rep):
    d = int(cfg['dim'])
    if cfg['cohort'] not in RG.REGIMES_2D:          # jcp-regimes edit E2: regime cohorts only
        raise SystemExit(f"cohort {cfg['cohort']} not admissible in jcp-regimes ps mode")
    coh = RG.Cohort(d, cfg['cohort'])
    att = int(cfg.get('attempt', 0))
    assert att in (0, 1), 'one escalation at most (A1.7)'
    Ns = cfg.get('Ns') or (PS_CFG[d]['Ns'] if att == 0 else PS_CFG[d]['esc'])
    dt_base = PS_CFG[d]['dt']
    for j in cfg['cases']:
        stem = outdir / f'{d}d' / coh.name / 'PS' / f'c{j:03d}_a{att}'
        if Path(str(stem) + '.json').exists():
            continue
        dt = dt_base
        runs = ps_case(coh, j, Ns, dt, 2 * att, 'spatial', dt_base, rep['gates_all_pass'])
        stab = stability_ok(runs, d, dt, Ns[-1])
        if not stab:
            dt = dt / 2
            runs += ps_case(coh, j, Ns, dt, 2 * att + 1, 'stability', dt_base, rep['gates_all_pass'])
            stab = stability_ok([r for r in runs if r['attempt'] == 2 * att + 1], d, dt, Ns[-1])
        sel_att = max(r['attempt'] for r in runs)
        sel = [r for r in runs if r['attempt'] == sel_att]
        spatial = [next(r['id'] for r in sel if r['N'] == N and r['dt'] == dt) for N in Ns]
        half = next(r['id'] for r in sel if r['N'] == Ns[-1] and r['dt'] == dt / 2)
        arrays = {r['id']: r['lat'] for r in runs}
        arrays['t0_ps'] = next(r['lat'] for r in sel if r['id'] == half)[0]
        arrays['t0'] = RC.ic_lattice(coh, j)
        meta = dict(dim=d, cohort=coh.name, case=j, desc=coh.desc(j), Ns=Ns, dt_base=dt_base, ladder_attempt=att,
                    runs=[{k: v for k, v in r.items() if k != 'lat'} for r in runs], selected_spatial=spatial,
                    selected_half=half, stability_ok=stab, gates_ok=bool(all(r['checks_ok'] for r in sel)),
                    finite=bool(all(r['finite'] for r in sel)), job=job_info())
        stem.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(str(stem) + '.npz', SI.npz_bytes(**arrays))
        atomic_write(str(stem) + '.json', json.dumps(V_clean(meta), indent=1).encode())
        rep.setdefault('ps_cases', []).append(dict(case=j, attempt=att, stability_ok=stab, finite=meta['finite'],
                                                   gates_ok=meta['gates_ok']))
        save(outdir, rep)


def save(outdir, rep):
    tmp = outdir / 'result.json.tmp'
    tmp.write_text(json.dumps(V_clean(rep), indent=1))
    tmp.replace(outdir / 'result.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    if not cfg.get('allow_cpu'):
        assert jax.default_backend() == 'gpu', jax.default_backend()
    assert os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest'
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    if cfg.get('mode') == 'fd' and list(outdir.rglob('*.enc')):
        raise SystemExit('sealed jobs never resume (A5.5): ciphertext already present in the job directory')
    if (outdir / 'result.json').exists():           # read BEFORE the first save overwrites it (verified resume)
        PREV_MANIFEST.update({m['file']: m['sha256'] for m in
                              json.loads((outdir / 'result.json').read_text()).get('manifest', []) if 'sha256' in m})
    rep = dict(config={k: v for k, v in cfg.items() if k not in ('seeds_file', 'key')}, job=job_info(), complete=False)
    if cfg.get('smoke_override'):            # LOCAL FUNCTIONAL SMOKE ONLY (A4.4 v): reduced levels; never a reference
        assert not os.environ.get('SLURM_JOB_ID'), 'smoke_override is refused inside a cluster job'
        so = cfg['smoke_override']
        for kind, dd in so.get('levels', {}).items():
            for dim, v in dd.items():
                LEVELS[kind][int(dim)] = dict(base=[tuple(x) for x in v['base']], refine=tuple(v['refine']),
                                              scheme=v['scheme'])
        for dim, v in so.get('ps', {}).items():
            PS_CFG[int(dim)].update(v)
        for dim, v in so.get('x1', {}).items():
            X1_LEVELS[int(dim)] = [tuple(x) for x in v]
        rep['SMOKE_OVERRIDE'] = True
    pre = G.run_all('units')
    for dd in (2, 3):
        pre['gates'].append(G.g1_parity(dd))
    if cfg['mode'] == 'gates':
        full = G.run_all('all', ps_N={2: PS_CFG[2]['esc'][-1], 3: PS_CFG[3]['esc'][-1]},
                         ps_dt={2: PS_CFG[2]['dt'], 3: PS_CFG[3]['dt']}, units=False)
        pre['gates'] += full['gates']
    if cfg['mode'] == 'ps':                  # PS jobs: G3/G4 of their dimension in the same job (A2.4 step 2)
        dd = int(cfg['dim'])
        att = int(cfg.get('attempt', 0))
        Ns = cfg.get('Ns') or (PS_CFG[dd]['Ns'] if att == 0 else PS_CFG[dd]['esc'])
        ps_g = G.run_all('ps', ps_N={dd: Ns[-1]}, ps_dt={dd: PS_CFG[dd]['dt']}, dims=(dd,), units=False)
        pre['gates'] += ps_g['gates']
    rep['gates'] = pre['gates']
    rep['gates_all_pass'] = all(g['pass'] for g in pre['gates'])
    save(outdir, rep)
    log(f"gates all_pass={rep['gates_all_pass']}: " +
        ', '.join(f"{g['gate']}{g.get('d', '')}={g['pass']}" for g in pre['gates']), sealed_ok=True)
    if not rep['gates_all_pass']:
        log('GATE FAILURE: refusing to continue', sealed_ok=True)
        raise SystemExit(4)
    if cfg['mode'] == 'fd':
        mode_fd(cfg, outdir, rep)
    elif cfg['mode'] == 'calib':
        mode_calib(cfg, outdir, rep)
    elif cfg['mode'] == 'ps':
        mode_ps(cfg, outdir, rep)
    rep['complete'] = True
    rep['seconds'] = time.perf_counter() - T_BEGIN
    rep['peak_bytes'] = RC.peak_bytes()
    save(outdir, rep)
    print('JOB COMPLETE', flush=True)


if __name__ == '__main__':
    main()
