"""jcp-budget reference job (DESIGN 3; gates G-R1..G-R3, control C-R1). Adapted from E0 `refjob.py` @cc04f4b54
(solve_case, attempt, ps_case, stability_ok, mode_ps): same levels, tolerances, retry / tight / refinement procedure and
judge, generalised to a horizon T (outputs every 0.05 to T) and with a fine observation route. Never sealed cohorts:
only 2D dev6 / val32 and 3D val64 are accepted (asserted). Writes only under --out (this lane's namespace).

    python refrun.py --config <cfg.json> --out <dir>

cfg: {"mode": "fd" | "ps" | "x1" | "e0repro", "dim": 2|3, "cohort": "dev6"|"val32"|"val64", "cases": [...],
      "horizon": 0.5, "fine": true, "allow_refine": true}
  fd       R_ST to the horizon: three levels -> judge at every output time (lattice); floor failure -> tight; order /
           Peclet failure -> refinement level if allow_refine (DESIGN 3.1: T = 0.5 only); fine-lattice fields of all
           three final levels judged too (fine eps_ref); stored: lattice fields of every attempt, the finest level's
           fine-lattice field, t0 on both lattices, per-stage residuals / Newton counts.
  e0repro  G-R1: horizon 0.25, no fine route, the three base levels at prod (E0's first attempt) for the listed cases;
           lattice fields saved for the offline comparison with the E0 store.
  x1       C-R1: the E0 first-order ladder (upwind + BE, X1_LEVELS) to the horizon on the listed cases; judged.
  ps       pseudo-spectral cross-check runs (E0 PS_CFG sizes, horizon generalised), preceded by E0's PS gates.
Every mode first runs E0's unit gates and G1 parity (cheap) at the E0 defaults and refuses to continue on failure.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import cohorts as CH
import gates as G
import pscore as PS
import refcore as RC
import verdict as V

LEVELS = {2: dict(base=[(2048, 1.25e-3), (4096, 6.25e-4), (8192, 3.125e-4)], refine=(12288, 0.05 / 240)),
          3: dict(base=[(128, 2.5e-3), (256, 1.25e-3), (512, 6.25e-4)], refine=(768, 0.05 / 120))}   # E0 'ST'
TOL = {'prod': (1e-11, 1e-9, 20), 'retry': (1e-11, 1e-10, 40), 'tight': (1e-13, 1e-12, 40)}           # E0 TOL
PS_CFG = {2: dict(Ns=[1024, 1536, 2048], dt=0.05 / 1600, esc=[1536, 2048, 3072]),
          3: dict(Ns=[128, 192, 256], dt=0.05 / 320, esc=[192, 256, 384])}                          # E0 PS_CFG (A8.1)
X1_LEVELS = {2: [(512, 2.5e-3), (1024, 1.25e-3), (2048, 6.25e-4)], 3: [(64, 5e-3), (128, 2.5e-3), (256, 1.25e-3)]}
ALLOWED = {2: ('dev6', 'val32'), 3: ('val64',)}
T_BEGIN = time.perf_counter()
_SOLVERS, _PS = {}, {}


def log(s):
    print(f'[{time.perf_counter() - T_BEGIN:9.1f}s] {s}', flush=True)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def npz_bytes(**arrays):
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    return x


def job_info():
    return dict(job_id=os.environ.get('SLURM_JOB_ID'), commit=os.environ.get('SOURCE_COMMIT'),
                gpu=jax.devices()[0].device_kind, backend=jax.default_backend(), jax=jax.__version__,
                numpy=np.__version__, precision=os.environ.get('JAX_DEFAULT_MATMUL_PRECISION'))


def nout_of(T):
    n = int(round(T / 0.05))
    assert abs(n * 0.05 - T) < 1e-12 and n >= 5, T
    return n + 1


def solver(d, L, adv, max_newton, nout, nst):
    key = (d, L, adv, max_newton, nout, nst)
    if key not in _SOLVERS:
        _SOLVERS.clear()                       # one level resident at a time (memory)
        _SOLVERS[key] = RC.make_solver(d, L, adv=adv, max_newton=max_newton, nout=nout, nst=nst)
    return _SOLVERS[key]


def attempt(coh, j, L, dt, scheme, purpose, T, fine, nst, adv='central'):
    """One level run -> (record, lattice fields, fine fields or None, residuals, Newton counts)."""
    d = coh.dim
    ntol, ltol, mn = TOL['prod' if purpose in ('prod', 'refine') else purpose]
    t = time.perf_counter()
    u0 = coh.ic(L, j)
    nout = nout_of(T)
    run = solver(d, L, adv, mn, nout, nst)
    r = run(u0, coh.nu(j), dt, scheme, ntol, ltol, horizon=T)
    out = jax.block_until_ready(r['out'])
    lat = RC.to_lattice(out, L, d)
    fin = RC.to_fine(out, L, d) if fine else None
    ns, res, nwt, umax = int(r['stages']), np.asarray(r['res']), np.asarray(r['nwt']), float(r['umax'])
    del out, r, u0
    nexp = RC.n_stages(dt, scheme, T)
    rec = dict(purpose=purpose, L=L, dt=dt, scheme=scheme, adv=adv, ntol=ntol, ltol=ltol, max_newton=mn, stages=ns,
               expected_stages=nexp, steps_ok=bool(V.steps_ok(res, ns) and ns == nexp),
               max_res=float(np.nanmax(res[:ns])) if ns else None, tau=V.floor_proxy(res, ns),
               newton_total=int(nwt[:ns].sum()), newton_max=int(nwt[:ns].max()), umax_running=umax,
               lattice_finite=bool(np.isfinite(lat).all()) and (fin is None or bool(np.isfinite(fin).all())),
               seconds=time.perf_counter() - t, compile_seconds=run.compile_seconds, peak_bytes=RC.peak_bytes())
    rec['steps_ok'] = rec['steps_ok'] and rec['lattice_finite']
    log(f"  {coh.name} c{j} L={L} dt={dt:.3e} {purpose} adv={adv} stages={ns}/{nexp} ok={rec['steps_ok']} "
        f"maxres={rec['max_res']:.2e} newton={rec['newton_total']} {rec['seconds']:.0f}s peak={rec['peak_bytes']}")
    return rec, lat, fin, res[:ns], nwt[:ns]


WINDOWS = {'W1': (1, 5), 'W2': (6, 10), 'W3': (11, 20)}


def judge_windows(levels, t0, nu, T):
    """A1.5: the E0 judge on the fields restricted to {t0} + window (order, direction, floor per window time; solver, IC
    and Peclet are whole-run properties and are identical in every window)."""
    n = nout_of(T)
    out = {}
    for w, (a, b) in WINDOWS.items():
        if b >= n:
            continue
        idx = [0] + list(range(a, b + 1))
        lv = [dict(lvl, lat=np.asarray(lvl['lat'])[idx]) for lvl in levels]
        out[w] = V.judge(lv, t0, nu)
    return out


def fine_t0(coh, j):
    d = coh.dim
    m = RC.FINE[d]
    if d == 2:
        return np.asarray(CH.ic_2d(m, coh.D[j]))
    return np.asarray(CH.ic_3d(m, coh.T, j)).reshape(-1)


def solve_case(coh, j, T, fine, allow_refine):
    """E0 refjob.solve_case (DESIGN §4 procedure) at horizon T; refinement only if allow_refine."""
    d = coh.dim
    cfg = LEVELS[d]
    nst = RC.nst_for(cfg['base'] + [cfg['refine']], 'cnr', T)
    t0 = RC.ic_lattice(coh, j)
    attempts, arrays, fines = [], {}, {}

    def run(L, dt, purpose):
        rec, lat, fin, res, nwt = attempt(coh, j, L, dt, 'cnr', purpose, T, fine, nst)
        rec['id'] = f'a{len(attempts)}_L{L}'
        attempts.append(rec)
        arrays[rec['id']] = lat
        arrays['res_' + rec['id']] = res
        arrays['nwt_' + rec['id']] = nwt
        if fin is not None:
            fines[rec['id']] = fin
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
    if allow_refine and ok(jd['checks']) and jd['checks']['floor'] and not (jd['checks']['order'] and jd['checks']['peclet']):
        L, dt = cfg['refine']
        triple = triple[1:] + [get(L, dt, 'tight' if tight_used else 'refine')]
        refined = True
        jd = judge(triple)
        if ok(jd['checks']) and not jd['checks']['floor'] and not tight_used:
            triple = [get(x[0]['L'], dts[x[0]['L']], 'tight') for x in triple]
            tight_used = True
            jd = judge(triple)
    final = [x[1] for x in triple]
    arrays['t0'] = t0
    jf = jfw = None
    fine_arrays = None
    jw = judge_windows([x[0] for x in triple], t0, coh.nu(j), T)
    if fine:
        tf = fine_t0(coh, j)
        arrays['t0_fine'] = tf
        if all(i in fines for i in final):
            lv = [dict(L=x[0]['L'], lat=fines[x[1]], tau=x[0]['tau'], steps_ok=x[0]['steps_ok'], umax=x[0]['umax'])
                  for x in triple]
            jf = V.judge(lv, tf, coh.nu(j))
            jfw = judge_windows(lv, tf, coh.nu(j), T)
        fine_arrays = {'fine_' + final[2]: fines[final[2]], 't0_fine': arrays.pop('t0_fine')}
        # G-R3: the fine lattice contains the scoring lattice (2D stride 4 incl. walls; 3D: the 63^3 nodes are the
        # fine nodes with odd index 2k-1 -> k/64): bitwise equality of the shared nodes for every final level
        g3 = []
        for i in final:
            F, Lt = fines[i], arrays[i]
            sf = RC.FINE[d] // RC.LATTICE[d]
            if d == 2:
                g3.append(bool(np.array_equal(F[:, ::sf, ::sf], Lt)))
            else:
                mf = RC.FINE[d] - 1
                Fc = F.reshape((F.shape[0],) + (mf,) * 3)[:, sf - 1::sf, sf - 1::sf, sf - 1::sf]
                g3.append(bool(np.array_equal(Fc.reshape(F.shape[0], -1), Lt)))
        g_r3 = all(g3)
    meta = dict(dim=d, cohort=coh.name, case=j, kind='ST', horizon=T, times=RC.times(T).tolist(), desc=coh.desc(j),
                cohort_sha256=coh.sha256, lattice=RC.LATTICE[d], fine_lattice=RC.FINE[d] if fine else None,
                metric_id=f'{d}|lattice={RC.LATTICE[d]}|times=0:0.05:{T:g}|norm=fixed_t0|max_k>=1',
                t0_sha256=sha(np.ascontiguousarray(t0).tobytes()), scheme='cnr', attempts=attempts, final=final,
                history=history, judge=jd, judge_fine=jf, judge_windows=jw, judge_fine_windows=jfw, outcome=jd['outcome'], refined=refined, tight_used=tight_used,
                allow_refine=allow_refine, floor_stab=V.FLOOR_STAB, G_R3=(g_r3 if fine else None))
    meta['fine_file'] = None if fine_arrays is None else f'c{j:03d}_fine.npz'
    return arrays, fine_arrays, meta


def write(outdir, stem, arrays, meta, manifest):
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suf, data in (('.npz', npz_bytes(**arrays)), ('.json', json.dumps(clean(meta), indent=1).encode())):
        p = Path(str(stem) + suf)
        tmp = Path(str(p) + '.tmp')
        tmp.write_bytes(data)
        os.replace(tmp, p)
        manifest.append(dict(file=str(p.relative_to(outdir)), sha256=sha(data)))


def mode_fd(cfg, outdir, rep):
    d, T = int(cfg['dim']), float(cfg['horizon'])
    coh = CH.Cohort(d, cfg['cohort'])
    rep['cohort'] = dict(name=coh.name, n=coh.n, sha256=coh.sha256)
    for j in cfg['cases']:
        stem = outdir / f'{d}d' / coh.name / f'T{T:g}' / f'c{j:03d}'
        if Path(str(stem) + '.json').exists():
            log(f'exists, skipped (no resume trust): {stem}')
            continue
        t = time.perf_counter()
        arrays, fine_arrays, meta = solve_case(coh, j, T, bool(cfg.get('fine', True)), bool(cfg.get('allow_refine', False)))
        meta['job'], meta['seconds'] = job_info(), time.perf_counter() - t
        if fine_arrays is not None:            # the fine field in its own file (it stays on the cluster; A1.8)
            data = npz_bytes(**fine_arrays)
            pf = Path(str(stem) + '_fine.npz')
            pf.parent.mkdir(parents=True, exist_ok=True)
            pf.write_bytes(data)
            rep['manifest'].append(dict(file=str(pf.relative_to(outdir)), sha256=sha(data)))
            meta['fine_sha256'] = sha(data)
        write(outdir, stem, arrays, meta, rep['manifest'])
        jd = meta['judge']
        rep['cases'].append(dict(case=j, outcome=meta['outcome'], refined=meta['refined'], tight=meta['tight_used'],
                                 eps_ref=jd['eps_ref'], eps_fine=(meta['judge_fine'] or {}).get('eps_ref'),
                                 G_R3=meta['G_R3'], seconds=meta['seconds']))
        log(f"CASE {coh.name} {j} T={T:g}: {meta['outcome']} eps={jd['eps_ref']} p={jd['p']} G_R3={meta['G_R3']}")
        save(outdir, rep)


def mode_e0repro(cfg, outdir, rep):
    """G-R1: the E0 base levels at horizon 0.25 at prod tolerance, no fine route; fields saved for the offline check."""
    d = int(cfg['dim'])
    coh = CH.Cohort(d, cfg['cohort'])
    nst = RC.NST
    for j in cfg['cases']:
        arrays = {}
        recs = []
        for L, dt in LEVELS[d]['base']:
            rec, lat, _, res, nwt = attempt(coh, j, L, dt, 'cnr', 'prod', 0.25, False, nst)
            arrays[f'L{L}'] = lat
            recs.append(rec)
        stem = outdir / f'{d}d' / coh.name / 'E0REPRO' / f'c{j:03d}'
        write(outdir, stem, arrays, dict(case=j, cohort=coh.name, attempts=recs, job=job_info()), rep['manifest'])
        save(outdir, rep)


def mode_x1(cfg, outdir, rep):
    """C-R1: the E0 first-order ladder (upwind + BE) to the horizon must be judged order-fail."""
    d, T = int(cfg['dim']), float(cfg['horizon'])
    coh = CH.Cohort(d, cfg['cohort'])
    nst = RC.nst_for(X1_LEVELS[d], 'be', T)
    rep['x1'] = []
    for j in cfg['cases']:
        lv = []
        for L, dt in X1_LEVELS[d]:
            rec, lat, _, res, nwt = attempt(coh, j, L, dt, 'be', 'prod', T, False, nst, adv='upwind')
            lv.append(dict(L=L, lat=lat, tau=rec['tau'], steps_ok=rec['steps_ok'], umax=rec['umax_running']))
        jd = V.judge(lv, RC.ic_lattice(coh, j), coh.nu(j))
        rep['x1'].append(dict(case=j, outcome=jd['outcome'], p=jd['p'], checks=jd['checks'],
                              fired=bool(jd['outcome'] != 'accepted')))
        log(f"X1 {coh.name} {j}: {jd['outcome']} p={jd['p']}")
        save(outdir, rep)


def ps_case(coh, j, Ns, dt, attempt_no, purpose_tag, dt_base, ps_gates_ok, T):
    """E0 refjob.ps_case with the horizon."""
    d = coh.dim
    m = RC.LATTICE[d]
    runs = []
    for N in Ns:
        for dtx, purpose in [(dt, purpose_tag)] + ([(dt / 2, 'half')] if N == Ns[-1] else []):
            t = time.perf_counter()
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
            r = _PS[(N, d)](jnp.asarray(b0), coh.nu(j), dtx, S, horizon=T)
            kexp = int(round(np.log2(dt_base / dtx)))
            checks_ok = bool(np.isfinite(ic_rel) and ic_rel <= 1e-12 and ic_max <= 1e-13 and contour <= 1e-10
                             and ps_gates_ok)
            runs.append(dict(id=f'N{N}_dt{kexp}_a{attempt_no}', N=N, dt=dtx, attempt=attempt_no, purpose=purpose,
                             umax=r['umax'], finite=r['finite'], ic_quad_rel=ic_rel, ic_quad_max=ic_max,
                             contour_rel=contour, job_gates_ok=ps_gates_ok, checks_ok=checks_ok, lat=r['lat'],
                             seconds=time.perf_counter() - t, peak_bytes=RC.peak_bytes()))
            log(f"  PS {coh.name} c{j} {runs[-1]['id']} umax={r['umax']:.3f} finite={r['finite']} checks={checks_ok} "
                f"{runs[-1]['seconds']:.0f}s")
            del r
    return runs


def stability_ok(runs, d, dt, N3):
    um = [r['umax'] for r in runs]
    if not all(np.isfinite(u) for u in um):
        return False
    return bool(dt <= 0.66 * 2.8 / (2 * d * np.pi * N3 * max(um)))


def mode_ps(cfg, outdir, rep):
    d, T = int(cfg['dim']), float(cfg['horizon'])
    coh = CH.Cohort(d, cfg['cohort'])
    att = int(cfg.get('attempt', 0))                 # A2.3: one escalation ladder (E0 A1.7), staged after an UNRESOLVED verdict
    assert att in (0, 1)
    Ns, dt_base = PS_CFG[d]['Ns' if att == 0 else 'esc'], PS_CFG[d]['dt']
    for j in cfg['cases']:
        stem = outdir / f'{d}d' / coh.name / f'T{T:g}' / 'PS' / f'c{j:03d}_a{att}'
        dt = dt_base
        runs = ps_case(coh, j, Ns, dt, 2 * att, 'spatial', dt_base, rep['gates_all_pass'], T)
        stab = stability_ok(runs, d, dt, Ns[-1])
        if not stab:
            dt = dt / 2
            runs += ps_case(coh, j, Ns, dt, 2 * att + 1, 'stability', dt_base, rep['gates_all_pass'], T)
            stab = stability_ok([r for r in runs if r['attempt'] == 2 * att + 1], d, dt, Ns[-1])
        sel_att = max(r['attempt'] for r in runs)
        sel = [r for r in runs if r['attempt'] == sel_att]
        spatial = [next(r['id'] for r in sel if r['N'] == N and r['dt'] == dt) for N in Ns]
        half = next(r['id'] for r in sel if r['N'] == Ns[-1] and r['dt'] == dt / 2)
        arrays = {r['id']: r['lat'] for r in runs}
        arrays['t0'] = RC.ic_lattice(coh, j)
        meta = dict(dim=d, cohort=coh.name, case=j, horizon=T, desc=coh.desc(j), Ns=Ns, dt_base=dt_base, ladder_attempt=att,
                    runs=[{k: v for k, v in r.items() if k != 'lat'} for r in runs], selected_spatial=spatial,
                    selected_half=half, stability_ok=stab, gates_ok=bool(all(r['checks_ok'] for r in sel)),
                    finite=bool(all(r['finite'] for r in sel)), job=job_info())
        write(outdir, stem, arrays, meta, rep['manifest'])
        rep['cases'].append(dict(case=j, stability_ok=stab, finite=meta['finite'], gates_ok=meta['gates_ok']))
        save(outdir, rep)


def save(outdir, rep):
    tmp = outdir / 'result.json.tmp'
    tmp.write_text(json.dumps(clean(rep), indent=1))
    tmp.replace(outdir / 'result.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    d = int(cfg['dim'])
    assert cfg['cohort'] in ALLOWED[d], ('cohort not allowed in this lane', cfg['cohort'])
    assert not CH.is_sealed(d, cfg['cohort'])
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    if not cfg.get('allow_cpu'):
        assert jax.default_backend() == 'gpu', jax.default_backend()
    assert os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest'
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    rep = dict(config=cfg, job=job_info(), complete=False, manifest=[], cases=[])
    pre = G.run_all('units')
    if not cfg.get('skip_g1'):
        pre['gates'].append(G.g1_parity(d))
    if cfg['mode'] == 'ps':
        Ns = PS_CFG[d]['Ns' if int(cfg.get('attempt', 0)) == 0 else 'esc']
        pre['gates'] += G.run_all('ps', ps_N={d: Ns[-1]}, ps_dt={d: PS_CFG[d]['dt']}, dims=(d,), units=False)['gates']
    rep['gates'] = pre['gates']
    rep['gates_all_pass'] = all(g['pass'] for g in pre['gates'])
    save(outdir, rep)
    log(f"gates all_pass={rep['gates_all_pass']}: " + ', '.join(f"{g['gate']}{g.get('d', '')}={g['pass']}" for g in pre['gates']))
    if not rep['gates_all_pass']:
        raise SystemExit(4)
    dict(fd=mode_fd, e0repro=mode_e0repro, x1=mode_x1, ps=mode_ps)[cfg['mode']](cfg, outdir, rep)
    rep['complete'] = True
    rep['seconds'] = time.perf_counter() - T_BEGIN
    rep['peak_bytes'] = RC.peak_bytes()
    save(outdir, rep)
    (outdir / 'COMPLETE').write_text('complete\n')
    print('JOB COMPLETE', flush=True)


if __name__ == '__main__':
    main()
