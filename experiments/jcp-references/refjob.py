"""jcp-references cluster job (DESIGN §3–§9, A1–A4).

    python refjob.py --config <cfg.json> --out <dir>

modes
  gates  all code gates (gates.run_all('all')), incl. G3b/G3c at the production PS sizes
  fd     R_ST / R_S references of one cohort: per case and kind, three levels -> judge; floor failure -> tight attempt
         at all three levels; order or Peclet failure -> refinement level, judged on the finest three; solver failure
         -> one retry (Newton 40, BiCGStab 1e-10). Writes one npz + json per (case, kind); encrypted for sealed cohorts.
  calib  floor-proxy calibration (prod / tight / tighter, every level, both kinds) on the calibration dev cases, the
         D1 time-only / space-only ladders, X1 (upwind+BE ladder on every dev case) and X4 (doubled middle dt)
  ps     pseudo-spectral cross-check runs (spatial triple at dt, half step at N3, stability rerun) for given cases
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
import gates as G
import pscore as PS
import refcore as RC
import sealio as SI
import verdict as V

LEVELS = {  # kind -> dim -> dict(base [(L, dt)], refine (L, dt), scheme)
    'ST': {2: dict(base=[(2048, 1.25e-3), (4096, 6.25e-4), (8192, 3.125e-4)], refine=(12288, 0.05 / 240), scheme='cnr'),
           3: dict(base=[(128, 2.5e-3), (256, 1.25e-3), (512, 6.25e-4)], refine=(768, 0.05 / 120), scheme='cnr')},
    'S': {2: dict(base=[(2048, .005), (4096, .005), (8192, .005)], refine=(12288, .005), scheme='be'),
          3: dict(base=[(128, .005), (256, .005), (512, .005)], refine=(768, .005), scheme='be')},
}
TOL = {'prod': (1e-11, 1e-9, 20), 'retry': (1e-11, 1e-10, 40), 'tight': (1e-13, 1e-12, 40),
       'tighter': (3e-14, 1e-12, 60)}
PS_CFG = {2: dict(Ns=[1024, 1536, 2048], dt=5e-5, esc=3072), 3: dict(Ns=[128, 192, 256], dt=2.5e-4, esc=384)}
CALIB_CASES = {2: ('dev6', [0, 4]), 3: ('dev4', [0, 2])}
X1_LEVELS = {2: [(512, 2.5e-3), (1024, 1.25e-3), (2048, 6.25e-4)], 3: [(64, 5e-3), (128, 2.5e-3), (256, 1.25e-3)]}

_SOLVERS = {}
T_BEGIN = time.perf_counter()


def log(s):
    print(f'[{time.perf_counter() - T_BEGIN:9.1f}s] {s}', flush=True)


def solver(d, L, adv, max_newton):
    key = (d, L, adv, max_newton)
    if key not in _SOLVERS:
        _SOLVERS[key] = RC.make_solver(d, L, adv=adv, max_newton=max_newton)
    return _SOLVERS[key]


def attempt(coh, j, L, dt, scheme, purpose, adv='central'):
    """One level run -> (record, lattice fields)."""
    d = coh.dim
    ntol, ltol, mn = TOL['prod' if purpose in ('prod', 'refine') else purpose]
    t = time.perf_counter()
    r = solver(d, L, adv, mn)(coh.ic(L, j), coh.nu(j), dt, scheme, ntol, ltol)
    out = jax.block_until_ready(r['out'])
    lat = RC.to_lattice(out, L, d)
    ns = int(r['stages'])
    umax_running = float(r['umax'])
    res = np.asarray(r['res'])
    nwt = np.asarray(r['nwt'])
    del out, r
    rec = dict(purpose=purpose, L=L, dt=dt, scheme=scheme, adv=adv, ntol=ntol, ltol=ltol, max_newton=mn, stages=ns,
               expected_stages=RC.n_stages(dt, scheme), steps_ok=V.steps_ok(res, ns) and ns == RC.n_stages(dt, scheme),
               max_res=float(np.nanmax(res[:ns])), tau=V.floor_proxy(res, ns), newton_total=int(nwt[:ns].sum()),
               newton_max=int(nwt[:ns].max()), umax_running=umax_running, seconds=time.perf_counter() - t)
    return rec, lat, res[:ns], nwt[:ns]


def level_entry(rec, lat, umax_running):
    return dict(L=rec['L'], lat=lat, tau=rec['tau'], steps_ok=rec['steps_ok'], umax=umax_running)


def solve_case(coh, j, kind):
    """DESIGN §4 / A1.5 / A2.1 procedure for one (case, kind). Returns (arrays, meta)."""
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
            f"ok={rec['steps_ok']} maxres={rec['max_res']:.2e} newton={rec['newton_total']} {rec['seconds']:.1f}s")
        return rec, lat

    def get(L, dt, purpose):
        rec, lat = run(L, dt, purpose)
        if not rec['steps_ok'] and purpose in ('prod', 'refine'):
            rec, lat = run(L, dt, 'retry')
        return level_entry(rec, lat, rec['umax_running']), rec['id']

    triple = [get(L, dt, 'prod') for L, dt in cfg['base']]
    history = []
    jd = V.judge([x[0] for x in triple], t0, coh.nu(j))
    history.append(dict(levels=[x[1] for x in triple], outcome=jd['outcome'], checks=jd['checks']))
    tight_used = False
    if jd['checks']['solver'] and not jd['checks']['floor']:
        triple = [get(L, dt, 'tight') for L, dt in cfg['base']]
        tight_used = True
        jd = V.judge([x[0] for x in triple], t0, coh.nu(j))
        history.append(dict(levels=[x[1] for x in triple], outcome=jd['outcome'], checks=jd['checks']))
    refined = False
    if jd['checks']['solver'] and jd['checks']['floor'] and not (jd['checks']['order'] and jd['checks']['peclet']):
        L, dt = cfg['refine']
        triple = triple[1:] + [get(L, dt, 'tight' if tight_used else 'refine')]
        refined = True
        jd = V.judge([x[0] for x in triple], t0, coh.nu(j))
        history.append(dict(levels=[x[1] for x in triple], outcome=jd['outcome'], checks=jd['checks']))
        if jd['checks']['solver'] and not jd['checks']['floor'] and not tight_used:
            base = {L_: dt_ for L_, dt_ in cfg['base']}
            base[cfg['refine'][0]] = cfg['refine'][1]
            triple = [get(e['L'], base[e['L']], 'tight') for e, _ in triple]
            tight_used = True
            jd = V.judge([x[0] for x in triple], t0, coh.nu(j))
            history.append(dict(levels=[x[1] for x in triple], outcome=jd['outcome'], checks=jd['checks']))
    final_ids = [x[1] for x in triple]
    arrays['t0'] = t0
    meta = dict(dim=d, cohort=coh.name, case=j, kind=kind, desc=coh.desc(j), cohort_sha256=coh.sha256,
                metric_id=V.METRIC_ID[d], lattice=RC.LATTICE[d], times=RC.TIMES.tolist(),
                t0_sha256=SI.sha(np.ascontiguousarray(t0).tobytes()), scheme=cfg['scheme'], attempts=attempts,
                final=final_ids, history=history, judge=jd, outcome=jd['outcome'], refined=refined,
                tight_used=tight_used)
    return arrays, meta


def job_info():
    return dict(job_id=os.environ.get('SLURM_JOB_ID'), commit=os.environ.get('SOURCE_COMMIT'),
                gpu=jax.devices()[0].device_kind, backend=jax.default_backend(), jax=jax.__version__,
                numpy=np.__version__, precision=os.environ.get('JAX_DEFAULT_MATMUL_PRECISION'))


def write_case(outdir, arrays, meta, sealed, key, manifest):
    stem = outdir / f"{meta['dim']}d" / meta['cohort'] / meta['kind'] / f"c{meta['case']:03d}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    npz = SI.npz_bytes(**arrays)
    js = json.dumps(V_clean(meta), indent=1).encode()
    if sealed:
        p1, c1 = SI.write_sealed(str(stem) + '.npz.enc', npz, key)
        p2, c2 = SI.write_sealed(str(stem) + '.json.enc', js, key)
        manifest.append(dict(file=str(stem.relative_to(outdir)) + '.npz.enc', plain=p1, cipher=c1))
        manifest.append(dict(file=str(stem.relative_to(outdir)) + '.json.enc', plain=p2, cipher=c2))
    else:
        tmp = Path(str(stem) + '.npz.tmp')
        tmp.write_bytes(npz)
        tmp.replace(str(stem) + '.npz')
        Path(str(stem) + '.json').write_bytes(js)
        manifest.append(dict(file=str(stem.relative_to(outdir)) + '.npz', sha256=SI.sha(npz)))


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


def qc_aggregate(metas, gpu_seconds):
    """Frozen QC schema for sealed / OOD cohorts (A1.11, A2.9): counts and quantiles only."""
    def q(vals):
        v = [x for x in vals if x is not None and np.isfinite(x)]
        return dict(n=len(v), min=min(v) if v else None, median=float(np.median(v)) if v else None,
                    max=max(v) if v else None)
    out = dict(cases=len(metas), outcomes={}, refined=0, tight=0, attempts=0, failed_attempts=0)
    for m in metas:
        out['outcomes'][m['outcome']] = out['outcomes'].get(m['outcome'], 0) + 1
        out['refined'] += int(m['refined'])
        out['tight'] += int(m['tight_used'])
        out['attempts'] += len(m['attempts'])
        out['failed_attempts'] += sum(not a['steps_ok'] for a in m['attempts'])
    jd = [m['judge'] for m in metas]
    out['p'] = q([p for x in jd for p in x['p']])
    out['p_undefined'] = sum(p is None for x in jd for p in x['p'])
    out['eps_ref'] = q([x['eps_ref'] for x in jd])
    out['peclet'] = q([x['peclet'] for x in jd])
    out['newton_total'] = q([float(sum(a['newton_total'] for a in m['attempts'])) for m in metas])
    out['gpu_seconds'] = gpu_seconds
    return out


def mode_fd(cfg, outdir, rep):
    d = int(cfg['dim'])
    sealed = CH.is_sealed(d, cfg['cohort'])
    key = cfg.get('key')
    seed = None
    if sealed:
        seeds = SI.read_seeds(cfg['seeds_file'], key, cfg['seeds_plain_sha256'])
        seed = int(seeds[f"{d}d/{cfg['cohort']}"])
    coh = CH.Cohort(d, cfg['cohort'], seed)
    cases = cfg.get('cases') or list(range(coh.n))
    rep['cohort'] = dict(name=coh.name, n=coh.n, sha256=None if sealed else coh.sha256, cases=cases)
    manifest, metas = [], []
    for j in cases:
        for kind in cfg['kinds']:
            stem = outdir / f'{d}d' / coh.name / kind / f'c{j:03d}'
            if Path(str(stem) + ('.json.enc' if sealed else '.json')).exists():
                log(f'skip existing {stem}')
                continue
            t = time.perf_counter()
            arrays, meta = solve_case(coh, j, kind)
            meta['job'] = job_info()
            meta['seconds'] = time.perf_counter() - t
            write_case(outdir, arrays, meta, sealed, key, manifest)
            metas.append(meta)
            jd = meta['judge']
            if not sealed:
                rep.setdefault('cases', []).append(dict(case=j, kind=kind, outcome=meta['outcome'], refined=meta['refined'],
                                                        tight=meta['tight_used'], eps_ref=jd['eps_ref'], p=jd['p'],
                                                        seconds=meta['seconds']))
                log(f"CASE {coh.name} {j} {kind}: {meta['outcome']} eps={jd['eps_ref']} p={jd['p']} {meta['seconds']:.0f}s")
            else:
                log(f'CASE sealed {kind} done ({meta["seconds"]:.0f}s)')
            save(outdir, rep)
    if sealed:
        mbytes = json.dumps(manifest).encode()
        p, c = SI.write_sealed(outdir / 'manifest.json.enc', mbytes, key)
        rep['sealed_manifest'] = dict(plain_sha256=p, cipher_sha256=c,
                                      ciphertexts=[dict(file=m['file'], cipher=m['cipher']) for m in manifest])
        rep['qc'] = qc_aggregate(metas, time.perf_counter() - T_BEGIN)
    else:
        rep['manifest'] = manifest


def mode_calib(cfg, outdir, rep):
    d = int(cfg['dim'])
    name, cases = CALIB_CASES[d]
    coh = CH.Cohort(d, name)
    rep['calib'], rep['d1'], rep['x1'], rep['x4'] = [], [], [], []
    store = {}
    for j in cases:
        for kind in ('ST', 'S'):
            cfgk = LEVELS[kind][d]
            for L, dt in cfgk['base']:
                lats, recs = {}, {}
                for purpose in ('prod', 'tight', 'tighter'):
                    rec, lat, res, nwt = attempt(coh, j, L, dt, cfgk['scheme'], purpose)
                    lats[purpose], recs[purpose] = lat, rec
                    store[f'{kind}_c{j}_L{L}_{purpose}'] = lat
                n0 = float(np.linalg.norm(RC.ic_lattice(coh, j).ravel()))
                diff = lambda a, b: float(np.max(V.tnorm(lats[a] - lats[b])[1:]) / n0)
                row = dict(case=j, kind=kind, L=L, dt=dt, prod_vs_tight=diff('prod', 'tight'),
                           tight_vs_tighter=diff('tight', 'tighter'),
                           tau={p: recs[p]['tau'] for p in recs}, max_res={p: recs[p]['max_res'] for p in recs},
                           steps_ok={p: recs[p]['steps_ok'] for p in recs}, seconds={p: recs[p]['seconds'] for p in recs},
                           newton_total={p: recs[p]['newton_total'] for p in recs})
                row['proxy_ok'] = bool(row['prod_vs_tight'] <= recs['prod']['tau'] and
                                       row['tight_vs_tighter'] <= recs['tight']['tau'])
                rep['calib'].append(row)
                log(f'CALIB {kind} c{j} L={L}: {row}')
                save(outdir, rep)
        # D1 (ST only): time-only ladder at the finest L, space-only ladder at the finest dt
        cfgk = LEVELS['ST'][d]
        (L1, dt1), (L2, dt2), (L3, dt3) = cfgk['base']
        n0 = float(np.linalg.norm(RC.ic_lattice(coh, j).ravel()))
        tl = [store.get(f'ST_c{j}_L{L3}_prod') if dt == dt3 else attempt(coh, j, L3, dt, 'cnr', 'prod')[1]
              for dt in (dt1, dt2, dt3)]
        sl = [attempt(coh, j, L, dt3, 'cnr', 'prod')[1] if L != L3 else store[f'ST_c{j}_L{L3}_prod']
              for L in (L1, L2, L3)]
        for name_, lad, hs in (('time', tl, (dt1, dt2, dt3)), ('space', sl, (1 / L1, 1 / L2, 1 / L3))):
            d12, d23 = V.tnorm(lad[0] - lad[1])[1:], V.tnorm(lad[1] - lad[2])[1:]
            rep['d1'].append(dict(case=j, ladder=name_, p=[V.solve_order(d12[k], d23[k], *hs) for k in range(5)],
                                  d23=(d23 / n0).tolist()))
        log(f"D1 c{j}: {rep['d1'][-2:]}")
        save(outdir, rep)
    # X1: upwind + BE ladder on every dev case
    for j in range(coh.n):
        lv = []
        for L, dt in X1_LEVELS[d]:
            rec, lat, res, nwt = attempt(coh, j, L, dt, 'be', 'prod', adv='upwind')
            lv.append(dict(L=L, lat=lat, tau=rec['tau'], steps_ok=rec['steps_ok'], umax=rec['umax_running']))
        jd = V.judge(lv, RC.ic_lattice(coh, j), coh.nu(j))
        rep['x1'].append(dict(case=j, p=jd['p'], outcome=jd['outcome']))
        log(f"X1 c{j}: p={jd['p']} {jd['outcome']}")
        save(outdir, rep)
    # X4 (2D only by design: dev6 case 0, ST, middle level dt doubled)
    if d == 2:
        cfgk = LEVELS['ST'][2]
        lv = []
        for i, (L, dt) in enumerate(cfgk['base']):
            rec, lat, res, nwt = attempt(coh, 0, L, dt * (2 if i == 1 else 1), 'cnr', 'prod')
            lv.append(dict(L=L, lat=lat, tau=rec['tau'], steps_ok=rec['steps_ok'], umax=rec['umax_running']))
        jd = V.judge(lv, RC.ic_lattice(coh, 0), coh.nu(0))
        rep['x4'] = dict(p=jd['p'], outcome=jd['outcome'])
        log(f"X4: {rep['x4']}")
    np.savez(outdir / f'calib_{d}d_fields.npz', **store)
    rep['calib_fields_sha256'] = SI.sha((outdir / f'calib_{d}d_fields.npz').read_bytes())


def ps_case(coh, j, Ns, dt, attempt_no, purpose_tag):
    d = coh.dim
    m = RC.LATTICE[d]
    runs = []
    for N in Ns:
        for k, (dtx, purpose) in enumerate([(dt, purpose_tag)] + ([(dt / 2, 'half')] if N == Ns[-1] else [])):
            t = time.perf_counter()
            b0 = PS.ic_coefs_2d(coh.D[j], N) if d == 2 else PS.ic_coefs_3d(coh.T, j, N)
            S = PS.lattice_S(N, m)
            if (N, d) not in _PS:
                _PS[(N, d)] = PS.make_ps(N, d)
            r = _PS[(N, d)](jnp.asarray(b0), coh.nu(j), dtx, S)
            kexp = int(round(np.log2(PS_CFG[d]['dt'] / dtx)))
            runs.append(dict(id=f'N{N}_dt{kexp}_a{attempt_no}', N=N, dt=dtx, attempt=attempt_no, purpose=purpose,
                             umax=r['umax'], finite=r['finite'], lat=r['lat'], seconds=time.perf_counter() - t))
            log(f"  PS {coh.name} c{j} {runs[-1]['id']} umax={r['umax']:.3f} finite={r['finite']} "
                f"{runs[-1]['seconds']:.0f}s")
    return runs


_PS = {}


def stability_ok(runs, d, dt, N3):
    umax = max(r['umax'] for r in runs)
    return dt <= 0.66 * 2.8 / (2 * d * np.pi * N3 * umax)


def mode_ps(cfg, outdir, rep):
    d = int(cfg['dim'])
    coh = CH.Cohort(d, cfg['cohort'])
    pc = PS_CFG[d]
    Ns = cfg.get('Ns') or pc['Ns']
    for j in cfg['cases']:
        stem = outdir / f'{d}d' / coh.name / 'PS' / f'c{j:03d}'
        if Path(str(stem) + '.json').exists():
            continue
        dt = pc['dt']
        runs = ps_case(coh, j, Ns, dt, 0, cfg.get('purpose', 'spatial'))
        stab = stability_ok(runs, d, dt, Ns[-1])
        if not stab:
            dt = dt / 2
            runs += ps_case(coh, j, Ns, dt, 1, 'stability')
            stab = stability_ok([r for r in runs if r['attempt'] == 1], d, dt, Ns[-1])
        sel_att = max(r['attempt'] for r in runs)
        sel = [r for r in runs if r['attempt'] == sel_att]
        spatial = [next(r['id'] for r in sel if r['N'] == N and r['dt'] == dt) for N in Ns]
        half = next(r['id'] for r in sel if r['N'] == Ns[-1] and r['dt'] == dt / 2)
        b0 = PS.ic_coefs_2d(coh.D[j], Ns[-1]) if d == 2 else PS.ic_coefs_3d(coh.T, j, Ns[-1])
        arrays = {r['id']: r['lat'] for r in runs}
        arrays['t0_ps'] = runs[-1]['lat'][0]
        arrays['t0'] = RC.ic_lattice(coh, j)
        meta = dict(dim=d, cohort=coh.name, case=j, desc=coh.desc(j), Ns=Ns, dt_base=pc['dt'],
                    runs=[{k: v for k, v in r.items() if k != 'lat'} for r in runs], selected_spatial=spatial,
                    selected_half=half, stability_ok=bool(stab), gates_ok=rep['gates_all_pass'],
                    finite=bool(all(r['finite'] for r in sel)), b0_norm=float(np.linalg.norm(b0)), job=job_info())
        stem.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(stem) + '.npz', **arrays)
        Path(str(stem) + '.json').write_text(json.dumps(V_clean(meta), indent=1))
        rep.setdefault('ps_cases', []).append(dict(case=j, stability_ok=bool(stab), finite=meta['finite']))
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
    rep = dict(config=cfg, job=job_info(), complete=False)
    if cfg.get('smoke_override'):            # LOCAL FUNCTIONAL SMOKE ONLY: reduced levels; never a reference
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
    # every mode: unit gates + G1 first
    pre = G.run_all('units')
    for dd in (2, 3):
        pre['gates'].append(G.g1_parity(dd))
    if cfg['mode'] == 'gates':
        full = G.run_all('all', ps_N={2: PS_CFG[2]['esc'], 3: PS_CFG[3]['esc']},
                         ps_dt={2: PS_CFG[2]['dt'], 3: PS_CFG[3]['dt']})
        pre['gates'] += full['gates']
    pre['all_pass'] = all(g['pass'] for g in pre['gates'])
    rep['gates'] = pre['gates']
    rep['gates_all_pass'] = pre['all_pass']
    save(outdir, rep)
    log(f"gates all_pass={pre['all_pass']}: " + ', '.join(f"{g['gate']}{g.get('d', '')}={g['pass']}" for g in pre['gates']))
    if cfg['mode'] == 'ps':
        # PS jobs also need G3/G4 in the same job (A2.4 step 2)
        dd = int(cfg['dim'])
        ps_g = G.run_all('ps', ps_N={dd: PS_CFG[dd]['Ns'][-1]}, ps_dt={dd: PS_CFG[dd]['dt']}, dims=(dd,), units=False)
        rep['gates'] += ps_g['gates']
        rep['gates_all_pass'] = all(g['pass'] for g in rep['gates'])
        save(outdir, rep)
    if not rep['gates_all_pass']:
        log('GATE FAILURE: refusing to continue')
        raise SystemExit(4)
    if cfg['mode'] == 'fd':
        mode_fd(cfg, outdir, rep)
    elif cfg['mode'] == 'calib':
        mode_calib(cfg, outdir, rep)
    elif cfg['mode'] == 'ps':
        mode_ps(cfg, outdir, rep)
    rep['complete'] = True
    rep['seconds'] = time.perf_counter() - T_BEGIN
    save(outdir, rep)
    print('JOB COMPLETE', flush=True)


if __name__ == '__main__':
    main()
