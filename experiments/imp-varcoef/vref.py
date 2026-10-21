"""imp-varcoef reference job (DESIGN §4, A1.5, A1.6, A2.5, A2.7, A2.19): the E0 recipe with variable kappa.

Per (contrast, case): three nested CN-R levels of vfd.vref_solver (E0 prod tolerances, one retry at E0's retry tolerances),
E0 verdict.judge (unchanged), conditional Richardson eps_ref; reference field = finest level on the 257^2 lattice.
Cross-check (A2.5) on the listed cases: four PSI SBDF2 runs (spatial / temporal / quadrature refinement) and the distance
to the reference. Reference-predicate controls (A2.19) on the first case with real levels.

    python vref.py --config configs/<job>.json --out <dir>

config: {"contrasts": [100, 10], "cases": [["dev", 0], ...], "levels": [[1024, 0.0025], [2048, 0.00125], [4096, 0.000625]],
         "crosscheck": [["dev", 0], ...], "controls": true}
Writes <dir>/ref_C<C>_<split>_<j>.npz (lat (6,257,257) finest, the three level lattices, res arrays) and <dir>/result.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'jcp-references'))
import vcoef as VC       # noqa: E402
import vfd as VF         # noqa: E402
import vspec as VS       # noqa: E402
import refcore as FC     # noqa: E402  (jcp-fom2d copy == jcp-references copy, byte-identical)
import cohorts as CH     # noqa: E402
import verdict as V      # noqa: E402  (jcp-references/verdict.py, E0 acceptance, unchanged)

TOL = dict(prod=(1e-11, 1e-9), retry=(1e-11, 1e-10))
NEWTON = dict(prod=20, retry=40)
XC_BAR_CONV = 2.5e-5
XC_BAR_DIST = 2.5e-4
ELIG_EPS = 1e-3


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, np.float64)).tobytes()).hexdigest()


def lat_dist(F, G, n0):
    return float(max(np.linalg.norm((np.asarray(F[k]) - np.asarray(G[k])).ravel()) for k in range(1, 6)) / n0)


class Solvers:
    def __init__(self):
        self.c = {}

    def get(self, L, purpose):
        k = (L, purpose)
        if k not in self.c:
            self.c[k] = VF.vref_solver(L, max_newton=NEWTON[purpose])
        return self.c[k]


def level(S, case, kp, L, dt):
    u0 = VF.fd_ic(case['desc'], L)
    ns = FC.n_stages(dt, 'cnr')
    recs = []
    for purpose in ('prod', 'retry'):
        run = S.get(L, purpose)
        t = time.perf_counter()
        r = run(u0, kp, dt, 'cnr', *TOL[purpose])
        r = jax.block_until_ready(r)
        secs = time.perf_counter() - t
        res = np.asarray(r['res'])
        lat = FC.to_lattice(r['out'], L, 2)
        ok = bool(V.steps_ok(res, ns) and int(r['stages']) == ns and np.all(np.isfinite(lat)))
        KX, KY = VF.faces(jnp.asarray(kp), L)
        rec = dict(L=L, dt=dt, purpose=purpose, stages=int(r['stages']), expected=ns, steps_ok=ok,
                   max_res=float(np.nanmax(res[:ns])), tau=V.floor_proxy(res, ns), umax=float(r['umax']),
                   newton_total=int(np.asarray(r['nwt'])[:ns].sum()), seconds=secs, compile_seconds=run.compile_seconds,
                   kmin_face=float(min(jnp.min(KX), jnp.min(KY))), kmax_face=float(max(jnp.max(KX), jnp.max(KY))))
        recs.append(rec)
        log(f"   L={L} dt={dt:.3e} {purpose} ok={ok} maxres={rec['max_res']:.2e} newton={rec['newton_total']} "
            f"{secs:.1f}s (compile {run.compile_seconds:.1f}s)")
        if ok:
            break
    return recs, lat, res


def crosscheck(case, kp, Lat_ref, n0, eps):
    runs = {}
    specs = {'base': (256, 0.05 / 128, 512), 'space': (192, 0.05 / 128, 384), 'time': (256, 0.05 / 256, 512),
             'quad': (256, 0.05 / 128, 768)}
    stats = {}
    for k, (N, dt, Q) in specs.items():
        t = time.perf_counter()
        q = VS.make_psi(N, dt, 'SBDF2', 1e-12, Q=Q)
        F, st = q(VS.ic(case['desc'], N), jnp.asarray(kp))
        F = np.asarray(jax.block_until_ready(F))
        runs[k] = F
        st = jax.device_get(st)
        stats[k] = dict(N=N, dt=dt, Q=Q, nbad=int(st['nbad']), it_max=int(st['it_max']), res_max=float(st['res_max']),
                        seconds=time.perf_counter() - t, finite=bool(np.all(np.isfinite(F))))
    b = runs['base']
    diffs = {k: lat_dist(runs[k], b, n0) for k in ('space', 'time', 'quad')}
    conv = all(v <= XC_BAR_CONV for v in diffs.values()) and all(s['nbad'] == 0 and s['finite'] for s in stats.values())
    dist = lat_dist(b, Lat_ref, n0)
    bar = max(XC_BAR_DIST, 3 * eps) if eps is not None else XC_BAR_DIST
    return dict(stats=stats, diffs=diffs, converged=bool(conv), distance=dist, bar=bar,
                passed=bool(conv and dist <= bar))


def ref_controls(levels, t0, nu):
    """A2.19 reference-predicate fixtures on real levels: each must give its own outcome."""
    u1, u2, u3 = (np.asarray(l['lat']) for l in levels)
    base = [dict(l) for l in levels]
    out = {}

    def J(lv, nu_=nu, t0_=t0):
        return V.judge(lv, t0_, nu_)['outcome']
    out['baseline'] = J(base)
    lv = [dict(l) for l in base]
    lv[1]['steps_ok'] = False
    out['T-ref-solver'] = dict(outcome=J(lv), expect='solver-fail')
    lv = [dict(l) for l in base]
    lv[0]['tau'] = 1.0
    out['T-ref-floor'] = dict(outcome=J(lv), expect='unresolved-floor')
    out['T-ref-peclet'] = dict(outcome=J(base, 1e-9), expect='peclet-fail')
    lv = [dict(l) for l in base]
    lv[2]['lat'] = 2 * u2 - u1
    out['T-ref-order'] = dict(outcome=J(lv), expect='order-fail')
    w = np.array(u3, copy=True)
    rng = np.random.default_rng(1)
    for k in range(1, 6):
        e = (u1[k] - u2[k]).ravel()
        v = rng.standard_normal(e.size)
        v -= v @ e / (e @ e) * e
        v /= np.linalg.norm(v)
        w[k] = u2[k] - 0.25 * np.linalg.norm(e) * v.reshape(u2[k].shape)
    lv = [dict(l) for l in base]
    lv[2]['lat'] = w
    out['T-ref-direction'] = dict(outcome=J(lv), expect='direction-inconsistent')
    w = np.array(u3, copy=True)
    w[0] = w[0] * (1 + 1e-6)
    lv = [dict(l) for l in base]
    lv[2]['lat'] = w
    out['T-ref-ic'] = dict(outcome=J(lv), expect='ic-mismatch')
    for k, v in out.items():
        if k != 'baseline':
            v['fires'] = v['outcome'] == v['expect']
    out['all_fire'] = all(v['fires'] for k, v in out.items() if k not in ('baseline',))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = dict(config=cfg, backend=jax.default_backend(), devices=[str(d) for d in jax.devices()],
               jax_version=jax.__version__, cases={}, crosscheck={}, controls=None)
    cohorts = {s: VC.cohort(s) for s in {c[0] for c in cfg['cases']}}
    S = Solvers()
    T0 = time.perf_counter()
    for C in cfg['contrasts']:
        for split, j in cfg['cases']:
            case = cohorts[split][j]
            kp = VC.kappa_params(case, float(C))
            cid = f'C{C:g}|{split}|{j}'
            log(f'CASE {cid} desc={np.round(case["desc"], 4).tolist()}')
            t0 = np.asarray(CH.ic_2d(256, case['desc']))
            levels, recs = [], []
            for L, dt in cfg['levels']:
                r, lat, res = level(S, case, kp, int(L), float(dt))
                recs.append(r)
                levels.append(dict(L=int(L), lat=lat, tau=r[-1]['tau'], steps_ok=r[-1]['steps_ok'], umax=r[-1]['umax']))
            nu_pe = recs[0][-1]['kmin_face']
            jd = V.judge(levels, t0, nu_pe)
            eps = jd['eps_ref']
            elig = bool(jd['outcome'] == 'accepted' and eps is not None and eps <= ELIG_EPS)
            np.savez_compressed(out / f'ref_C{C:g}_{split}_{j}.npz', lat=levels[-1]['lat'], t0=t0,
                                **{f'lat_L{l["L"]}': l['lat'] for l in levels})
            rec = dict(desc=case['desc'].tolist(), xi=case['xi'].tolist(), gmin=case['gmin'], gmax=case['gmax'],
                       realised=VC.realised(case, float(C)), levels=recs, judge={k: v for k, v in jd.items()},
                       eps_ref=eps, eligible_pre_crosscheck=elig, ref_sha256=sha(levels[-1]['lat']),
                       n0=float(np.linalg.norm(t0.ravel())))
            rep['cases'][cid] = rec
            log(f'  -> {jd["outcome"]} p={np.round([p if p is not None else -1 for p in jd["p"]], 3).tolist()} '
                f'eps={eps} eligible={elig} elapsed={time.perf_counter() - T0:.0f}s')
            if [split, j] in cfg.get('crosscheck', []):
                xc = crosscheck(case, kp, levels[-1]['lat'], rec['n0'], eps)
                rep['crosscheck'][cid] = xc
                log(f'  XC conv={xc["converged"]} diffs={xc["diffs"]} dist={xc["distance"]:.2e} passed={xc["passed"]}')
            if cfg.get('controls') and rep['controls'] is None:
                rep['controls'] = ref_controls(levels, t0, nu_pe)
                log(f'  CONTROLS {rep["controls"]}')
            (out / 'result.json').write_text(json.dumps(rep, indent=1, default=float))
    rep['elapsed_seconds'] = time.perf_counter() - T0
    rep['complete'] = True
    (out / 'result.json').write_text(json.dumps(rep, indent=1, default=float))
    (out / 'COMPLETE').write_text('complete\n')


if __name__ == '__main__':
    main()
