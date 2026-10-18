"""jcp-hr-baselines phase-II tuning job (Stage A) for one cell (DESIGN.md 3, 6, A1.3-A1.8, A2.2-A2.8, A15-A23, A24).

Builds, from TRAINING draws only: full snapshot set, POD (certified), trust radii, in-cell adequacy check, (2D 256^2:
the A1.3 12-step-vs-all-step saturation check), the generator probe basis, advection POD + Q-DEIM data, ECSW training
states and one support-targeted NNLS path per rank (through 16r, A16/A20/A22). Then Stage A on the 6 screening cases:
every registered configuration (rank x {none, ecsw k2/k4/k8, qdeim p/r 1/2/4} x option x {native, common}) with
in-job certification, warm screening time (burned), E0 score, tight (native) anchor on all 6 cases and tighter on the
first 2 (A8.2); the generator-probe `none`/native runs; the full-rank limiting comparator (FOM) on the verdict cohort.
Writes progressively: <out>/tune.json (summaries) and <out>/arrays/*.npz (basis, HR online data per configuration,
lattice fields and trajectories). Everything later jobs need is in <out>/artifacts/ (sha256-listed).

Usage: python tune.py <config.json> <out_dir>
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'e0'))
import hrcore as H  # noqa: E402
import rom as RM  # noqa: E402
import build as BD  # noqa: E402
import pipeline as PL  # noqa: E402
import score as SC  # noqa: E402

T0 = time.perf_counter()
SCREEN = list(range(6))                         # 2D dev6 0-5; 3D 923801 0-5 (selection subset = timing cases)
HR_GRID = (('none', '-'), ('ecsw', 'k2'), ('ecsw', 'k4'), ('ecsw', 'k8'), ('qdeim', 1), ('qdeim', 2), ('qdeim', 4))


def log(*a):
    print(f'[{time.perf_counter() - T0:8.1f}s]', *a, flush=True)


class Out:
    def __init__(self, d):
        self.d = Path(d)
        (self.d / 'arrays').mkdir(parents=True, exist_ok=True)
        (self.d / 'artifacts').mkdir(parents=True, exist_ok=True)
        self.res = dict(phases={})

    def save(self):
        tmp = self.d / 'tune.json.tmp'
        PL.dump(tmp, self.res)
        os.replace(tmp, self.d / 'tune.json')

    def npz(self, sub, name, **kw):
        fn = self.d / sub / f"{name.replace('|', '~')}.npz"
        tmp = fn.with_suffix('.tmp.npz')
        np.savez(tmp, **kw)
        os.replace(tmp, fn)

    def phase(self, name, fn, store=None):
        t0 = time.perf_counter()
        try:
            v = fn()
            self.res['phases'][name] = dict(status='ok', seconds=time.perf_counter() - t0, device_peak=PL.device_peak(),
                                            host_maxrss=PL.host_rss())
            log(f'OK   {name} ({time.perf_counter() - t0:.1f}s)')
        except Exception as e:  # noqa: BLE001
            self.res['phases'][name] = dict(status='error', error=f'{type(e).__name__}: {e}',
                                            trace=traceback.format_exc()[-3000:], seconds=time.perf_counter() - t0)
            log(f'ERR  {name}: {type(e).__name__}: {e}')
            v = None
        if store is not None:
            self.res[store] = v
        self.save()
        return v


def hr_points(kind, r, hrp, cell):
    if kind == 'none':
        return 0
    if kind == 'ecsw':
        rec = cell.ecsw_rec.get(r, {}).get(hrp)
        return None if rec is None else int(rec['size'])
    return int(hrp * r)


def main():
    cfg = json.loads(Path(sys.argv[1]).read_text())
    O = Out(sys.argv[2])
    if not cfg.get('allow_cpu'):
        assert jax.default_backend() == 'gpu', jax.default_backend()
    assert os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest'
    log(f'jax_backend={jax.default_backend()} devices={jax.devices()}')
    O.res.update(config=cfg, env=dict(backend=jax.default_backend(), devices=[str(x) for x in jax.devices()],
                                      job=os.environ.get('SLURM_JOB_ID'), commit=cfg.get('commit')))
    d, L = cfg['d'], cfg['L']
    ranks, rmax = cfg['ranks'], max(cfg['ranks'])
    options = [tuple(o) for o in cfg['options']]
    coh = 'dev6' if d == 2 else 'val64'
    fam_tr = H.Family(d, 'train')
    fam = H.Family(d, coh, count=None if d == 2 else 64)
    verdict_cases = list(range(fam.n)) if d == 3 else None
    cell = PL.Cell(d, L, log=log)
    refs = O.phase('refs', lambda: SC.Refs(d, coh, fam, SCREEN))
    if refs is None:
        return
    O.res['n0'] = {j: refs.n0(j) for j in refs.f}
    n0 = O.res['n0']

    # ---- offline construction (training only) ----------------------------------------------------------------
    rows = list(range(fam_tr.n if cfg.get('n_traj') is None else cfg['n_traj']))
    bb = O.phase('basis', lambda: cell.build_basis(fam_tr, rows, rmax, ranks))
    if bb is None:
        return
    S, Phi, sig, trust = bb
    bb = None
    cell.set_basis(Phi, trust)
    O.res['basis'] = cell.info['basis[central]']
    O.npz('artifacts', 'Phi', Phi=np.asarray(Phi), trust=np.asarray([trust.get(r, np.nan) for r in ranks]),
          ranks=np.asarray(ranks))

    def adequacy():
        held = (3, 5, 6, 8, 12, 17, 22, 35, 45)
        rr = list(range(32))
        Sh, _ = BD.snapshots(d, L, fam_tr, rr, save_steps=held, log=None)
        Si = S[:len(rr) * len(H.SNAP_STEPS)]
        out = {}
        for r in ranks:
            Pr = np.asarray(Phi[:, :r])
            e = lambda X: float(np.sqrt(np.mean(np.sum((X - (X @ Pr) @ Pr.T) ** 2, 1) / np.sum(X * X, 1))))
            out[r] = dict(held=e(Sh), inset=e(Si))
            out[r]['ratio'] = out[r]['held'] / max(out[r]['inset'], 1e-300)
            out[r]['adequate'] = bool(out[r]['ratio'] <= 1.10)
        return out
    O.phase('adequacy', adequacy, store='adequacy')

    if cfg.get('saturation'):                          # A1.3: 2D 256^2 only
        def saturation():
            Sall, _ = BD.snapshots(d, L, fam_tr, rows, save_steps=tuple(range(51)), log=None)
            Pa, sa, _ = BD.pod(Sall, rmax)
            out = {}
            for r in ranks:
                for tag, P_ in (('12step', np.asarray(Phi[:, :r])), ('allstep', np.asarray(Pa[:, :r]))):
                    C = BD.project_codes(Sall, jnp.asarray(P_))
                    n2 = np.sum(Sall * Sall, 1)
                    e2 = np.maximum(n2 - np.sum(np.asarray(C) ** 2, 1), 0.)
                    out.setdefault(r, {})[tag] = float(np.sqrt(np.mean(e2 / n2)))
                out[r]['adequate'] = bool(out[r]['12step'] <= 1.10 * out[r]['allstep'])
            del Sall
            return out
        O.phase('saturation', saturation, store='saturation')

    def probe_basis():                                 # A2.3 generator probe: the banks' generator (upwind BE)
        cp = PL.Cell(d, L, log=log)
        Su, Pu, su, tu = cp.build_basis(fam_tr, rows, rmax, ranks, gen='upwind')
        del Su
        O.npz('artifacts', 'Phi_probe', Phi=np.asarray(Pu), trust=np.asarray([tu.get(r, np.nan) for r in ranks]))
        return dict(info=cp.info['basis[upwind]'], trust=tu)
    probe = O.phase('generator_probe_basis', probe_basis, store='probe_basis')

    pmax = 4 * rmax
    p_list = sorted({m * r for r in ranks for m in (1, 2, 4)})
    O.phase('adv_pod', lambda: cell.build_adv(S, pmax, p_list))
    del S
    O.res['adv'] = {k: v for k, v in cell.info.get('adv', {}).items() if k != 'sigma'}
    O.phase('ecsw_states', lambda: cell.ecsw_states(fam_tr), store='ecsw_states')
    for r in ranks:
        def ecsw_r(r=r):
            rec = cell.ecsw(r)
            kw = {}
            for att, cps in cell.ecsw_paths_raw[r].items():
                kw[f'{att}~drops'] = np.asarray(cps['_drops'])
                kw[f'{att}~bnorm'] = np.asarray(cps['_bnorm'])
                kw[f'{att}~terminal~nodes'] = np.asarray(cps['_terminal']['nodes'])
                kw[f'{att}~terminal~weights'] = np.asarray(cps['_terminal']['weights'])
                kw[f'{att}~terminal~resvec'] = np.asarray(cps['_terminal']['resvec'])
                for k, v in cps.items():
                    if not k.startswith('_') and v is not None:
                        kw[f'{att}~{k}~nodes'] = np.asarray(v['nodes'])
                        kw[f'{att}~{k}~weights'] = np.asarray(v['weights'])
                        kw[f'{att}~{k}~resvec'] = np.asarray(v['resvec'])
            O.npz('artifacts', f'ECSW_r{r}', **kw)
            return cell.info['ecsw'][r]['paths']
        O.phase(f'ecsw_r{r}', ecsw_r, store=f'ecsw_r{r}')

    # ---- Stage A ------------------------------------------------------------------------------------------------
    O.res.setdefault('stageA', {})
    O.res.setdefault('construction_failures', {})
    for r in ranks:
        for kind, hrp in HR_GRID:
            hp = hrp if kind != 'qdeim' else int(hrp * r)
            try:
                dat, ainfo = cell.arm_data(kind, r, hp)
            except Exception as e:  # noqa: BLE001
                O.res['construction_failures'][f'{kind}|{r}|{hp}'] = f'{type(e).__name__}: {e}'
                O.save()
                continue
            if kind == 'ecsw':
                O.npz('artifacts', f'HR~ecsw~{r}~{hp}', nodes=np.asarray(cell.ecsw_rec[r][hp]['nodes']),
                      weights=np.asarray(cell.ecsw_rec[r][hp]['weights']))
            elif kind == 'qdeim':
                O.npz('artifacts', f'HR~qdeim~{r}~{hp}', P=np.asarray(cell.piv[hp]), T1=np.asarray(dat['T1']),
                      T2=np.asarray(dat['T2']), T3E=np.asarray(dat['T3E']))
            for option in options:
                def cell_block(kind=kind, hp=hp, dat=dat, option=option, r=r, ainfo=ainfo):
                    tight = {j: cell.run(kind, r, dat, 'native', option, 'tight', fam.ic(L, j), fam.nu(j)) for j in SCREEN}
                    tighter = {j: cell.run(kind, r, dat, 'native', option, 'tighter', fam.ic(L, j), fam.nu(j))
                               for j in SCREEN[:2]}
                    out = {}
                    for solver in ('native', 'common'):
                        cid = PL.cfg_id(kind, r, hp, option[0], option[1], solver)
                        recs = {}
                        for j in SCREEN:
                            recs[j] = cell.run(kind, r, dat, solver, option, 'prod', fam.ic(L, j), fam.nu(j))
                            warm = cell.run(kind, r, dat, solver, option, 'prod', fam.ic(L, j), fam.nu(j), certify=False,
                                            burn_s=0.1)
                            recs[j]['seconds_warm'] = warm['seconds']
                            recs[j]['timed_vs_scored'] = float(np.max(np.abs(warm['fields_lat'] - recs[j]['fields_lat'])))
                        q1 = PL.q1_config(recs, tight, tighter, {j: n0[j] for j in SCREEN}, SCREEN[:2], SCREEN)
                        e0 = {j: refs.score_safe(recs[j]['fields_lat'], j) for j in SCREEN}
                        O.npz('arrays', f'A~{cid}', W=np.stack([recs[j]['W'] for j in SCREEN]),
                              F=np.stack([recs[j]['fields_lat'] for j in SCREEN]),
                              Ftight=np.stack([tight[j]['fields_lat'] for j in SCREEN]),
                              Ftighter=np.stack([tighter[j]['fields_lat'] for j in SCREEN[:2]]),
                              C=np.stack([recs[j]['C'] for j in SCREEN]))
                        es = [e0[j]['e'] for j in SCREEN]
                        out[cid] = dict(q1=q1, e0=e0, hr_points=hr_points(kind, r, hp, cell), arm=ainfo,
                                        W=float(np.max(es)) if all(np.isfinite(es)) else float('nan'),
                                        med=float(np.median(es)) if all(np.isfinite(es)) else float('nan'),
                                        t=float(np.median([recs[j]['seconds_warm'] for j in SCREEN])),
                                        runs={j: dict(cert=recs[j]['cert'], seconds_warm=recs[j]['seconds_warm'],
                                                      timed_vs_scored=recs[j]['timed_vs_scored']) for j in SCREEN},
                                        anchors=dict(tight={j: tight[j]['cert'] for j in SCREEN},
                                                     tighter={j: tighter[j]['cert'] for j in SCREEN[:2]}))
                        log(f'   {cid}: q1={q1["q1"]} W={out[cid]["W"]:.4g} t={out[cid]["t"]:.4g}s')
                    return out
                res = O.phase(f'A|{kind}|{r}|{hp}|{option[0]}|{option[1]:.8g}', cell_block)
                if res:
                    O.res['stageA'].update(res)
                    O.save()
            del dat

    # ---- ours: batch-1 comparators, same-job warm timing on the 6 timing cases (A4.8, A5.3) and replay (G-ours) ----
    def ours_pass():
        import ours as OU
        out = {}
        if d == 2:
            ox = OU.OursP1(L, cfg['ours'], log=log)
            items = [(k, ox.fn(k)) for k in cfg['ours']]
        else:
            items = []
            oxs = {}
            for spec in cfg['ours']:
                bank, Rp, rule, a = spec['bank'], spec['Rp'], spec['rule'], spec['a']
                kk = f'{bank}|{Rp}|{rule}'
                if kk not in oxs:
                    oxs[kk] = OU.Ours3D(bank, Rp, rule, a=a, log=log)
                items.append((spec['id'], (oxs[kk], oxs[kk].fn(tuple(spec['spec'])))))
        for key, f in items:
            if d == 2:
                fx, lat, u0f = f, ox.lattice, (lambda j: ox.u0(fam, j))
            else:
                o3, fx = f
                lat, u0f = o3.lattice, (lambda j, o3=o3: o3.u0(fam, j))
            rec = {}
            for j in SCREEN:
                u, v = u0f(j), jnp.float64(fam.nu(j))
                o = fx(u, v)
                jax.block_until_ready(o['fields'])               # warm
                PL.burn(0.1)
                t0 = time.perf_counter()
                o = fx(u, v)
                jax.block_until_ready(o['fields'])
                ts = time.perf_counter() - t0
                Fl = lat(o)
                rec[j] = dict(seconds_warm=ts, e0=refs.score_safe(Fl, j)['e'])
                O.npz('arrays', f'X~{key}~{j}', F=Fl)
            out[key] = rec
            log(f'   ours {key}: ' + ' '.join(f"{j}:{rec[j]['seconds_warm']:.4f}s" for j in SCREEN))
        return out
    if cfg.get('ours'):
        O.phase('ours_pass', ours_pass, store='ours_pass')

    # ---- generator probe runs (A2.3): `none` native on the screening cases with the upwind-generated basis -------
    if probe is not None:
        def probe_runs():
            Pu = jnp.asarray(np.load(O.d / 'artifacts' / 'Phi_probe.npz')['Phi'])
            out = {}
            for r in ranks:
                datp = dict(Phi=Pu[:, :r], LPhi=H.lap_cols(Pu[:, :r], d, L), DPhi=H.dsum_cols(Pu[:, :r], d, L))
                for option in options:
                    cid = PL.cfg_id('none', r, '-', option[0], option[1], 'native', gen='upwind')
                    es = {j: refs.score_safe(cell.run('none', r, datp, 'native', option, 'prod', fam.ic(L, j),
                                                      fam.nu(j), certify=False)['fields_lat'], j)['e'] for j in SCREEN}
                    out[cid] = es
            return out
        O.phase('generator_probe_runs', probe_runs, store='probe_runs')

    # ---- full-rank limiting comparator (FOM) on the verdict cohort ------------------------------------------------
    def fom_runs():
        fom = H.make_fom(d, L)
        famv = fam if d == 3 else None
        out = {}
        cohorts = [('dev6', H.Family(2, 'dev6')), ('val32', H.Family(2, 'val32'))] if d == 2 else [('val64', fam)]
        for cname, f_ in cohorts:
            rf = SC.Refs(d, cname, f_, list(range(f_.n)))
            for option in options:
                for j in range(f_.n):
                    o = fom(f_.ic(L, j), f_.nu(j), H.sched(*option), 1e-10, 1e-12)
                    Fl = cell.lattice(o['out'])
                    out[f'{cname}|{j}|{option[0]}|{option[1]:.8g}'] = rf.score_safe(Fl, j)['e']
        return out
    O.phase('fom_limiting', fom_runs, store='fom_limiting')
    O.res['timers'] = cell.T.t
    O.res['total_seconds'] = time.perf_counter() - T0
    O.save()
    log('done')


if __name__ == '__main__':
    main()
