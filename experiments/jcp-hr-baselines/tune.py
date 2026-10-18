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
    global SCREEN
    cfg = json.loads(Path(sys.argv[1]).read_text())
    if cfg.get('dry'):                                  # local CPU dry run only (never in a cluster config)
        SCREEN = [0, 1, 2]
    O = Out(sys.argv[2])
    if not cfg.get('allow_cpu'):
        assert jax.default_backend() == 'gpu', jax.default_backend()
    assert os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest'
    log(f'jax_backend={jax.default_backend()} devices={jax.devices()}')
    O.res.update(config=cfg, env=dict(backend=jax.default_backend(), devices=[str(x) for x in jax.devices()],
                                      job=os.environ.get('SLURM_JOB_ID'), commit=cfg.get('commit')))
    d, L = cfg['d'], cfg['L']
    ranks = cfg['ranks']
    allr = ranks + ([cfg['ext_rank']] if cfg.get('ext_rank') else [])
    rmax = max(allr)                                                       # basis to the extension rung (A2.18)
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
    bb = O.phase('basis', lambda: cell.build_basis(fam_tr, rows, rmax, allr))
    if bb is None:
        return
    S, Phi, sig, trust = bb
    bb = None
    cell.set_basis(Phi, trust)
    O.res['basis'] = cell.info['basis[central]']
    if d == 2:                                          # 3D: the lattice is every interior node (Phi itself)
        O.npz('artifacts', 'Phi_lat', Phi_lat=np.asarray(Phi)[H.lattice_nodes(d, L)], nodes=H.lattice_nodes(d, L))
    O.npz('artifacts', 'Phi', Phi=np.asarray(Phi), trust=np.asarray([trust.get(r, np.nan) for r in allr]),
          ranks=np.asarray(allr))

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


    pmax = 4 * rmax
    p_list = sorted({m * r for r in allr for m in (1, 2, 4)})
    O.phase('adv_pod', lambda: cell.build_adv(S, pmax, p_list))
    del S
    O.res['adv'] = {k: v for k, v in cell.info.get('adv', {}).items() if k != 'sigma'}
    # generator probe after the central snapshots are released (host memory: one snapshot matrix at a time)
    PROBE = {}

    def probe_basis():                                 # A2.3 generator probe: the banks' generator (upwind BE)
        cp = PL.Cell(d, L, log=log)
        Su, Pu, su, tu = cp.build_basis(fam_tr, rows, rmax, ranks, gen='upwind')
        del Su
        PROBE['Phi'] = np.asarray(Pu)                  # host copy for the probe runs (not an artifact: disk)
        del Pu
        return dict(info=cp.info['basis[upwind]'], trust=tu)
    probe = O.phase('generator_probe_basis', probe_basis, store='probe_basis')
    O.phase('ecsw_states', lambda: cell.ecsw_states(fam_tr), store='ecsw_states')
    O.res.setdefault('construction_failures', {})
    for r in ranks:
        def ecsw_r(r=r):
            if d == 3 and r == max(ranks):                 # A22.2 / A24.4: device peak before the maximal path
                O.res.setdefault('memory_admission', {})['before'] = PL.device_peak()
            rec = cell.ecsw(r)
            if d == 3 and r == max(ranks):
                pk = PL.device_peak()
                bf = O.res['memory_admission']['before']   # process-cumulative peaks: the path set a new peak iff after > before
                O.res['memory_admission'].update(after=pk, limit=120e9, path_set_peak=bool(pk > bf),
                                                 admitted=bool(0 <= pk and (pk <= bf or pk <= 120e9)))
                if not O.res['memory_admission']['admitted']:
                    cell.ecsw_rec.pop(r, None)             # rank r: "not constructible (memory)"
                    O.res['construction_failures'][f'ecsw|{r}|*'] = f'not constructible (memory): peak {pk} B > 120 GB'
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
    search_t0 = time.perf_counter()

    def screen(r, kind, hrp, tag='A'):
        """One (rank, HR kind, knob): construction data, then every option x {native, common} on SCREEN with Q1."""
        hp = hrp if kind != 'qdeim' else int(hrp * r)
        try:
            dat, ainfo = cell.arm_data(kind, r, hp)
        except Exception as e:  # noqa: BLE001
            O.res['construction_failures'][f'{kind}|{r}|{hp}'] = f'{type(e).__name__}: {e}'
            O.save()
            return
        quality = cell.info['ecsw'][r]['paths']['quality'] if kind == 'ecsw' else None
        if kind == 'ecsw':
            O.npz('artifacts', f'HR~ecsw~{r}~{hp}', nodes=np.asarray(cell.ecsw_rec[r][hp]['nodes']),
                  weights=np.asarray(cell.ecsw_rec[r][hp]['weights']))
        elif kind == 'qdeim':
            O.npz('artifacts', f'HR~qdeim~{r}~{hp}', P=np.asarray(cell.piv[hp]), T1=np.asarray(dat['T1']),
                  T2=np.asarray(dat['T2']), T3E=np.asarray(dat['T3E']))
        for option in options:
            def cell_block(option=option):
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
                    # output coefficients only: every arm's fields are W @ Phi_r^T (rom.make_query), so lattice fields
                    # are recomputed offline from artifacts/Phi_lat (disk: 3D fields would be ~70 GB per cell)
                    O.npz('arrays', f'A~{cid}', W=np.stack([recs[j]['W'] for j in SCREEN]),
                          Wtight=np.stack([tight[j]['W'] for j in SCREEN]),
                          Wtighter=np.stack([tighter[j]['W'] for j in SCREEN[:2]]),
                          C=np.stack([recs[j]['C'] for j in SCREEN]),
                          gamma=np.stack([recs[j]['gamma'] for j in SCREEN]))
                    es = [e0[j]['e'] for j in SCREEN]
                    out[cid] = dict(q1=q1, e0=e0, hr_points=hr_points(kind, r, hp, cell), arm=ainfo, quality=quality,
                                    stage=tag, kind=kind, r=r, hrp=hp, scheme=option[0], dt=option[1], solver=solver,
                                    W=float(np.max(es)) if all(np.isfinite(es)) else float('nan'),
                                    med=float(np.median(es)) if all(np.isfinite(es)) else float('nan'),
                                    t=float(np.median([recs[j]['seconds_warm'] for j in SCREEN])),
                                    runs={j: dict(cert=recs[j]['cert'], seconds_warm=recs[j]['seconds_warm'],
                                                  timed_vs_scored=recs[j]['timed_vs_scored']) for j in SCREEN},
                                    anchors=dict(tight={j: tight[j]['cert'] for j in SCREEN},
                                                 tighter={j: tighter[j]['cert'] for j in SCREEN[:2]}))
                    log(f'   {cid}: q1={q1["q1"]} W={out[cid]["W"]:.4g} t={out[cid]["t"]:.4g}s')
                return out
            res = O.phase(f'{tag}|{kind}|{r}|{hp}|{option[0]}|{option[1]:.8g}', cell_block)
            if res:
                O.res['stageA'].update(res)
                O.save()

    for r in ranks:
        for kind, hrp in HR_GRID:
            screen(r, kind, hrp)

    # ---- A2.18 / A16.2 bounded extension (one rung per direction, while the C-search allocation remains) -------------
    def extension():
        import stats as ST
        used_h = (time.perf_counter() - search_t0) / 3600
        alloc = cfg.get('search_allocation_hours')
        items = [dict(id=k, t=v['t'], W=v['W']) for k, v in O.res['stageA'].items()
                 if v['q1']['q1'] and np.isfinite(v['W']) and np.isfinite(v['t'])]
        F = ST.front(items)
        todo = set()
        for f in F:
            v = O.res['stageA'][f['id']]
            if v['kind'] == 'ecsw' and v['hrp'] == 'k8':
                todo.add((v['r'], 'ecsw', 'k16'))
            if cfg.get('ext_rank') and v['r'] == max(ranks):
                todo.add((cfg['ext_rank'], v['kind'], v['hrp'] if v['kind'] != 'qdeim' else v['hrp'] // v['r']))
        rec = dict(front=[f['id'] for f in F], triggered=sorted(map(str, todo)), search_hours_used=used_h,
                   allocation_hours=alloc, run=bool(todo) and (alloc is None or used_h < alloc))
        O.res['extension'] = rec
        O.save()
        if not rec['run']:
            return rec
        for r, kind, hrp in sorted(todo, key=str):
            if kind == 'ecsw' and r not in cell.ecsw_rec:
                O.phase(f'ecsw_r{r}', lambda r=r: ecsw_r(r), store=f'ecsw_r{r}')
            screen(r, kind, hrp, tag='EXT')
        return rec
    O.phase('extension', extension)
    O.res['search_hours'] = (time.perf_counter() - search_t0) / 3600
    O.save()

    # ---- ours: batch-1 comparators, same-job warm timing on the 6 timing cases (A4.8, A5.3) and replay (G-ours) ----
    def ours_pass():
        import ours as OU
        out = {}
        if d == 2:
            exp = json.loads((HERE / 'inputs' / 'p1_analysis.json').read_text())['configurations']
            ref_e = {k: {j: exp[k]['e0_per_case'][f'dev6|{j}'] for j in SCREEN} for k in cfg['ours']}
            ox = OU.OursP1(L, cfg['ours'], log=log)
            items = [(k, ox, ox.fn(k)) for k in cfg['ours']]
        else:
            exp = json.loads((HERE / 'inputs' / 'speed3d_batch1_e0.json').read_text())['configs']
            ref_e = {k: {j: exp[k]['e0'][str(j)] for j in SCREEN} for k in cfg['ours']}
            items, oxs = [], {}
            for k in cfg['ours']:                          # '<setting>|<rule>|fs|<scheme>|<dt>|K<k>' (speed3d keys)
                st, rule, kind, scheme, dt, K = k.split('|')
                assert kind == 'fs', k
                bank, Rp, a = OU.S3_SETTINGS[st]
                if (st, rule) not in oxs:
                    oxs[(st, rule)] = OU.Ours3D(bank, Rp, rule, a=a, log=log)
                o3 = oxs[(st, rule)]
                items.append((k, o3, o3.fn(('fs', scheme, float(dt), int(K[1:])))))
        for key, ox_, fx in items:
            rec = {}
            for j in SCREEN:
                u, v = ox_.u0(fam, j), jnp.float64(fam.nu(j))
                o = fx(u, v)
                jax.block_until_ready(o['fields'])               # warm
                PL.burn(0.1)
                t0 = time.perf_counter()
                o = fx(u, v)
                jax.block_until_ready(o['fields'])
                ts = time.perf_counter() - t0
                Fl = ox_.lattice(o)
                e = refs.score_safe(Fl, j)['e']
                rel = abs(e - ref_e[key][j]) / abs(ref_e[key][j]) if np.isfinite(e) else float('inf')
                rec[j] = dict(seconds_warm=ts, e0=e, e0_exported=ref_e[key][j], replay_rel=rel)
                O.npz('arrays', f'X~{key}~{j}', F=Fl)
            worst = max(rec[j]['replay_rel'] for j in SCREEN)
            out[key] = dict(cases=rec, replay_worst_rel=worst, G_ours=bool(worst <= 1e-10))   # DESIGN 8: G-ours
            log(f'   ours {key}: G-ours={worst <= 1e-10} ({worst:.2e}) ' +
                ' '.join(f"{j}:{rec[j]['seconds_warm']:.4f}s" for j in SCREEN))
        return out
    if cfg.get('ours'):
        O.phase('ours_pass', ours_pass, store='ours_pass')

    # ---- generator probe runs (A2.3): `none` native on the screening cases with the upwind-generated basis -------
    if probe is not None:
        def probe_runs():
            Pu = jnp.asarray(PROBE.pop('Phi'))
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
            jj = list(range(f_.n)) if not cfg.get('dry') else [0, 1]
            rf = SC.Refs(d, cname, f_, jj)
            for option in options:
                for j in jj:
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
