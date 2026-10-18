"""jcp-hr-baselines phase-II tuning job (Stage A) for one cell (DESIGN.md 3, 6, A1.3-A1.8, A2.2-A2.8, A15-A26).

Order (A26.3: required work first, the registered defer order last):
  1. refs; ours' batch-1 comparators (G-ours replay + warm timing on the 6 timing cases, A24.5);
  2. offline construction from TRAINING draws: snapshots, POD (certified), trust radii; (2D 256^2) the A1.3 saturation
     check with the all-step basis installed on failure; in-cell adequacy on the basis actually used (A2.2); advection
     POD; generator-probe basis (A2.3); ECSW training states and one support-targeted NNLS path per rank (A16/A20/A24);
  3. Stage A on the screening cases: HR arms (ECSW k2/k4/k8, Q-DEIM p/r 1/2/4) at every rank, then `none` at every rank,
     each x option x {native, common}, with in-job Q1 (tight native anchor on every case, tighter on the first two),
     a burned warm timing run checked against the scored run (<= 1e-12, A6.6) and E0 scores;
  4. the full-rank limiting comparator (central FOM) on the verdict cohort, convergence checked;
  5. deferrable in-job (A30.1/A31, run while the job's budget allows, in reverse drop order): the generator-probe
     runs, then the A2.18/A16.2 extension (A25.1), which is dropped first.
Every configuration is in a predeclared inventory with status not_run / construction_failed / run_failed / complete /
deferred; every completed configuration and case is checkpointed to <out>/tune.json at once. Arrays: <out>/arrays/
(output coefficients W, anchors, trajectories, gamma: lattice fields are W @ Phi_lat^T) and <out>/artifacts/ (basis,
HR data, ECSW paths; sha256-listed by the batch wrapper).

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
import build as BD  # noqa: E402
import pipeline as PL  # noqa: E402
import score as SC  # noqa: E402

T0 = time.perf_counter()
SCREEN = list(range(6))                         # 2D dev6 0-5; 3D 923801 0-5 (selection subset = timing cases)
HR_GRID = (('ecsw', 'k2'), ('ecsw', 'k4'), ('ecsw', 'k8'), ('qdeim', 1), ('qdeim', 2), ('qdeim', 4))
TIMED_TOL = 1e-12                               # A6.6: timed output vs its scored output
ADEQ_HELD = (3, 5, 6, 8, 12, 17, 22, 35, 45)    # A2.2 held-out steps
DEFER_FRAC = 0.85                               # deferrable work starts only before this fraction of budget_hours


def log(*a):
    print(f'[{time.perf_counter() - T0:8.1f}s]', *a, flush=True)


def disc_finite(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return float('nan')
    return float(np.max(np.abs(a - b)))


class Out:
    def __init__(self, d):
        self.d = Path(d)
        (self.d / 'arrays').mkdir(parents=True, exist_ok=True)
        (self.d / 'artifacts').mkdir(parents=True, exist_ok=True)
        self.res = dict(phases={}, inventory={}, stageA={}, construction_failures={})

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
                                            trace=traceback.format_exc()[-3000:], seconds=time.perf_counter() - t0,
                                            device_peak=PL.device_peak())
            log(f'ERR  {name}: {type(e).__name__}: {e}')
            v = None
        if store is not None:
            self.res[store] = v
        self.save()
        return v


def hr_points(kind, r, hp, cell):
    if kind == 'none':
        return 0
    if kind == 'ecsw':
        rec = cell.ecsw_rec.get(r, {}).get(hp)
        return None if rec is None else int(rec['size'])
    return int(hp)


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
    budget_s = 3600. * cfg.get('budget_hours', 1e9)
    coh = 'dev6' if d == 2 else 'val64'
    fam_tr = H.Family(d, 'train')
    fam = H.Family(d, coh, count=None if d == 2 else 64)
    cell = PL.Cell(d, L, log=log)
    INV = O.res['inventory']

    def set_status(cid, st, **kw):
        INV[cid] = dict(status=st, **kw)

    grid = [(r, 'none', '-') for r in ranks] + [(r, k, h) for r in ranks for k, h in HR_GRID]
    for r, kind, hrp in grid:                          # predeclared registered inventory (A26.2)
        hp = hrp if kind != 'qdeim' else int(hrp * r)
        for option in options:
            for solver in ('native', 'common'):
                set_status(PL.cfg_id(kind, r, hp, option[0], option[1], solver), 'not_run')
    O.res['fom_limiting'] = {}                          # A33.1: FOM inventory predeclared at job setup
    for cname_ in (('dev6', 'val32') if d == 2 else ('val64',)):
        nn = (6 if cname_ == 'dev6' else 32) if d == 2 else 64
        for option in options:
            for j in (range(nn) if not cfg.get('dry') else [0, 1]):
                O.res['fom_limiting'][f'{cname_}|{j}|{option[0]}|{option[1]:.8g}'] = dict(status='not_run')
    refs = O.phase('refs', lambda: SC.Refs(d, coh, fam, SCREEN))
    if refs is None:
        return
    O.res['n0'] = {j: refs.n0(j) for j in refs.f}
    n0 = O.res['n0']
    ELIG = [j for j in SCREEN if bool(refs.st[j]['eligible'])]       # A32.1: fixed mask from the reference store
    O.res['eligible_screen_cases'] = ELIG

    # ---- 1. ours: batch-1 comparators, G-ours replay and same-job warm timing on the 6 timing cases --------------
    O.res['ours_pass'] = {}

    def ours_pass():
        import ours as OU
        if d == 2:
            exp = json.loads((HERE / 'inputs' / 'p1_analysis.json').read_text())['configurations']
            ref_e = {k: {j: exp[k]['e0_per_case'][f'dev6|{j}'] for j in SCREEN} for k in cfg['ours']}
        else:
            exp = json.loads((HERE / 'inputs' / 'speed3d_batch1_e0.json').read_text())['configs']
            ref_e = {k: {j: exp[k]['e0'][str(j)] for j in SCREEN} for k in cfg['ours']}
        oxs = {}
        for key in cfg['ours']:
            rec = dict(status='construction_failed', cases={},
                       batch=(1 if d == 2 else exp[key].get('batch', 1)),
                       qualification=('P1 Q1+Q2 (A21.1; FRONT-2D v1 member if listed in A27)' if d == 2 else
                                      'qualified subset (jcp-qualify Q2, A29)' if exp[key].get('batch') == 2 else
                                      'diagnostic only (A22.0; Q1 + deployed-rule evidence)'))
            O.res['ours_pass'][key] = rec
            try:
                if d == 2:
                    if 'p1' not in oxs:
                        oxs['p1'] = OU.OursP1(L, cfg['ours'], log=log)
                    ox_ = oxs['p1']
                    fx = ox_.fn(key)
                else:
                    st, rule = key.split('|')[:2]
                    if (st, rule) not in oxs:
                        oxs[(st, rule)] = OU.OursS3(st, rule, log=log)
                    ox_ = oxs[(st, rule)]
                    fx = ox_.fn(key)
            except Exception as e:  # noqa: BLE001
                rec['error'] = f'{type(e).__name__}: {e}'
                O.save()
                log(f'   ours {key}: construction failed: {rec["error"]}')
                continue
            rec['status'] = 'running'
            try:
                for j in SCREEN:
                    u, v = ox_.u0(fam, j), jnp.float64(fam.nu(j))
                    o_w = fx(u, v)
                    jax.block_until_ready(o_w['fields'])            # warm call; its output is the scored one
                    Fw = ox_.lattice(o_w)
                    PL.burn(0.1)
                    t0 = time.perf_counter()
                    o = fx(u, v)
                    jax.block_until_ready(o['fields'])
                    ts = time.perf_counter() - t0
                    Fl = ox_.lattice(o)
                    sc = refs.score_safe(Fw, j)
                    e = sc['e']
                    rel = abs(e - ref_e[key][j]) / abs(ref_e[key][j]) if np.isfinite(e) else float('inf')
                    tv = disc_finite(Fl, Fw)
                    rec['cases'][j] = dict(seconds_warm=ts, e0=sc, e0_exported=ref_e[key][j], replay_rel=rel,
                                           timed_vs_scored=tv, timing_ok=bool(np.isfinite(tv) and tv <= TIMED_TOL))
                    O.npz('arrays', f'X~{key}~{j}', F=Fw)
                    O.save()
                worst = max(c['replay_rel'] for c in rec['cases'].values())
                rec.update(status='complete', replay_worst_rel=worst, G_ours=bool(worst <= 1e-10),
                           timing_ok=all(c['timing_ok'] for c in rec['cases'].values()))
                log(f'   ours {key}: G-ours={rec["G_ours"]} ({worst:.2e}) ' +
                    ' '.join(f"{j}:{c['seconds_warm']:.4f}s" for j, c in rec['cases'].items()))
            except Exception as e:  # noqa: BLE001
                rec.update(status='run_failed', error=f'{type(e).__name__}: {e}', trace=traceback.format_exc()[-2000:])
                log(f'   ours {key}: run failed: {rec["error"]}')
            O.save()
        oxs.clear()
        return {k: v['status'] for k, v in O.res['ours_pass'].items()}
    if cfg.get('ours'):
        O.phase('ours_pass', ours_pass, store='ours_status')

    # ---- 2. offline construction (training only) ----------------------------------------------------------------
    rows = list(range(fam_tr.n if cfg.get('n_traj') is None else cfg['n_traj']))
    bb = O.phase('basis', lambda: cell.build_basis(fam_tr, rows, rmax, allr))
    if bb is None:
        for cid in INV:
            set_status(cid, 'construction_failed', reason='basis phase failed')
        O.save()
        return
    S, Phi, sig, trust = bb
    bb = None
    cell.set_basis(Phi, trust)
    O.res['basis'] = cell.info['basis[central]']
    O.res['basis_used'] = '12-step (central)'

    if cfg.get('saturation'):                          # A1.3: 2D 256^2 only; all-step basis installed on failure
        def saturation():
            Sall, sinfo = BD.snapshots(d, L, fam_tr, rows, save_steps=tuple(range(51)), log=None)
            ca = PL.Cell(d, L, log=log)
            _, Pa, sa, ta = ca.basis_from(Sall, sinfo, rmax, allr, 'allstep')
            n2 = np.sum(Sall * Sall, 1)
            out = {}
            for r in ranks:
                for tag, P_ in (('12step', cell.Phi[:, :r]), ('allstep', Pa[:, :r])):
                    C = np.asarray(BD.project_codes(Sall, P_))
                    e2 = np.maximum(n2 - np.sum(C ** 2, 1), 0.)
                    out.setdefault(r, {})[tag] = float(np.sqrt(np.mean(e2 / n2)))
                out[r]['adequate'] = bool(out[r]['12step'] <= 1.10 * out[r]['allstep'])
            del Sall
            ok = all(out[r]['adequate'] for r in ranks)
            if not ok:                                  # install the all-step basis (certified on all states)
                cell.set_basis(Pa, ta)
                cell.constructible_r = ca.constructible_r
                O.res['basis'] = ca.info['basis[allstep]']
                O.res['basis_used'] = 'all-step (A1.3 saturation failed; 1024^2 cell labelled snapshot-limited)'
            return dict(per_rank=out, adequate=ok, basis_used=O.res['basis_used'])
        O.phase('saturation', saturation, store='saturation')
    Phi, trust = cell.Phi, cell.trust
    if d == 2:                                          # 3D: the lattice is every interior node (Phi itself)
        O.npz('artifacts', 'Phi_lat', Phi_lat=np.asarray(Phi)[H.lattice_nodes(d, L)], nodes=H.lattice_nodes(d, L))
    O.npz('artifacts', 'Phi', Phi=np.asarray(Phi), trust=np.asarray([trust.get(r, np.nan) for r in allr]),
          ranks=np.asarray(allr), constructible=np.asarray(sorted(cell.constructible_r)))

    def adequacy():                                    # A2.2: the first 32 of the seeded permutation, actual basis
        rset = set(rows)
        perm = [int(j) for j in np.random.default_rng(BD.ECSW_SEED_TRAJ).permutation(fam_tr.n) if int(j) in rset][:32]
        Sh, _ = BD.snapshots(d, L, fam_tr, perm, save_steps=ADEQ_HELD, log=None)
        ns = len(H.SNAP_STEPS)
        Si = np.concatenate([S[rows.index(j) * ns:(rows.index(j) + 1) * ns] for j in perm])
        out = dict(trajectories=perm)
        for r in ranks:
            Pr = cell.Phi[:, :r]

            def e(X):
                C = np.asarray(BD.project_codes(X, Pr))
                n2 = np.sum(X * X, 1)
                return float(np.sqrt(np.mean(np.maximum(n2 - np.sum(C ** 2, 1), 0.) / n2)))
            out[r] = dict(held=e(Sh), inset=e(Si))
            out[r]['ratio'] = out[r]['held'] / max(out[r]['inset'], 1e-300)
            out[r]['adequate'] = bool(out[r]['ratio'] <= 1.10)
        out['adequate'] = all(out[r]['adequate'] for r in ranks)
        return out
    O.phase('adequacy', adequacy, store='adequacy')

    pmax = 4 * rmax
    p_list = sorted({m * r for r in allr for m in (1, 2, 4)})
    O.phase('adv_pod', lambda: cell.build_adv(S, pmax, p_list))
    del S
    O.res['adv'] = {k: v for k, v in cell.info.get('adv', {}).items() if k != 'sigma'}

    PROBE = {}                                         # generator probe after the central snapshots are released

    def probe_basis():                                 # A2.3: the banks' generator (upwind BE)
        cp = PL.Cell(d, L, log=log)
        Su, Pu, su, tu = cp.build_basis(fam_tr, rows, rmax, ranks, gen='upwind')
        del Su
        PROBE.update(Phi=np.asarray(Pu), constructible=sorted(cp.constructible_r))
        O.npz('artifacts', 'Phi_probe_lat', Phi_lat=PROBE['Phi'][H.lattice_nodes(d, L)] if d == 2 else PROBE['Phi'])
        del Pu
        return dict(info=cp.info['basis[upwind]'], trust=tu, constructible=sorted(cp.constructible_r))
    probe = O.phase('generator_probe_basis', probe_basis, store='probe_basis')
    O.phase('ecsw_states', lambda: cell.ecsw_states(fam_tr), store='ecsw_states')

    def ecsw_r(r):
        big = d == 3 and r == max(ranks)
        cell.ecsw(r)
        if big:                                        # A26.1: process-cumulative peak after the maximal path
            pk = PL.device_peak()
            ok = bool(np.isfinite(pk) and 0 <= pk <= 120e9) or bool(cfg.get('dry') and pk == -1)   # CPU dry run: no telemetry
            O.res['memory_admission'] = dict(peak_after=pk, limit=120e9, admitted=ok)
            if not ok:
                cell.ecsw_rec.pop(r, None)
                O.res['construction_failures'][f'ecsw|{r}|*'] = (f'not constructible (memory): cumulative device peak '
                                                                 f'{pk} B not certified below 120 GB')
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
    for r in ranks:
        if O.phase(f'ecsw_r{r}', lambda r=r: ecsw_r(r), store=f'ecsw_r{r}') is None:
            O.res['construction_failures'][f'ecsw|{r}|*'] = O.res['phases'][f'ecsw_r{r}']['error']

    # ---- 3. Stage A ---------------------------------------------------------------------------------------------
    search_t0 = time.perf_counter()

    def screen(r, kind, hrp, tag='A'):
        """One (rank, HR kind, knob): construction data, then every option x {native, common} on SCREEN with Q1.
        Each solver's configuration is checkpointed as soon as it completes; failures are isolated per solver."""
        hp = hrp if kind != 'qdeim' else int(hrp * r)
        cids = {(o, s): PL.cfg_id(kind, r, hp, o[0], o[1], s) for o in options for s in ('native', 'common')}
        try:
            dat, ainfo = cell.arm_data(kind, r, hp)
        except Exception as e:  # noqa: BLE001
            msg = f'{type(e).__name__}: {e}'
            O.res['construction_failures'][f'{kind}|{r}|{hp}'] = msg
            for cid in cids.values():
                set_status(cid, 'construction_failed', reason=msg, stage=tag)
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
            pname = f'{tag}|{kind}|{r}|{hp}|{option[0]}|{option[1]:.8g}'

            def block(option=option):
                tight, tighter = {}, {}
                apart = O.res.setdefault('anchors_partial', {}).setdefault(pname, {})
                for j in SCREEN:                       # A33.1: each anchor checkpointed as it completes
                    tight[j] = cell.run(kind, r, dat, 'native', option, 'tight', fam.ic(L, j), fam.nu(j))
                    apart[f'tight|{j}'] = dict(cert=tight[j]['cert'], W=tight[j]['W'])
                    O.save()
                for j in SCREEN[:2]:
                    tighter[j] = cell.run(kind, r, dat, 'native', option, 'tighter', fam.ic(L, j), fam.nu(j))
                    apart[f'tighter|{j}'] = dict(cert=tighter[j]['cert'], W=tighter[j]['W'])
                    O.save()
                for solver in ('native', 'common'):
                    cid = cids[(option, solver)]
                    try:
                        recs = {}
                        part = O.res.setdefault('stageA_partial', {}).setdefault(cid, {})
                        for j in SCREEN:
                            recs[j] = cell.run(kind, r, dat, solver, option, 'prod', fam.ic(L, j), fam.nu(j))
                            warm = cell.run(kind, r, dat, solver, option, 'prod', fam.ic(L, j), fam.nu(j),
                                            certify=False, burn_s=0.1)
                            recs[j]['seconds_warm'] = warm['seconds']
                            recs[j]['timed_vs_scored'] = disc_finite(warm['fields_lat'], recs[j]['fields_lat'])
                            part[j] = dict(cert=recs[j]['cert'], seconds_warm=warm['seconds'],   # A32.2: per-case
                                           timed_vs_scored=recs[j]['timed_vs_scored'],
                                           e0=refs.score_safe(recs[j]['fields_lat'], j), W=recs[j]['W'])
                            O.save()
                        q1 = PL.q1_config(recs, tight, tighter, {j: n0[j] for j in SCREEN}, SCREEN[:2], SCREEN)
                        e0 = {j: refs.score_safe(recs[j]['fields_lat'], j) for j in SCREEN}
                        O.npz('arrays', f'{tag}~{cid}', W=np.stack([recs[j]['W'] for j in SCREEN]),
                              Wtight=np.stack([tight[j]['W'] for j in SCREEN]),
                              Wtighter=np.stack([tighter[j]['W'] for j in SCREEN[:2]]),
                              C=np.stack([recs[j]['C'] for j in SCREEN]),
                              gamma=np.stack([recs[j]['gamma'] for j in SCREEN]))
                        es = [e0[j]['e'] for j in ELIG]                  # A32.1: reference-eligible case-kinds only
                        tv = [recs[j]['timed_vs_scored'] for j in SCREEN]
                        timing_ok = bool(all(np.isfinite(x) and x <= TIMED_TOL for x in tv))
                        O.res['stageA'][cid] = dict(
                            q1=q1, e0=e0, hr_points=hr_points(kind, r, hp, cell), arm=ainfo, quality=quality,
                            stage=tag, kind=kind, r=r, hrp=hp, scheme=option[0], dt=option[1], solver=solver,
                            W=float(np.max(es)) if es and all(np.isfinite(es)) else float('nan'),
                            med=float(np.median(es)) if es and all(np.isfinite(es)) else float('nan'),
                            eligible_cases=ELIG,
                            t=float(np.median([recs[j]['seconds_warm'] for j in SCREEN])), timing_ok=timing_ok,
                            runs={j: dict(cert=recs[j]['cert'], seconds_warm=recs[j]['seconds_warm'],
                                          timed_vs_scored=recs[j]['timed_vs_scored']) for j in SCREEN},
                            anchors=dict(tight={j: tight[j]['cert'] for j in SCREEN},
                                         tighter={j: tighter[j]['cert'] for j in SCREEN[:2]}))
                        set_status(cid, 'complete', stage=tag)
                        O.res['stageA_partial'].pop(cid, None)
                        v = O.res['stageA'][cid]
                        log(f'   {cid}: q1={q1["q1"]} W={v["W"]:.4g} t={v["t"]:.4g}s timing_ok={timing_ok}')
                    except Exception as e:  # noqa: BLE001
                        set_status(cid, 'run_failed', reason=f'{type(e).__name__}: {e}', stage=tag)
                    O.save()
                O.res['anchors_partial'].pop(pname, None)
                O.res['search_hours'] = sum(v['seconds'] for k_, v in O.res['phases'].items()
                                            if k_.startswith('A|') or k_.startswith('EXT|')) / 3600
                return True
            if O.phase(pname, block) is None:                       # anchors failed: both solvers of this option
                for s in ('native', 'common'):
                    if INV.get(cids[(option, s)], {}).get('status') in (None, 'not_run'):
                        set_status(cids[(option, s)], 'run_failed', stage=tag, reason=O.res['phases'][pname]['error'])
                O.save()

    def deferrable():
        return time.perf_counter() - T0 < DEFER_FRAC * budget_s

    for r in ranks:                                    # never deferred: the HR grids (DESIGN 12)
        for kind, hrp in HR_GRID:
            screen(r, kind, hrp)
    for r in ranks:                                    # never deferred: `none`, both solvers, every rank (A1.14, A2.15)
        screen(r, 'none', '-')
    O.res['search_hours_registered'] = (time.perf_counter() - search_t0) / 3600

    # ---- 4. full-rank limiting comparator (central FOM) on the verdict cohort ------------------------------------
    def fom_runs():
        fom = H.make_fom(d, L)
        cohorts = [('dev6', H.Family(2, 'dev6')), ('val32', H.Family(2, 'val32'))] if d == 2 else [('val64', fam)]
        ntol = 1e-10
        for cname, f_ in cohorts:                      # the setup inventory must match the cohorts
            assert all(f'{cname}|{j}|{o[0]}|{o[1]:.8g}' in O.res['fom_limiting']
                       for o in options for j in (range(f_.n) if not cfg.get('dry') else [0, 1]))
        for cname, f_ in cohorts:
            jj = list(range(f_.n)) if not cfg.get('dry') else [0, 1]
            rf = SC.Refs(d, cname, f_, jj)
            for option in options:
                for j in jj:
                    key = f'{cname}|{j}|{option[0]}|{option[1]:.8g}'
                    try:
                        o = fom(f_.ic(L, j), f_.nu(j), H.sched(*option), ntol, 1e-12)
                        res = np.asarray(o['res'])
                        conv = bool(np.all(np.isfinite(res)) and float(np.max(res)) <= ntol * (1 + 1e-9))
                        sc = rf.score_safe(cell.lattice(o['out']), j) if conv else None
                        O.res['fom_limiting'][key] = dict(converged=conv, worst_rel_residual=float(np.nanmax(res)),
                                                          e0=sc, status='complete' if conv else 'run_failed')
                    except Exception as e:  # noqa: BLE001
                        O.res['fom_limiting'][key] = dict(status='run_failed', error=f'{type(e).__name__}: {e}')
                    O.save()
        return len(O.res['fom_limiting'])
    O.phase('fom_limiting', fom_runs, store='fom_count')

    # ---- 5. deferrable work (A30.1: in-job deferrable items are the extension and the generator-probe runs) --------
    O.res['deferred'] = []

    # ---- generator probe runs (A2.3): `none` native on the screening cases with the upwind-generated basis ---------
    O.res['probe_runs'] = {}
    if probe is not None and 'Phi' in PROBE and not deferrable():
        O.res['deferred'].append('generator probe runs (budget)')
    if probe is not None and 'Phi' in PROBE:
        def probe_runs():
            Pu = jnp.asarray(PROBE.pop('Phi'))
            for r in ranks:
                for option in options:
                    cid = PL.cfg_id('none', r, '-', option[0], option[1], 'native', gen='upwind')
                    if r not in PROBE['constructible']:
                        O.res['probe_runs'][cid] = dict(status='construction_failed', reason='probe rank not certified')
                    elif not deferrable():
                        O.res['probe_runs'][cid] = dict(status='deferred', reason='budget')
                    else:
                        datp = dict(Phi=Pu[:, :r], LPhi=H.lap_cols(Pu[:, :r], d, L), DPhi=H.dsum_cols(Pu[:, :r], d, L))
                        rec = dict(status='complete', cases={})
                        try:
                            Ws = []
                            for j in SCREEN:
                                o = cell.run('none', r, datp, 'native', option, 'prod', fam.ic(L, j), fam.nu(j))
                                rec['cases'][j] = dict(e0=refs.score_safe(o['fields_lat'], j), cert=o['cert'],
                                                       W=o['W'])
                                Ws.append(o['W'])
                                O.res['probe_runs'][cid] = rec
                                O.save()
                            O.npz('arrays', f'PROBE~{cid}', W=np.stack(Ws))
                        except Exception as e:  # noqa: BLE001
                            rec.update(status='run_failed', error=f'{type(e).__name__}: {e}')
                        O.res['probe_runs'][cid] = rec
                    O.save()
            return len(O.res['probe_runs'])
        O.phase('generator_probe_runs', probe_runs, store='probe_count')
    def extension():                                   # A2.18 / A16.2 / A25.1
        import stats as ST
        # A28.3: accumulated C-search = the classified Stage-A screening phases (not wall time since Stage A)
        used_h = sum(v['seconds'] for k, v in O.res['phases'].items() if k.startswith('A|')) / 3600
        alloc = cfg.get('search_allocation_hours')
        items = [dict(id=k, t=v['t'], W=v['W']) for k, v in O.res['stageA'].items()
                 if v['q1']['q1'] and v['timing_ok'] and np.isfinite(v['W']) and np.isfinite(v['t'])]
        F = ST.front(items)
        todo = set()
        for f in F:
            v = O.res['stageA'][f['id']]
            if v['kind'] == 'ecsw' and v['hrp'] == 'k8':
                todo.add((v['r'], 'ecsw', 'k16'))
            if cfg.get('ext_rank') and v['r'] == max(ranks):
                todo.add((cfg['ext_rank'], v['kind'], v['hrp'] if v['kind'] != 'qdeim' else v['hrp'] // v['r']))
        run = bool(todo) and (alloc is None or used_h < alloc) and deferrable()
        rec = dict(front=[f['id'] for f in F], triggered=sorted(map(str, todo)), search_hours_used=used_h,
                   allocation_hours=alloc, run=run)
        O.res['extension'] = rec
        for r, kind, hrp in sorted(todo, key=str):
            hp = hrp if kind != 'qdeim' else int(hrp * r)
            for option in options:
                for s in ('native', 'common'):
                    set_status(PL.cfg_id(kind, r, hp, option[0], option[1], s), 'not_run' if run else 'deferred',
                               stage='EXT')
        O.save()
        if not run:
            return rec
        for r, kind, hrp in sorted(todo, key=str):
            if kind == 'ecsw' and r not in cell.ecsw_rec:
                if O.phase(f'ecsw_r{r}', lambda r=r: ecsw_r(r), store=f'ecsw_r{r}') is None:
                    O.res['construction_failures'][f'ecsw|{r}|*'] = O.res['phases'][f'ecsw_r{r}']['error']
            screen(r, kind, hrp, tag='EXT')
        return rec
    O.phase('extension', extension)
    O.res['search_hours'] = sum(v['seconds'] for k, v in O.res['phases'].items()
                                if k.startswith('A|') or k.startswith('EXT|')) / 3600

    O.res['inventory_counts'] = {s: sum(1 for v in INV.values() if v['status'] == s)
                                 for s in ('not_run', 'construction_failed', 'run_failed', 'complete', 'deferred')}
    O.res['timers'] = cell.T.t
    O.res['total_seconds'] = time.perf_counter() - T0
    O.save()
    log('done', O.res['inventory_counts'])


if __name__ == '__main__':
    main()
