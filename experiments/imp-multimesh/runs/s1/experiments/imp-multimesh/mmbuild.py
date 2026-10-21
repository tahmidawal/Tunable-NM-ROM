"""imp-multimesh build / accuracy driver (DESIGN 3, 5, A1, A2). One mode per invocation:

  sources  J0: per source mesh s, FOM snapshots of the 576 training trajectories, POD (rmax), certificate, codes,
           trust radii, ECSW training states and NNLS paths per rank (kappa path to 8). Timed phases; saves the basis,
           codes and rules for the later jobs.
  target   J1/J3/J4: on a target mesh L, for each arm (pm / cp-refit / cp-reuse), the basis (loaded or resampled),
           trust radii from transformed codes, the rule (loaded, refitted or mapped), then every (r, kappa)
           configuration on the 38 cases: prod (certified, fresh Jacobian every step), tight, tighter, E0 scores, Q1.
           Optional: the PM snapshot-cost estimate trajectories at L.

    python mmbuild.py <mode> <config.json> <outdir>
Every phase is timed (wall seconds, block_until_ready) and written to <outdir>/<mode>.json after every step.
"""
from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import mmcore as M

H, BD, RM, PL = M.H, M.BD, M.RM, M.PL
import score as SC          # noqa: E402  (jcp-hr-baselines, byte-identical)

D = 2
COHORTS = ('dev6', 'val32')


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def check_env(cfg):
    if not cfg.get('allow_cpu'):
        assert jax.default_backend() == 'gpu', jax.default_backend()
    assert os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest'
    assert jax.config.jax_enable_x64
    log(f'jax_backend={jax.default_backend()} x64=True precision=highest device={jax.devices()[0].device_kind}')


def dev_mem():
    try:
        s = jax.devices()[0].memory_stats() or {}
        return dict(in_use=int(s.get('bytes_in_use', -1)), peak=int(s.get('peak_bytes_in_use', -1)))
    except Exception:  # noqa: BLE001
        return dict(in_use=-1, peak=-1)


class Rec:
    def __init__(self, path, base):
        self.path, self.d = Path(path), base
        self.save()

    def save(self):
        PL.dump(self.path, self.d)


def timed(rec, key, fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    jax.block_until_ready(out) if isinstance(out, jax.Array) else None
    rec.d.setdefault('timers', {})[key] = rec.d['timers'].get(key, 0.) + time.perf_counter() - t0
    rec.d.setdefault('mem', {})[key] = dev_mem()
    rec.save()
    return out


def cases_all(which='all38'):
    out = [('dev6', j) for j in range(6)]
    if which == 'all38':
        out += [('val32', j) for j in range(32)]
    return out


def families():
    return {c: H.Family(D, c) for c in COHORTS}


def refs_for(fams):
    return {c: SC.Refs(D, c, fams[c], range(fams[c].n)) for c in COHORTS}


# ------------------------------------------------------------------------------------------- sources ----

def build_source(rec, key, s, steps, ranks, kappas, nnls_s, out, ntraj=None, log_=log, ecsw_ranks=None):
    """Snapshots -> POD -> certificate -> codes -> trust -> states -> ECSW per rank. Returns dict of artefacts."""
    fam = H.Family(D, 'train')
    rows = list(range(fam.n if ntraj is None else ntraj))
    save = tuple(range(51)) if steps == 'all' else H.SNAP_STEPS
    T = BD.Timer()
    cell = PL.Cell(D, s, log=log_, timer=T)
    t0 = time.perf_counter()
    S, sinfo = BD.snapshots(D, s, fam, rows, save_steps=save, scheme='CNR', log=log_)
    t_snap = time.perf_counter() - t0
    rmax = max(ranks)
    t0 = time.perf_counter()
    S, Phi, sig, trust = cell.basis_from(S, sinfo, rmax, ranks, 'central')
    t_pod = time.perf_counter() - t0
    t0 = time.perf_counter()
    C = np.asarray(BD.project_codes(S, Phi, 1 << 16))
    t_codes = time.perf_counter() - t0
    del S
    gc.collect()
    gate = cell.info['basis[central]']['gate']
    info = dict(s=s, steps=steps, n_snap=int(C.shape[0]), rows=len(rows), snap_seconds=t_snap, pod_seconds=t_pod,
                codes_seconds=t_codes, gate=gate, trust=trust, sigma=sig[:rmax + 1].tolist(),
                snapshots=sinfo, pod_timers=dict(T.t))
    certified = [r for r in ranks if r in set(gate['constructible'])]
    info['certified_ranks'] = certified          # A4.1: a failed certificate makes that rank NOT-CONSTRUCTIBLE
    cell.set_basis(Phi, trust)
    t0 = time.perf_counter()
    cell.ecsw_states(fam)
    info['states_seconds'] = time.perf_counter() - t0
    info['ecsw'] = {}
    rules = {}
    for r in [x for x in (ranks if ecsw_ranks is None else ecsw_ranks) if x in certified]:
        t0 = time.perf_counter()
        c0 = T.t.get('ecsw_contrib', 0.)
        n0_ = T.t.get('ecsw_nnls', 0.)
        rec_r = cell.ecsw(r, kappas=tuple(kappas), max_seconds=nnls_s)
        info['ecsw'][r] = dict(cell.info['ecsw'][r], seconds=time.perf_counter() - t0,
                               contrib_seconds=T.t['ecsw_contrib'] - c0, nnls_seconds=T.t['ecsw_nnls'] - n0_)
        for k, v in rec_r.items():
            if v is not None:
                rules[(r, k)] = (np.asarray(v['nodes']), np.asarray(v['weights']))
        log_(f'   source {s} r={r}: ecsw {info["ecsw"][r]["seconds"]:.0f}s supports '
             f'{ {k: (None if v is None else len(v["nodes"])) for k, v in rec_r.items()} }')
        rec.d[key] = info
        rec.save()
    od = Path(out) / f'src{s}'
    od.mkdir(parents=True, exist_ok=True)
    np.save(od / 'Phi.npy', np.asarray(Phi))
    np.save(od / 'C.npy', C)
    np.save(od / 'Phi_lat.npy', np.asarray(Phi[jnp.asarray(H.lattice_nodes(D, s))]))
    np.savez(od / 'rules.npz', **{f'n|{r}|{k}': v[0] for (r, k), v in rules.items()},
             **{f'w|{r}|{k}': v[1] for (r, k), v in rules.items()})
    (od / 'trust.json').write_text(json.dumps({str(k): v for k, v in trust.items()}))
    return dict(Phi=Phi, C=C, trust=trust, rules=rules, info=info, states=cell.states)


def mode_sources(cfg, out):
    rec = Rec(Path(out) / 'sources.json', dict(config=cfg, gpu=jax.devices()[0].device_kind, job=os.environ.get('SLURM_JOB_ID'),
                                               commit=os.environ.get('SOURCE_COMMIT'), complete=False))
    for sc in cfg['sources']:
        s = int(sc['L'])
        log('SOURCE', s)
        a = build_source(rec, f'src{s}', s, sc['steps'], cfg['ranks'], cfg['kappas'], cfg['nnls_seconds'], out)
        del a
        gc.collect()
    rec.d['complete'] = True
    rec.save()


# -------------------------------------------------------------------------------------------- target ----

def load_source(path):
    p = Path(path)
    Phi = np.load(p / 'Phi.npy', mmap_mode='r')
    C = np.load(p / 'C.npy')
    z = np.load(p / 'rules.npz')
    rules = {}
    for k in z.files:
        t, r, kk = k.split('|')
        rules.setdefault((int(r), kk), [None, None])[0 if t == 'n' else 1] = z[k]
    trust = {int(k): v for k, v in json.loads((p / 'trust.json').read_text()).items()}
    return Phi, C, {k: tuple(v) for k, v in rules.items()}, trust


def contrib_and_nnls(states, Phi_dev, r, L, kappas, nnls_s, blocked):
    t0 = time.perf_counter()
    if blocked:
        G, b, cand, cinfo = M.ecsw_contrib_blocked(states, Phi_dev, r, L, log=log)
    else:
        G, b, cand, cinfo = BD.ecsw_contrib(states, Phi_dev[:, :r], D, L)
        jax.block_until_ready(G)
    tc = time.perf_counter() - t0
    t0 = time.perf_counter()
    recr, summ = BD.ecsw_weights(G, b, cand, r, tuple(kappas), max_seconds=nnls_s)
    tn = time.perf_counter() - t0
    del G, b
    rules = {f'k{k}': (None if v is None else (np.asarray(v['nodes']), np.asarray(v['weights'])))
             for k, v in recr.items()}
    summ.pop('checkpoints_all', None)
    return rules, dict(contrib_seconds=tc, nnls_seconds=tn, contrib=dict(cinfo, block_norms=None), paths=summ,
                       supports={k: (None if v is None else len(v[0])) for k, v in rules.items()})


def run_arm(rec, arm, L, cfg, fams, refs, states, out):
    """One arm on target mesh L: basis, trust, rules, then every configuration."""
    name, kind, s = arm['name'], arm['kind'], int(arm['source'])
    A = rec.d['arms'].setdefault(name, dict(kind=kind, source=s, L=L, timers={}, configs={}))
    for r in cfg['ranks']:                        # A3.1/A4.1: the registered universe starts UNRESOLVED
        for k in cfg['kappas']:
            A['configs'][f'r{r}|k{k}|CNR'] = dict(status='UNRESOLVED', reason='not run')
    rec.save()
    src_info = json.loads((Path(arm['source_dir']).parent / 'sources.json').read_text()) \
        if (Path(arm['source_dir']).parent / 'sources.json').exists() else None
    cert_s = None if src_info is None else src_info.get(f'src{s}', {}).get('certified_ranks')
    Phi_s, C_s, rules_s, trust_s = load_source(arm['source_dir'])
    rmax = max(cfg['ranks'])
    t0 = time.perf_counter()
    if s == L:
        Phi = jax.block_until_ready(jnp.asarray(np.asarray(Phi_s[:, :rmax])))
        R, rinfo = None, dict(identity=True)
        A['timers']['load'] = time.perf_counter() - t0
    else:
        Phi, R, rinfo = M.resample_basis(np.asarray(Phi_s[:, :rmax]), s, L)
        A['timers']['resample'] = rinfo['seconds']
        if L > s:
            rinfo['node_consistency'] = M.node_consistency(np.asarray(Phi_s[:, :rmax]), s, Phi, L)
    A['resample'] = rinfo
    if not rinfo.get('orth_ok', True):
        for r in cfg['ranks']:
            for k in cfg['kappas']:
                A['configs'][f'r{r}|k{k}|CNR'] = dict(status='NOT-CONSTRUCTIBLE', reason='resampled basis certificate')
        rec.save()
        return
    t0 = time.perf_counter()
    trust = {}
    rank_info = {}
    for r in cfg['ranks']:
        if cert_s is not None and r not in cert_s:
            for k in cfg['kappas']:
                A['configs'][f'r{r}|k{k}|CNR'] = dict(status='NOT-CONSTRUCTIBLE', reason='source basis certificate')
            continue
        if s == L:
            trust[r] = trust_s[r]
            continue
        if R is not None:
            ok, cond = M.rank_ok(R, r)
            rank_info[r] = dict(ok=ok, cond=cond)
            if not ok:
                continue
        Ct = M.transform_codes(C_s, s, L, r, R)
        trust[r] = BD.trust_radii(Ct, [r], D)[r]
    A['timers']['codes'] = time.perf_counter() - t0
    A['trust'], A['rank'] = trust, rank_info
    cell = M.Cell(L, Phi, trust, log=log)
    rec.save()
    np.save(Path(out) / f'Phi_lat_{name}.npy', np.asarray(Phi[jnp.asarray(H.lattice_nodes(D, L))]))
    blocked = L > cfg.get('dense_max_L', 1024)
    lat = jnp.asarray(H.lattice_nodes(D, L))
    Phi_lat = np.asarray(Phi[lat])
    for r in cfg['ranks']:
        if r not in trust:
            for k in cfg['kappas']:
                if A['configs'][f'r{r}|k{k}|CNR']['status'] == 'UNRESOLVED':
                    A['configs'][f'r{r}|k{k}|CNR'] = dict(status='NOT-CONSTRUCTIBLE', reason='rank check (A2.4)')
            continue
        Wk = {}
        if kind == 'refit' and s != L:
            rr, inf = contrib_and_nnls(states, Phi, r, L, cfg['kappas'], cfg['nnls_seconds'], blocked)
            A.setdefault('ecsw', {})[r] = inf
        elif kind == 'reuse' and s != L:
            assert L > s
            t0 = time.perf_counter()
            rr = {}
            for k in cfg['kappas']:
                v = rules_s.get((r, f'k{k}'))
                rr[f'k{k}'] = None if v is None else (M.map_nodes(v[0], s, L), v[1])
            A.setdefault('ecsw', {})[r] = dict(map_seconds=time.perf_counter() - t0,
                                               supports={k: (None if v is None else len(v[0])) for k, v in rr.items()})
            # control-3 style deterministic check on every reused rule: (L/s) Phi_L[p'] == Phi_s[p]
            for k, v in rr.items():
                if v is not None:
                    src = rules_s[(r, k)][0]
                    a_ = (L / s) * np.asarray(Phi[jnp.asarray(v[0]), :r])
                    b_ = np.asarray(Phi_s[src, :r])
                    A['ecsw'][r][f'map_check_{k}'] = float(np.max(np.abs(a_ - b_)) / np.max(np.abs(b_)))
        else:
            rr = {f'k{k}': (None if rules_s.get((r, f'k{k}')) is None else rules_s[(r, f'k{k}')]) for k in cfg['kappas']}
        rec.save()
        for k in cfg['kappas']:
            cid = f'r{r}|k{k}|CNR'
            v = rr.get(f'k{k}')
            if v is None:
                why = 'missing source rule (see sources.json path reason)'
                if kind == 'refit' and s != L:
                    pth = A['ecsw'][r]['paths']
                    why = pth['attempts'][pth['selected']].get('reason')
                A['configs'][cid] = dict(status='UNRESOLVED', reason=why)     # A4.1: no impossibility certificate
                rec.save()
                continue
            t0 = time.perf_counter()
            data, dinfo = cell.ecsw_data(r, v[0], v[1])
            t_data = time.perf_counter() - t0
            t0 = time.perf_counter()
            summ, arrs = M.run_config(cell, r, data, ('CNR', 0.005), cases_all(cfg.get('cases', 'all38')), fams, refs,
                                      tag=f'{name}|{cid}')
            summ.update(status='MEASURED-Q1' if summ['q1']['q1'] else 'MEASURED-FAIL', armdata_seconds=t_data,
                        data_info=dinfo, run_seconds=time.perf_counter() - t0, summary=M.summarise(summ['rows']))
            Wk[k] = {key[2:]: v for key, v in arrs.items() if key.startswith('W|')}
            A['configs'][cid] = summ
            np.savez_compressed(Path(out) / f'W_{name}_r{r}_k{k}.npz', nodes=v[0], weights=v[1], **arrs)
            log(f'  {name} {cid}: worst {summ["summary"]["worst"]:.4g} median {summ["summary"]["median"]:.4g} '
                f'Q1 {summ["q1"]["q1"]} [{summ["run_seconds"]:.0f}s]')
            del data
            rec.save()
        # rollout-agreement sensitivity (A2.5): kappa=4 vs kappa=8 at the same r, lattice distance on every case
        ks = sorted(Wk)
        if len(ks) == 2:
            Pl = Phi_lat[:, :r]
            dd = {}
            for cs in Wk[ks[0]]:
                coh, j = cs.split('|')
                Fa = H.to_lattice(np.asarray(Wk[ks[0]][cs]) @ Pl.T, D)
                Fb = H.to_lattice(np.asarray(Wk[ks[1]][cs]) @ Pl.T, D)
                dd[cs] = SC.lattice_distance(Fa, Fb, refs[coh].n0(int(j)))
            cid = f'r{r}|k{ks[0]}|CNR'
            A['configs'][cid]['rollout'] = dict(vs=f'k{ks[1]}', max=max(dd.values()), per_case=dd,
                                                ok=bool(all(np.isfinite(v) and v <= 2.5e-4 for v in dd.values())))
            rec.save()
    del cell, Phi
    gc.collect()


def pm_estimate(rec, L, n, seed):
    """A1.8/A2.8: snapshot-recipe FOM trajectory times at L for n random training trajectories."""
    fam = H.Family(D, 'train')
    rows = np.random.default_rng(seed).choice(fam.n, n, replace=False)
    fom = H.make_fom(D, L)
    sch = H.sched('CNR', 0.005, save_steps=H.SNAP_STEPS)
    times = []
    for j in rows:
        PL.burn(0.05)
        t0 = time.perf_counter()
        o = fom(fam.ic(L, int(j)), fam.nu(int(j)), sch, 1e-10, 1e-12)
        res = np.asarray(o['res'])
        jax.block_until_ready(o['out'])
        times.append(time.perf_counter() - t0)
        assert np.all(np.isfinite(res)) and float(np.max(res)) <= 1e-10 * (1 + 1e-9)
        del o
        log(f'   pm-estimate traj {int(j)}: {times[-1]:.1f}s')
    rec.d['pm_estimate'] = dict(L=L, rows=[int(x) for x in rows], seconds=times, mean=float(np.mean(times)),
                                estimate_576=float(576 * np.mean(times)), range_576=[576 * min(times), 576 * max(times)],
                                note='first trajectory includes compilation')
    rec.save()


def mode_target(cfg, out):
    L = int(cfg['L'])
    rec = Rec(Path(out) / f'target{L}.json', dict(config=cfg, gpu=jax.devices()[0].device_kind, L=L,
                                                  job=os.environ.get('SLURM_JOB_ID'),
                                                  commit=os.environ.get('SOURCE_COMMIT'), arms={}, complete=False))
    fams = families()
    refs = refs_for(fams)
    states = None
    if any(a['kind'] == 'refit' and int(a['source']) != L for a in cfg['arms']):
        t0 = time.perf_counter()
        states, sinfo = BD.ecsw_states(D, L, H.Family(D, 'train'), None)
        rec.d['states'] = dict(sinfo, seconds=time.perf_counter() - t0)
        rec.save()
        log(f'states at {L}: {rec.d["states"]["seconds"]:.0f}s')
    for arm in cfg['arms']:
        log('ARM', arm['name'])
        run_arm(rec, arm, L, cfg, fams, refs, states, out)
    if cfg.get('pm_estimate'):
        del states
        gc.collect()
        pm_estimate(rec, L, cfg['pm_estimate']['n'], cfg['pm_estimate']['seed'])
    rec.d['complete'] = True
    rec.save()


def main():
    mode, cfgp, out = sys.argv[1], sys.argv[2], sys.argv[3]
    cfg = json.loads(Path(cfgp).read_text())
    Path(out).mkdir(parents=True, exist_ok=True)
    check_env(cfg)
    if mode == 'sources':
        mode_sources(cfg, out)
    elif mode == 'target':
        mode_target(cfg, out)
    else:
        raise SystemExit(f'unknown mode {mode}')
    log('MMBUILD COMPLETE')


if __name__ == '__main__':
    main()
