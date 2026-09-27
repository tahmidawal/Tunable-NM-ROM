"""b-lowvisc gate: is the low-viscosity Burgers cell actually the harder regime?

ONE allocation, ONE GPU, both viscosity families side by side, no trained model anywhere.
This job exists to decide whether a training run is worth a day, and it answers the two legs
of the lane's hypothesis directly (DESIGN.md sections 2 and 6):

  (a) does POD degrade?  For each family: 128 truth trajectories at the evaluation mesh, the
      method-of-snapshots POD basis, and the static projection floor of every rank the panel
      would run, on the same six development cases.  The incumbent family's floors are
      compared against job 3780638's own `best-found` column, so the measurement is validated
      against a published number before the comparison is read.

  (b) does the full-order solve get more expensive?  The complete tuned Newton grid of job
      3780638 -- two time steps x four tolerances plus the two tight references -- timed here
      for BOTH families in one randomised order with three repetitions, plus a mesh probe at
      512 and 1024 intervals so the scaling is on the record too.

Everything is written incrementally to `result.json`.  Audit artifacts are written so that
`audit_gate.py` can recompute every reported number from saved arrays with NumPy alone, by a
different route than the one used here (the POD floors are recomputed from the snapshot Gram's
eigenvectors rather than from the modes).
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

import engines as e
import iterative_paths as ip
import ladder as LD
from ablation import pod_basis

import lv_common as LV

host, dump, sha_array, sha_file = LD.host, LD.dump, LD.sha_array, LD.sha_file
TIMES = np.array([0., .05, .1, .15, .2, .25], dtype=np.float64)
# every error in this job is measured on the observation grid = cfg['intervals']


def floors(F, V, ranks):
    """Static projection floor of each POD rank, in the two normalisations `engines.errors` uses.

    `F` is (cases, times, n) truth on the observation grid interior; `V` is (n, kmax) orthonormal.
    Returned percentages are over the six cases: worst over all output times, worst over the
    evolved times (t > 0), the median, and the t = 0 compression.
    """
    nc, nt, n = F.shape
    flat = F.reshape(nc * nt, n)
    C = flat @ V                                     # (cases*times, kmax)
    sq = np.cumsum(C ** 2, axis=1)
    tot = np.sum(flat ** 2, axis=1)
    n0 = np.linalg.norm(F[:, 0], axis=1)             # ||u(t=0)|| per case
    nr = np.linalg.norm(flat, axis=1).reshape(nc, nt)
    out = {}
    for k in ranks:
        res = np.sqrt(np.maximum(tot - sq[:, k - 1], 0.)).reshape(nc, nt)
        fi = res / n0[:, None]
        cr = res / np.maximum(nr, 1e-300)
        out[str(k)] = dict(
            rank=int(k),
            worst_all_times_percent=100. * float(np.max(fi)),
            worst_evolved_percent=100. * float(np.max(fi[:, 1:])),
            median_evolved_percent=100. * float(np.median(fi[:, 1:])),
            t0_percent=100. * float(np.max(fi[:, 0])),
            worst_all_times_current_percent=100. * float(np.max(cr)),
            per_case_worst_all_times_percent=(100. * np.max(fi, axis=1)).tolist())
    return out


def settings_of(cfg, name):
    for s in cfg['fom_settings']:
        if s['name'] == name:
            return s
    raise KeyError(name)


def frontier(rows, cost='median_gpu_ms', err='worst_evolved_percent'):
    """Non-dominated set on (cost, error): minimal, with ties broken by keeping both."""
    nd = []
    for r in rows:
        dominated = any((s[cost] <= r[cost] and s[err] <= r[err] and
                         (s[cost] < r[cost] or s[err] < r[err])) for s in rows if s is not r)
        if not dominated:
            nd.append(r['name'])
    return nd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    assert jax.default_backend() == 'gpu', jax.default_backend()
    assert jax.config.jax_enable_x64 and os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    begin = time.perf_counter()

    L, dt = int(cfg['intervals']), float(cfg['dt'])
    OBS = L                     # the observation grid: the panel mesh; finer probes are restricted to it
    ranks = sorted(set(cfg['pod_ranks']))
    assert int(cfg['leg_a_rank']) in ranks, (cfg['leg_a_rank'], ranks)
    fams = cfg['families']
    report = dict(config=cfg, lane='b-lowvisc', commit=os.environ.get('SOURCE_COMMIT'),
                  job_id=os.environ.get('SLURM_JOB_ID'), backend=jax.default_backend(),
                  gpu=jax.devices()[0].device_kind, x64=True, jax_version=jax.__version__,
                  matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'],
                  mem_fraction=os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION'),
                  intervals=L, dt=dt, observation_intervals=OBS, output_times=TIMES.tolist(),
                  final_cohort_unopened=True, trained_model_used=False,
                  timing_contract=('supplied dense initial field on GPU to six dense GPU output '
                                   'fields (gpu_seconds); the same invocation including the host '
                                   'upload and the host copy of the outputs (host_seconds); '
                                   'identical for every subject in this job'),
                  family_relation=LV.verify_family_relation(cfg['eval_seed'], cfg['eval_cases']),
                  families={}, reference=[], snapshots={}, pod={}, invocations=[], gates={},
                  verification=ip.verify(), complete=False)
    save = lambda: dump(out / 'result.json', report)
    save()
    print('FAMILY RELATION', report['family_relation']['passed'], flush=True)

    # ------------------------------------------------------------------ cohorts --
    cohorts, trains = {}, {}
    for f in fams:
        name = f['name']
        phys = LV.cohort(cfg, f['nu_lo'], f['nu_hi'])
        tr = LV.params_draw(cfg['train_seed'], cfg['train_trajectories'], f['nu_lo'], f['nu_hi'])
        assert not any(np.allclose(t, s) for t in tr for s in phys), f'train/eval overlap {name}'
        cohorts[name], trains[name] = phys, tr
        report['families'][name] = dict(
            nu_lo=f['nu_lo'], nu_hi=f['nu_hi'], physical_cases=phys.tolist(),
            physical_sha256=sha_array(phys), train_physical_sha256=sha_array(tr),
            viscosities=phys[:, 4].tolist(),
            grid_reynolds={str(m): LV.grid_reynolds(phys, m)
                           for m in sorted({L, *cfg.get('mesh_probe', {}).get('meshes', [])})},
            cohort_roles=['opened development'] * cfg['eval_cases']
                         + ['fresh development'] * cfg['eval_fresh_cases'])
    # the two cohorts must differ in the viscosity column and nowhere else
    a_name, b_name = fams[0]['name'], fams[1]['name']
    d = np.abs(cohorts[a_name] - cohorts[b_name])
    report['gates']['cohorts_differ_only_in_viscosity'] = dict(
        passed=bool(np.all(d[:, :4] == 0.)), max_descriptor_difference=float(np.max(d[:, :4])),
        families=[a_name, b_name])
    # values, not hashes: np.exp differs by 1 ulp between the GB10 and the cluster (CLAUDE.md)
    want = cfg.get('expected_incumbent_cases')
    if want is not None:
        got = cohorts['incumbent']
        ulp = np.abs(got - np.asarray(want)) / np.maximum(np.abs(np.asarray(want)), 1e-300)
        report['gates']['incumbent_cohort_matches_abl01_to_one_ulp'] = dict(
            passed=bool(np.max(ulp) <= 4e-16), max_relative_difference=float(np.max(ulp)),
            bitwise=bool(np.array_equal(got, np.asarray(want))))
    save()

    # --------------------------------------------------------------- references --
    rf, rt = int(cfg['reference_mesh']), float(cfg['reference_dt'])
    refs = {}
    q_ref, _ = e.make_fom(rf, rt)
    for fname in report['families']:
        for case, phys in enumerate(cohorts[fname]):
            t = time.perf_counter()
            f, it, rn = host(q_ref(jnp.asarray(e.initial(rf, phys)), float(phys[4]), 1e-11, 1e-9))
            assert np.isfinite(f).all() and np.max(rn) < 2e-11, (fname, case, float(np.max(rn)))
            f = np.array(f[:, ::rf // OBS, ::rf // OBS], copy=True)
            refs[(fname, case)] = f
            nm = f'ref_{fname}_L{rf}_case{case}.npz'
            np.savez_compressed(out / nm, fields=f, iterations=it, residuals=rn)
            report['reference'].append(dict(family=fname, case=case, intervals=rf, dt=rt,
                                            artifact=nm, downsampled_to=OBS,
                                            max_relative_residual=float(np.max(rn)),
                                            newton_iterations_total=int(np.sum(it)),
                                            field_sha256=sha_array(f),
                                            seconds=time.perf_counter() - t))
            print('REFERENCE', fname, case, round(time.perf_counter() - begin, 1), flush=True)
            save()
    del q_ref
    jax.clear_caches()

    # ---------------------------------------------- the same-grid truth, untimed --
    # Computed here, before the POD bases, so the snapshot matrix is still resident when the
    # audit artifact B = U^T F is formed.  The timed `fft_tight` subject below must reproduce
    # these fields bitwise (gate `timed_tight_matches_untimed_truth`), which is what makes this
    # the same object as the panel's same-grid reference.
    tight = cfg['same_grid_reference']
    ts = settings_of(cfg, tight)
    q_tight, pre_tight = ip.make_fom(L, ts['dt'], ts['preconditioner'])
    truth, truth_int = {}, {}
    for fname in report['families']:
        fs = []
        for case, phys in enumerate(cohorts[fname]):
            v = host(q_tight(jnp.asarray(e.initial(L, phys)), float(phys[4]),
                             ts['ntol'], ts['ltol'], *pre_tight))
            f = np.asarray(v[0])
            assert np.isfinite(f).all() and np.max(v[2]) <= ts['ntol'] * (1 + 1e-9), (fname, case)
            fs.append(f)
        truth[fname] = np.stack(fs)
        truth_int[fname] = np.ascontiguousarray(
            truth[fname][:, :, 1:-1, 1:-1].reshape(len(fs), len(TIMES), -1))
        report['families'][fname]['truth_sha256'] = sha_array(truth_int[fname])
        report['families'][fname]['truth_case_sha256'] = [sha_array(x) for x in truth[fname]]
    del q_tight, pre_tight
    jax.clear_caches()
    save()
    print('TRUTH', round(time.perf_counter() - begin, 1), flush=True)

    # ------------------------------------------------- snapshots and the POD bases --
    modes = {}
    for fname in report['families']:
        U, sinfo = LD.generate_snapshots(L, dt, trains[fname], cfg['train_state_stride'],
                                         cfg['snapshot_ntol'], cfg['snapshot_ltol'])
        assert sinfo['max_relative_residual'] <= cfg['snapshot_residual_bar'], (fname, sinfo)
        Ut = jnp.asarray(U.T)
        del U
        kmax = max(ranks)
        gram = Ut.T @ Ut
        w_all, Vg_all = jnp.linalg.eigh(gram)
        w_all, Vg_all = w_all[::-1], Vg_all[:, ::-1]
        V = pod_basis(Ut, kmax)[0]                       # the audited implementation
        Vg = np.asarray(Vg_all[:, :kmax])
        wk = np.asarray(w_all[:kmax])
        # parity of the two routes to the same modes, in-job
        Vown = Ut @ (jnp.asarray(Vg) / jnp.sqrt(jnp.clip(jnp.asarray(wk), 1e-300, None))[None, :])
        parity = float(jnp.max(jnp.abs(jnp.abs(Vown) - jnp.abs(V))))
        orth = float(jnp.max(jnp.abs(V.T @ V - jnp.eye(kmax))))
        GtV = np.asarray(gram @ jnp.asarray(Vg))
        # B = U^T F, the only piece of the audit route that needs the snapshot matrix alive
        B = np.asarray(Ut.T @ jnp.asarray(truth_int[fname].reshape(-1, Ut.shape[0]).T))
        modes[fname] = np.asarray(V)
        total = float(jnp.sum(jnp.clip(w_all, 0., None)))
        sinfo.update(pod_total_energy=total, pod_eigenvalues=wk.tolist(),
                     pod_singular_values=np.sqrt(np.clip(wk, 0., None)).tolist(),
                     pod_tail_fraction={str(k): float(max(total - float(np.sum(wk[:k])), 0.) / max(total, 1e-300))
                                        for k in ranks},
                     pod_mode_parity=parity, pod_orthonormality_deviation=orth,
                     gram_sha256=sha_array(np.asarray(gram)))
        np.savez_compressed(out / f'podaudit_{fname}.npz', gram_eigvecs=Vg, gram_eigvals=wk,
                            gram_times_eigvecs=GtV, snapshots_times_truth=B,
                            snapshot_count=np.int64(Ut.shape[1]))
        report['snapshots'][fname] = sinfo
        del Ut, gram, w_all, Vg_all, Vown, V
        jax.clear_caches()
        print('SNAPSHOTS', fname, round(sinfo['seconds'], 1), 'orth', f'{orth:.2e}', flush=True)
        save()

    # ------------------------------------------------------------ timed subjects --
    settings = {s['name']: s for s in cfg['fom_settings']}
    subjects = []
    for fname in report['families']:
        for s in cfg['fom_settings']:
            subjects.append(dict(name=f'{fname}__{s["name"]}__L{L}', family=fname,
                                 setting=s['name'], intervals=L, role='panel'))
    probe = cfg.get('mesh_probe') or {}
    for m in probe.get('meshes', []):
        for sn in probe.get('settings', []):
            for fname in probe.get('families', list(report['families'])):
                subjects.append(dict(name=f'{fname}__{sn}__L{m}', family=fname, setting=sn,
                                     intervals=int(m), role='mesh_probe'))
    report['declared_subjects'] = subjects
    save()

    foms = {}
    for sub in subjects:
        key = (sub['intervals'], settings[sub['setting']]['preconditioner'],
               settings[sub['setting']]['dt'])
        if key not in foms:
            foms[key] = ip.make_fom(key[0], key[2], key[1])
        sub['key'] = key
    inputs_u = {(f, m, c): e.initial(m, cohorts[f][c])
                for f in report['families'] for m in sorted({L, *probe.get('meshes', [])})
                for c in range(len(cohorts[f]))}

    def invoke(sub, u, case):
        fn, pre = foms[sub['key']]
        st = settings[sub['setting']]
        return fn(u, float(cohorts[sub['family']][case, 4]), st['ntol'], st['ltol'], *pre)

    t = time.perf_counter()
    for sub in subjects:
        jax.block_until_ready(invoke(sub, jnp.asarray(inputs_u[(sub['family'], sub['intervals'], 0)]), 0))
        print('WARM', sub['name'], round(time.perf_counter() - t, 1), flush=True)
    report['compile_warmup'] = dict(seconds=time.perf_counter() - t, subjects=len(subjects))
    save()

    order_rng = np.random.default_rng(cfg['order_seed'])
    artifacts, kept = {}, {}
    ncase = len(cohorts[fams[0]['name']])
    for rep in range(cfg['reps']):
        for case in range(ncase):
            for i in order_rng.permutation(len(subjects)):
                sub = subjects[int(i)]
                st = settings[sub['setting']]
                e.burn(cfg['burn_seconds'])
                ht = time.perf_counter()
                u = jax.device_put(np.array(inputs_u[(sub['family'], sub['intervals'], case)], copy=True))
                jax.block_until_ready(u)
                gt0 = time.perf_counter()
                value = invoke(sub, u, case)
                jax.block_until_ready(value)
                gs = time.perf_counter() - gt0
                f = np.asarray(value[0])
                hs = time.perf_counter() - ht
                v = host(value)
                assert np.isfinite(f).all(), sub['name']
                fdown = np.array(f[:, ::sub['intervals'] // OBS, ::sub['intervals'] // OBS], copy=True)
                h = sha_array(fdown)
                key = (sub['name'], case, h)
                if key not in artifacts:
                    nm = f'{sub["name"]}_case{case}_rep{rep}.npz'
                    np.savez_compressed(out / nm, fields=fdown)
                    artifacts[key] = nm
                if (sub['name'], case) not in kept:
                    kept[(sub['name'], case)] = fdown
                lr = np.asarray(v[3], dtype=float)
                used = lr[lr > 0.]
                report['invocations'].append(dict(
                    name=sub['name'], family=sub['family'], setting=sub['setting'],
                    role=sub['role'], intervals=sub['intervals'], case=case, rep=rep,
                    cohort=report['families'][sub['family']]['cohort_roles'][case],
                    nu=float(cohorts[sub['family']][case, 4]), gpu_seconds=gs, host_seconds=hs,
                    dt=st['dt'], ntol=st['ntol'], ltol=st['ltol'],
                    preconditioner=st['preconditioner'], field_sha256=h, artifact=artifacts[key],
                    error_vs_reference=e.errors(fdown, refs[(sub['family'], case)], OBS),
                    newton_iterations_total=int(np.sum(v[1])), max_newton_iterations=int(np.max(v[1])),
                    max_relative_newton_residual=float(np.max(v[2])),
                    nonlinear_converged=bool(np.max(v[2]) <= st['ntol'] * (1 + 1e-9)),
                    max_linear_relative_residual=float(np.max(used)) if used.size else 0.,
                    median_linear_relative_residual=float(np.median(used)) if used.size else 0.,
                    linear_solves=int(used.size)))
            save()
        print('TIMED', rep, round(time.perf_counter() - begin, 1), flush=True)

    # -------------------------------------------- same-grid errors and POD floors --
    same = []
    for fname in report['families']:
        for c in range(ncase):
            same.append(bool(np.array_equal(kept[(f'{fname}__{tight}__L{L}', c)], truth[fname][c])))
        for row in report['invocations']:
            if row['family'] == fname and row['intervals'] == L:
                row['error_same_grid'] = e.errors(kept[(row['name'], row['case'])],
                                                  truth[fname][row['case']], OBS)
    report['gates']['timed_tight_matches_untimed_truth'] = dict(
        passed=bool(all(same)), cases=len(same), bitwise_matches=int(sum(same)))
    for fname in report['families']:
        Fint, V = truth_int[fname], modes[fname]
        report['pod'][fname] = dict(
            ranks=ranks, floors=floors(Fint, V, ranks), truth_source=tight,
            truth_sha256=sha_array(Fint))
        np.savez_compressed(out / f'truth_{fname}.npz', fields=Fint,
                            projections=Fint.reshape(-1, Fint.shape[-1]) @ V)
        save()
        print('POD FLOORS', fname, {k: round(v['worst_all_times_percent'], 4)
                                    for k, v in report['pod'][fname]['floors'].items()}, flush=True)

    # -------------------------------- the published comparator for the incumbent family --
    cmp_floors = cfg.get('comparator_pod_best_found_percent') or {}
    if cmp_floors and 'incumbent' in report['pod']:
        got = report['pod']['incumbent']['floors']
        devs = {k: abs(got[k]['worst_all_times_percent'] - v) / max(abs(v), 1e-300)
                for k, v in cmp_floors.items() if k in got}
        report['gates']['incumbent_pod_floors_reproduce_bpn301'] = dict(
            passed=bool(devs and max(devs.values()) <= cfg.get('comparator_tolerance', 1e-3)),
            tolerance=cfg.get('comparator_tolerance', 1e-3),
            source=cfg.get('comparator_source'), expected=cmp_floors,
            got={k: got[k]['worst_all_times_percent'] for k in cmp_floors if k in got},
            max_relative_deviation=(max(devs.values()) if devs else None),
            per_rank_relative_deviation=devs)
        print('COMPARATOR', report['gates']['incumbent_pod_floors_reproduce_bpn301'], flush=True)
    save()

    # ------------------------------------------------------------------- summary --
    rows = {}
    for row in report['invocations']:
        rows.setdefault(row['name'], []).append(row)
    summary = []
    for name, rs in rows.items():
        r0 = rs[0]
        ms = sorted(1e3 * r['gpu_seconds'] for r in rs)
        ev = max(100. * r['error_vs_reference']['fixed_initial_max'] for r in rs)
        sg = ([100. * max(r['error_same_grid']['fixed_initial_max'] for r in rs)]
              if 'error_same_grid' in r0 else [None])
        sge = ([100. * max(max(r['error_same_grid']['fixed_initial_per_time'][1:]) for r in rs)]
               if 'error_same_grid' in r0 else [None])
        summary.append(dict(
            name=name, family=r0['family'], setting=r0['setting'], role=r0['role'],
            intervals=r0['intervals'], dt=r0['dt'], ntol=r0['ntol'], ltol=r0['ltol'],
            median_gpu_ms=float(np.median(ms)), all_gpu_ms=ms,
            worst_vs_reference_percent=ev,
            worst_all_times_percent=sg[0], worst_evolved_percent=sge[0],
            newton_iterations_total=int(max(r['newton_iterations_total'] for r in rs)),
            max_linear_relative_residual=float(max(r['max_linear_relative_residual'] for r in rs)),
            nonlinear_converged=bool(all(r['nonlinear_converged'] for r in rs))))
    report['subject_summary'] = summary
    # ----------------------------------------------------- the two hypothesis legs --
    if len(report['pod']) == 2:
        inc, low = report['pod']['incumbent']['floors'], report['pod']['lowvisc']['floors']
        report['leg_a_pod_degradation'] = {
            k: dict(rank=int(k), incumbent_percent=inc[k]['worst_all_times_percent'],
                    lowvisc_percent=low[k]['worst_all_times_percent'],
                    ratio=low[k]['worst_all_times_percent'] / max(inc[k]['worst_all_times_percent'], 1e-300),
                    incumbent_evolved_percent=inc[k]['worst_evolved_percent'],
                    lowvisc_evolved_percent=low[k]['worst_evolved_percent'],
                    evolved_ratio=low[k]['worst_evolved_percent'] / max(inc[k]['worst_evolved_percent'], 1e-300))
            for k in inc}
        by = {(s['family'], s['setting'], s['intervals']): s for s in summary}
        legb = {}
        for s in cfg['fom_settings']:
            a_, b_ = by.get(('incumbent', s['name'], L)), by.get(('lowvisc', s['name'], L))
            if a_ and b_:
                legb[s['name']] = dict(
                    incumbent_ms=a_['median_gpu_ms'], lowvisc_ms=b_['median_gpu_ms'],
                    cost_ratio=b_['median_gpu_ms'] / max(a_['median_gpu_ms'], 1e-300),
                    incumbent_worst_evolved_percent=a_['worst_evolved_percent'],
                    lowvisc_worst_evolved_percent=b_['worst_evolved_percent'],
                    incumbent_newton_iterations=a_['newton_iterations_total'],
                    lowvisc_newton_iterations=b_['newton_iterations_total'])
        report['leg_b_fom_cost'] = legb
        report['gates']['leg_a_pod_degrades'] = dict(
            bar=cfg['leg_a_bar'], rank=str(cfg['leg_a_rank']),
            ratio=report['leg_a_pod_degradation'][str(cfg['leg_a_rank'])]['ratio'],
            passed=bool(report['leg_a_pod_degradation'][str(cfg['leg_a_rank'])]['ratio'] >= cfg['leg_a_bar']))
        worst_cost = max((v['cost_ratio'] for v in legb.values()), default=0.)
        report['gates']['leg_b_fom_cost_rises'] = dict(
            bar=cfg['leg_b_bar'], max_cost_ratio=worst_cost,
            passed=bool(worst_cost >= cfg['leg_b_bar']))
    save()

    for fname in report['families']:
        panel = [s for s in summary if s['family'] == fname and s['role'] == 'panel'
                 and s['worst_evolved_percent'] is not None]
        report['families'][fname]['fom_frontier_same_grid'] = frontier(panel)
        report['families'][fname]['fom_frontier_vs_reference'] = frontier(
            panel, err='worst_vs_reference_percent')
    # ------------------------------------- the finer-mesh POD probe (guarded, last) --
    # Self-audit weakness 2: at L = 256 the low-viscosity family has more numerical than
    # physical viscosity in four of six cases, so a POD degradation measured there is entangled
    # with first-order-upwind diffusion.  Repeating the WHOLE leg-(a) measurement on a finer
    # mesh -- same 128 trajectories, same ranks, same floors, truth = that mesh's own tight
    # solve -- separates "the regime is physically sharper" from "the grid is too coarse".
    # It runs LAST and is wrapped, so an OOM or a failure here cannot void anything above.
    report['pod_probe'] = {}
    for m in cfg.get('pod_probe_meshes', []):
        m = int(m)
        try:
            t0 = time.perf_counter()
            qm, prem = ip.make_fom(m, ts['dt'], ts['preconditioner'])
            blk = {}
            for fname in report['families']:
                fs = []
                for phys in cohorts[fname]:
                    vv = host(qm(jnp.asarray(e.initial(m, phys)), float(phys[4]),
                                 ts['ntol'], ts['ltol'], *prem))
                    assert np.isfinite(vv[0]).all() and np.max(vv[2]) <= ts['ntol'] * (1 + 1e-9)
                    fs.append(np.asarray(vv[0]))
                Fm = np.ascontiguousarray(np.stack(fs)[:, :, 1:-1, 1:-1].reshape(len(fs), len(TIMES), -1))
                Um, si = LD.generate_snapshots(m, dt, trains[fname], cfg['train_state_stride'],
                                               cfg['snapshot_ntol'], cfg['snapshot_ltol'])
                Utm = jnp.asarray(Um.T)
                del Um
                Vm = pod_basis(Utm, max(ranks))[0]
                si['pod_orthonormality_deviation'] = float(
                    jnp.max(jnp.abs(Vm.T @ Vm - jnp.eye(max(ranks)))))
                blk[fname] = dict(snapshots=si, truth_sha256=sha_array(Fm),
                                  floors=floors(Fm, np.asarray(Vm), ranks))
                np.savez_compressed(out / f'podprobe_L{m}_{fname}.npz', truth=Fm,
                                    projections=Fm.reshape(-1, Fm.shape[-1]) @ np.asarray(Vm))
                del Utm, Vm
                jax.clear_caches()
                print('POD PROBE', m, fname,
                      {k: round(v['worst_all_times_percent'], 4) for k, v in blk[fname]['floors'].items()},
                      flush=True)
            if len(blk) == 2:
                blk['degradation'] = {
                    k: dict(rank=int(k),
                            incumbent_percent=blk['incumbent']['floors'][k]['worst_all_times_percent'],
                            lowvisc_percent=blk['lowvisc']['floors'][k]['worst_all_times_percent'],
                            ratio=blk['lowvisc']['floors'][k]['worst_all_times_percent']
                                  / max(blk['incumbent']['floors'][k]['worst_all_times_percent'], 1e-300))
                    for k in blk['incumbent']['floors']}
            blk['seconds'] = time.perf_counter() - t0
            report['pod_probe'][str(m)] = blk
            del qm, prem
            jax.clear_caches()
        except Exception as exc:                                     # noqa: BLE001
            report['pod_probe'][str(m)] = dict(failed=True, reason=str(exc)[:600])
            print('POD PROBE FAILED', m, str(exc)[:300], flush=True)
            jax.clear_caches()
        save()

    report['elapsed_seconds'] = time.perf_counter() - begin
    report['complete'] = True
    save()
    print('DONE', round(report['elapsed_seconds'], 1), flush=True)


if __name__ == '__main__':
    main()
