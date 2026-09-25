"""NS 3D Table-1 rows on the held-out TEST cohort at one mesh (DESIGN.md).

One allocation, one process: rebuild the frozen development model of `a2_h{n}` (bank from the
training seed, gated against its stored probe; head k=8 and importance rotation loaded), then

1. development cohort (seed 202609202, 16 cases): accuracy pass of every NM-ROM arm and every
   CNAB2 setting -> reproduction gate against the `a2_h{n}` summary (the source of the paper's
   development rows), <= 1e-6 relative;
2. test cohort (seed 202609221, 32 cases, disjointness-checked): accuracy pass with saved
   fields for the independent audit;
3. A-B-A timing of every arm on the test cohort (the reported numbers), then the same protocol on
   the development cohort (context: same-job, same-protocol development ratios);
4. Table-1 FOM rule per cohort, speedups, gates.

Derived from experiments/ns3d-operators/panel.py (operators removed, arm list configurable).
Usage: test_panel.py --config configs/test{n}.json --out output [--smoke]
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for extra in ('experiments/ns3d', 'experiments/ns3d-grok', 'experiments/ns3d-shift',
              'experiments/ns3d-shift-head', 'experiments/ns3d-operators/deps/ns2d',
              'experiments/ns3d-operators/deps/separable-decoder'):
    sys.path.insert(0, str(ROOT / extra))
sys.path.insert(0, str(HERE))
os.environ.setdefault('JAX_ENABLE_X64', '1')

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update('jax_enable_x64', True)
jax.config.update('jax_default_matmul_precision', 'highest')

import ns3d_fom as F  # noqa: E402
import diag_floor as D  # noqa: E402
import shift_rom as SR  # noqa: E402
import head_rom as H  # noqa: E402

UNSTABLE = 1.0
GATE = 1.10
ACC = 'nmrom_accurate_head_k8'


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gate_ratio(num, den):
    r = float(num) / float(den)
    return r, bool(1.0 / GATE <= r <= GATE)


def rows(seed, count):
    return {tuple(r) for r in np.round(F.parameters(int(seed), int(count)), 9)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.smoke:
        cfg.update(cfg.get('smoke', {}))
        if int(cfg['test_seed']) == 202609221:
            raise RuntimeError('a smoke run must never open the test cohort')
    out = args.out
    (out / 'fields').mkdir(parents=True, exist_ok=True)
    if jax.default_backend() != 'gpu':
        raise RuntimeError(f'need a GPU backend, got {jax.default_backend()}')
    t_job = time.time()
    n, horizon, R = int(cfg['n']), float(cfg['horizon']), int(cfg['rank'])
    report = dict(schema='ns3d-test-panel-v1', config=cfg, config_sha256=sha_file(args.config),
                  smoke=bool(args.smoke), device=str(jax.devices()),
                  source_commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
                  jax_version=jax.__version__,
                  matmul_precision=os.environ.get('JAX_DEFAULT_MATMUL_PRECISION'),
                  files={p: sha_file(ROOT / p) for p in (
                      'experiments/ns3d/ns3d_fom.py', 'experiments/ns3d-grok/diag_floor.py',
                      'experiments/ns3d-shift/shift_rom.py', 'experiments/ns3d-shift-head/head_rom.py',
                      'experiments/ns3d-test/test_panel.py')})
    try:
        report['gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,memory.total',
                                                 '--format=csv,noheader'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        report['gpu'] = 'unavailable'

    def dump():
        report['elapsed_seconds'] = time.time() - t_job
        D.dump_json(out / 'summary.json', report)

    # ------------------------------------------------------------- cohorts
    dev_par = F.parameters(int(cfg['dev_seed']), int(cfg['dev_cases']))
    test_par = F.parameters(int(cfg['test_seed']), int(cfg['test_cases']))
    report['dev_parameter_sha256'] = D.sha256_array(dev_par)
    report['test_parameter_sha256'] = D.sha256_array(test_par)
    test_rows = [tuple(r) for r in np.round(test_par, 9)]
    disjoint = {}
    against = [(int(cfg['train_seed']), int(cfg['head_train_cases']))] + \
              ([] if args.smoke else [(int(cfg['dev_seed']), int(cfg['dev_cases']))]) + \
              [(int(s), int(c)) for s, c in cfg['closed_seeds']]
    for seed, count in against:
        other = rows(seed, count)
        disjoint[f'{seed}x{count}'] = int(sum(r in other for r in test_rows))
    report['test_overlap_rows'] = disjoint
    if any(disjoint.values()):
        raise RuntimeError(f'test cohort overlaps another cohort: {disjoint}')
    train_rows = rows(cfg['train_seed'], cfg['head_train_cases'])
    dev_overlap = int(sum(tuple(r) in train_rows for r in np.round(dev_par, 9)))
    report['dev_overlap_with_training_rows'] = dev_overlap
    if dev_overlap:
        raise RuntimeError('development cohort overlaps training')

    # ------------------------------------------------------------- frozen model
    frozen = ROOT / cfg['frozen_dir']
    fsha = {name: sha_file(frozen / name) for name in ('head_k8.npz', 'rotation.npz', 'bank_probe.npz')}
    report['frozen_sha256'] = fsha
    for name, value in cfg['frozen_sha256'].items():
        if fsha[name] != value:
            raise RuntimeError(f'frozen file {name} changed')
    train_par = F.parameters(int(cfg['train_seed']), int(cfg['train_cases']))
    log('generating bank training trajectories')
    train, _, _ = D.generate(train_par, n, float(cfg['dt_truth']), horizon)
    centre_jit = jax.jit(lambda f: SR.shift_field(f, -SR.grid_centroid(f) * n))
    for c in range(len(train)):
        for t in range(train.shape[1]):
            train[c, t] = np.asarray(centre_jit(jnp.asarray(train[c, t])))
    basis_all, _, _ = D.pod_basis(train.reshape(len(train) * train.shape[1], -1), R, int(cfg['gram_block']))
    del train
    gc.collect()
    G, _ = D.orthonormalize_prefix(basis_all, R)
    G = np.ascontiguousarray(G)
    del basis_all
    ref = np.load(frozen / 'bank_probe.npz')
    probe, _, _ = D.generate(F.parameters(int(cfg['train_seed']), int(ref['cases'])), n,
                             float(cfg['dt_truth']), horizon)
    a_new = np.stack([G.T @ np.asarray(centre_jit(jnp.asarray(probe[c, t]))).ravel()
                      for c in range(len(probe)) for t in range(probe.shape[1])])
    signs = np.sign(np.sum(a_new * ref['coefficients'], axis=0))
    G = G * signs[None, :]
    gap = float(np.linalg.norm(a_new * signs[None, :] - ref['coefficients']) / np.linalg.norm(ref['coefficients']))
    report['bank_rebuild_gap'] = gap
    log(f'bank rebuild gap {gap:.3e}')
    if gap > 1e-8:
        raise RuntimeError(f'rebuilt bank differs from the frozen one: {gap}')
    del probe
    ops, rep = SR.build_operators_fast(G, n, int(cfg['modes']), check_modes=int(cfg['check_modes']))
    report['operator_checks'] = rep
    Gj = jnp.asarray(G)
    opsj = (Gj, jnp.asarray(ops['A']), jnp.asarray(ops['T']), jnp.asarray(ops['lam']), jnp.asarray(ops['Dd']))
    ck = np.load(frozen / 'head_k8.npz')
    p = {key: jnp.asarray(ck[f'p_{key}']) for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3', 'Ws')}
    Zj = jnp.asarray(ck['Z'])
    Hc = H.head_apply(p, Zj)
    Hn = jnp.sum(Hc * Hc, 1)
    C0 = jnp.zeros((R, 0))
    V = np.asarray(np.load(frozen / 'rotation.npz')['V'])
    Gr = G @ V
    span_args = {}
    for Rp in sorted({int(a['rank']) for a in cfg['arms'] if a['kind'] == 'span'}):
        span_args[Rp] = (jnp.asarray(np.ascontiguousarray(Gr[:, :Rp])),
                         jnp.asarray(np.ascontiguousarray((ops['A'] @ V)[:, :Rp])),
                         jnp.asarray(np.ascontiguousarray(np.einsum('mjk,ja,kb->mab', ops['T'], V[:, :Rp],
                                                                    V[:, :Rp], optimize=True))),
                         jnp.asarray(ops['lam']),
                         jnp.asarray(np.ascontiguousarray(np.einsum('dmr,rs->dms', ops['Dd'], V[:, :Rp]))))
    del Gr
    geom = F.geometry(n)
    damping, ic_iters = float(cfg['damping']), int(cfg['ic_iters'])

    # ------------------------------------------------------------- arms: name -> call(u0_jax, nu)
    arms = {}
    for a in cfg['arms']:
        dtv, iters = float(a['dt']), int(a['iters'])
        steps = D.nsteps_for(dtv, horizon)
        if a['kind'] == 'head':
            run = H.make_head_run(dtv, steps, steps // 5, n, R, int(a['k']), 0, iters=iters, damping=damping,
                                  ic_iters=ic_iters, frame='free')
            arms[a['name']] = dict(kind='nmrom', spec=a, call=(lambda u, nu, r_=run: r_(
                u, nu, *opsj, p, C0, Hc, Hn, Zj)))
        elif a['kind'] == 'span':
            Rp = int(a['rank'])
            run = SR.make_frozen_run(dtv, steps, steps // 5, n, Rp, iters=iters, damping=damping,
                                     extrapolate=True, diagnose=False)
            arms[a['name']] = dict(kind='nmrom', spec=a, call=(lambda u, nu, r_=run, a_=span_args[Rp]: r_(
                u, nu, *a_)))
        else:
            raise ValueError(a)
    for st in cfg['cnab2_steps']:
        st = int(st)
        solver = F.make_solver(horizon / st, st, st // 5)
        arms[f'cnab2_s{st}'] = dict(kind='cnab2', steps=st, spec=dict(steps=st),
                                    call=(lambda u, nu, s_=solver: s_(u, nu, geom)))
    if ACC not in arms:
        raise RuntimeError('config has no accurate arm')

    def accuracy(truth, visc, save, tag):
        ncase = len(truth)
        u0j = [jnp.asarray(truth[c, 0]) for c in range(ncase)]
        results, samples_all = {}, {}
        if save:
            rng = np.random.default_rng(int(cfg['sample_seed']))
            sidx = np.sort(rng.choice(n ** 3, size=min(int(cfg['sample_points']), n ** 3), replace=False))
            np.save(out / 'fields' / 'sample_index.npy', sidx)
            np.save(out / 'fields' / 'truth_samples.npy',
                    np.stack([truth[c].reshape(6, 3, -1)[:, :, sidx] for c in range(ncase)]))
        saved_truth, written = set(), 0
        cap = float(cfg.get('field_cap_gb', 4.0)) * 2 ** 30
        for name, arm in arms.items():
            t0 = time.time()
            errors = np.empty((ncase, 6))
            fields, samples, finite = {}, [], True
            for c in range(ncase):
                o = np.asarray(arm['call'](u0j[c], float(visc[c]))).reshape(truth.shape[1:])
                if not np.all(np.isfinite(o)):
                    finite = False
                    errors[c] = np.nan
                    if save:
                        samples.append(np.full((6, 3, len(sidx)), np.nan))
                    continue
                errors[c] = D.rel_rows(o, truth[c], truth[c, 0])
                chk = D.sumsq_rows(o, truth[c], truth[c, 0])
                if float(np.max(np.abs(errors[c] - chk))) > 1e-12 * max(1.0, float(np.max(np.abs(chk)))):
                    raise RuntimeError('error reductions disagree')
                if save:
                    samples.append(o.reshape(6, 3, -1)[:, :, sidx])
                    if c in tuple(cfg.get('full_cases', ())):
                        fields[c] = o
            stats = D.stats_from_cases(errors, 0.05) if finite else None
            if save and finite:
                worst = int(np.argmax(errors[:, 1:].max(1)))
                if worst not in fields:
                    fields[worst] = np.asarray(arm['call'](u0j[worst], float(visc[worst]))).reshape(truth.shape[1:])
                saved_truth |= set(fields)
                free = shutil.disk_usage(out).free / 2 ** 30
                if free < float(cfg['min_free_gb']):
                    raise RuntimeError(f'only {free:.0f} GB free; refusing to write fields')
                nbytes = 8 * (len(fields) * next(iter(fields.values())).size + len(samples) * samples[0].size)
                if written + nbytes > cap:
                    raise RuntimeError(f'saved-field cap {cap / 2**30:.1f} GB would be exceeded')
                written += nbytes
                np.savez(out / 'fields' / f'{name}.npz', cases=np.asarray(sorted(fields)),
                         fields=np.stack([fields[c] for c in sorted(fields)]), samples=np.stack(samples))
            if save:
                samples_all[name] = np.stack(samples)
            results[name] = dict(kind=arm['kind'], spec=arm['spec'], finite=finite, stats=stats,
                                 errors=errors.tolist(), seconds=time.time() - t0,
                                 unstable=bool((not finite) or stats['evolved_worst'] > UNSTABLE))
            if 'steps' in arm:
                results[name]['steps'] = arm['steps']
            log(f"[{tag}] {name}: evolved worst {stats['evolved_worst'] if stats else float('nan'):.6f} "
                f"({time.time() - t0:.1f}s)")
        if save:
            tcases = sorted(saved_truth)
            u0n = np.sqrt(np.sum(truth[:, 0].reshape(ncase, -1) ** 2, axis=1))
            np.savez(out / 'fields' / 'truth.npz', cases=np.asarray(tcases), fields=truth[tcases], u0_norm=u0n,
                     viscosity=visc, parameters=test_par)
            return results, samples_all, sidx, written
        return results, None, None, 0

    # ------------------------------------------------------------- 1. development: reproduction gate
    log('generating development truth')
    dev_truth, _, _ = D.generate(dev_par, n, float(cfg['dt_truth']), horizon)
    D.require_finite(dev_truth, 'dev truth')
    dev_results, _, _, _ = accuracy(dev_truth, dev_par[:, -1], False, 'dev')
    repro = {}
    for name, refname in cfg['reference_errors'].items():
        if name not in dev_results:  # only in --smoke (shortened CNAB2 grid)
            continue
        got = np.asarray(dev_results[name]['errors'])
        want = np.asarray(cfg['reference_values'][refname])[:len(got)]
        rel = float(np.max(np.abs(got - want) / np.maximum(np.abs(want), 1e-9)))  # t=0 errors ~1e-16
        repro[name] = dict(reference=refname, max_relative_gap=rel, passed=bool(rel <= 1e-6))
    missing = [nm for nm in arms if nm not in repro]
    report['reproduction_gate'] = dict(arms=repro, arms_without_reference=missing,
                                       passed=bool(all(v['passed'] for v in repro.values()) and not missing))
    report['dev'] = dict(results=dev_results)
    log(f"reproduction gate: {report['reproduction_gate']['passed']} max gap "
        f"{max(v['max_relative_gap'] for v in repro.values()):.3e}; missing {missing}")
    dump()

    # ------------------------------------------------------------- 2. test cohort
    log('generating TEST truth (seed %d, %d cases)' % (int(cfg['test_seed']), len(test_par)))
    truth, _, _ = D.generate(test_par, n, float(cfg['dt_truth']), horizon)
    D.require_finite(truth, 'test truth')
    report['test_opened'] = dict(seed=int(cfg['test_seed']), cases=len(test_par), smoke=bool(args.smoke))
    results, acc_samples, sidx, written = accuracy(truth, test_par[:, -1], True, 'test')
    report['results'] = results
    report['saved_field_bytes'] = written
    dump()

    # ------------------------------------------------------------- 3. timing (A-B-A), per cohort
    def burn(seconds=2.0):
        a = jnp.ones((1024, 1024)) * 1e-3
        f = jax.jit(lambda a: a @ a * 1e-3 + 1e-3)
        t = time.perf_counter()
        while time.perf_counter() - t < seconds:
            a = f(a)
        jax.block_until_ready(a)

    def timing_block(tr, visc, res, seed, parity):
        ncase = len(tr)
        u0j = [jnp.asarray(tr[c, 0]) for c in range(ncase)]
        names = [nm for nm, r in res.items() if r['finite']]
        reps = int(cfg['timing_rounds'])
        log_ = []

        def timed(name, c, phase, rnd, prev):
            start = time.perf_counter()
            o = arms[name]['call'](u0j[c], float(visc[c]))
            jax.block_until_ready(o)
            log_.append((phase, rnd, name, c, prev, (time.perf_counter() - start) * 1e3))
            return o

        for name in names:  # warm-up (compiled already by the accuracy pass)
            for _ in range(int(cfg['burn_calls'])):
                for c in range(ncase):
                    jax.block_until_ready(arms[name]['call'](u0j[c], float(visc[c])))
        order_rng = np.random.default_rng(int(seed))
        burn()
        prev = None
        for rnd in range(reps):                               # A1
            for name in order_rng.permutation(names):
                for c in range(ncase):
                    timed(name, c, 'A1', rnd, prev)
                    prev = name
        burn()
        for name in list(order_rng.permutation(names)):       # B, arm-major
            for rnd in range(reps):
                for c in range(ncase):
                    timed(name, c, 'B', rnd, prev)
                    prev = name
        burn()
        for rnd in range(reps):                               # A2
            for name in order_rng.permutation(names):
                for c in range(ncase):
                    o = timed(name, c, 'A2', rnd, prev)
                    prev = name
                    if parity and rnd == reps - 1:           # outside the timed region
                        of = np.asarray(o).reshape(tr.shape[1:])
                        e = D.rel_rows(of, tr[c], tr[c, 0])
                        ref_e = np.asarray(res[name]['errors'][c])
                        gapv = float(np.max(np.abs(e - ref_e) / np.maximum(np.abs(ref_e), 1e-12)))
                        ts = of.reshape(6, 3, -1)[:, :, sidx]
                        ref_s = acc_samples[name][c]
                        fgap = float(np.max(np.abs(ts - ref_s)) / max(float(np.max(np.abs(ref_s))), 1e-300))
                        res[name]['timed_field_gap'] = max(res[name].get('timed_field_gap', 0.0), fgap)
                        res[name]['timed_output_gap'] = max(res[name].get('timed_output_gap', 0.0), gapv, fgap)
                    del o
        raw = {}
        for phase, rnd, name, c, pv, ms in log_:
            raw.setdefault(name, dict(A1=[], B=[], A2=[], cases=[], phases=[], predecessor=[], ms=[]))
            raw[name][phase].append(ms)
            raw[name]['cases'].append(c)
            raw[name]['phases'].append(phase)
            raw[name]['predecessor'].append(pv)
            raw[name]['ms'].append(ms)
        tim = {}
        for name, r in raw.items():
            allms = r['A1'] + r['B'] + r['A2']
            drift, drift_ok = gate_ratio(np.median(r['A2']), np.median(r['A1']))
            order, order_ok = gate_ratio(np.median(r['B']), np.median(r['A1'] + r['A2']))
            tim[name] = dict(median_ms=float(np.median(allms)), min_ms=float(np.min(allms)),
                             max_ms=float(np.max(allms)), samples=len(allms),
                             median_A1=float(np.median(r['A1'])), median_B=float(np.median(r['B'])),
                             median_A2=float(np.median(r['A2'])), drift_ratio=drift, drift_passed=drift_ok,
                             order_ratio=order, order_passed=order_ok)
        ctrl = {nm: gate_ratio(np.median(1.15 * np.asarray(r['A2'])), np.median(r['A1']))[1] for nm, r in raw.items()}
        coverage = all(len(raw[nm]['ms']) == 3 * reps * ncase and min(raw[nm]['ms']) > 0 for nm in names)
        return dict(protocol='A1 interleaved / B arm-major / A2 interleaved', rounds=reps,
                    burn_calls=int(cfg['burn_calls']), arms=tim, coverage=bool(coverage),
                    positive_control_all_failed=bool(not any(ctrl.values())), raw=raw), names

    def fom_rule(res, tim):
        acc = res[ACC]['stats']['evolved_worst']
        cands = [nm for nm, r in res.items() if r['kind'] == 'cnab2' and not r['unstable']
                 and r['stats']['evolved_worst'] <= acc]
        fom = min(cands, key=lambda nm: tim[nm]['median_ms']) if cands else None
        if fom:
            for nm in res:
                if nm in tim:
                    res[nm]['speedup_vs_fom'] = tim[fom]['median_ms'] / tim[nm]['median_ms']
        for nm, r in res.items():
            if not r['finite'] or nm not in tim:
                continue
            own = [c for c, rr in res.items() if rr['kind'] == 'cnab2' and not rr['unstable']
                   and rr['stats']['evolved_worst'] <= r['stats']['evolved_worst']]
            if own:
                best = min(own, key=lambda c: tim[c]['median_ms'])
                r['matched_cnab2'] = best
                r['speedup_vs_matched_cnab2'] = tim[best]['median_ms'] / tim[nm]['median_ms']
        return dict(rule='fastest stable CNAB2 (all finite, evolved worst <= 100 %) with evolved worst <= '
                         'NM-ROM accurate (head k=8)', chosen=fom, nmrom_accurate_evolved_worst=acc,
                    candidates=cands)

    log('timing: TEST cohort')
    test_timing, names = timing_block(truth, test_par[:, -1], results, int(cfg['timing_seed']), True)
    report['timing'] = test_timing
    report['fom_rule'] = fom_rule(results, test_timing['arms'])
    dump()
    log('timing: development cohort (context)')
    dev_timing, _ = timing_block(dev_truth, dev_par[:, -1], dev_results, int(cfg['timing_seed']) + 1, False)
    report['dev']['timing'] = dev_timing
    report['dev']['fom_rule'] = fom_rule(dev_results, dev_timing['arms'])

    # ------------------------------------------------------------- 4. gates
    fom = report['fom_rule']['chosen']
    gated = list(cfg['gated_arms']) + ([fom] if fom else [])
    tol = 1e-9
    for nm in names:
        results[nm]['timed_output_tolerance'] = tol
    report['gates'] = dict(
        reproduction=report['reproduction_gate']['passed'],
        bank_rebuild=bool(gap <= 1e-8),
        drift=all(test_timing['arms'][nm]['drift_passed'] for nm in gated),
        order=all(test_timing['arms'][nm]['order_passed'] for nm in gated),
        positive_control_failed_as_required=test_timing['positive_control_all_failed'],
        timed_outputs_match=bool(all(results[nm].get('timed_output_gap', np.inf) <= tol for nm in names)),
        coverage=test_timing['coverage'],
        finite_reported_arms=all(results[nm]['finite'] for nm in gated),
        fom_found=fom is not None,
        gated_arms=gated)
    report['gates']['all_passed'] = all(v for k, v in report['gates'].items() if k != 'gated_arms')
    report['dev']['gates'] = dict(
        drift=all(dev_timing['arms'][nm]['drift_passed'] for nm in gated if nm in dev_timing['arms']),
        order=all(dev_timing['arms'][nm]['order_passed'] for nm in gated if nm in dev_timing['arms']),
        note='context only; the development gate that matters is the reproduction gate')
    report['status'] = 'final' if report['gates']['all_passed'] else 'PROVISIONAL (diagnostic only)'
    report['complete'] = True
    dump()
    log(f"FOM {fom}; gates {report['gates']}")
    for nm in list(cfg['gated_arms']) + ([fom] if fom else []):
        log(f"  {nm}: test {results[nm]['stats']['evolved_worst']:.5f} {test_timing['arms'][nm]['median_ms']:.3f} ms "
            f"x{results[nm].get('speedup_vs_fom', float('nan')):.3f} | dev "
            f"{dev_results[nm]['stats']['evolved_worst']:.5f} x{dev_results[nm].get('speedup_vs_fom', float('nan')):.3f}")


if __name__ == '__main__':
    main()
