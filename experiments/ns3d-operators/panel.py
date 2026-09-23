"""One mesh, one allocation, one process: NM-ROM (accurate head k=8, fast span R'=16),
the CNAB2 step ladder and every trained neural operator on the evaluation cohort, then
A-B-A timing of all of them (DESIGN.md section 5).

Usage: panel.py --config configs/panel{n}.json --out output [--smoke]
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
              'experiments/ns3d-shift-head', 'experiments/ns3d-operators/deps/ns2d', 'experiments/ns3d-operators/deps/separable-decoder'):
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


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gate_ratio(num, den):
    r = float(num) / float(den)
    return r, bool(1.0 / GATE <= r <= GATE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.smoke:
        cfg.update(cfg.get('smoke', {}))
    out = args.out
    (out / 'fields').mkdir(parents=True, exist_ok=True)
    if jax.default_backend() != 'gpu':
        raise RuntimeError(f'need a GPU backend, got {jax.default_backend()}')
    import torch
    import ops3d as O
    env = O.configure()
    t_job = time.time()
    n, horizon, R = int(cfg['n']), float(cfg['horizon']), int(cfg['rank'])
    report = dict(schema='ns3d-operators-panel-v1', config=cfg, config_sha256=sha_file(args.config),
                  smoke=bool(args.smoke), torch_env=env, device=str(jax.devices()),
                  source_commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
                  files={p: sha_file(ROOT / p) for p in (
                      'experiments/ns3d/ns3d_fom.py', 'experiments/ns3d-shift/shift_rom.py',
                      'experiments/ns3d-shift-head/head_rom.py', 'experiments/ns3d-operators/ops3d.py',
                      'experiments/ns3d-operators/panel.py')})
    try:
        report['gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,memory.total',
                                                 '--format=csv,noheader'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        report['gpu'] = 'unavailable'

    def dump():
        report['elapsed_seconds'] = time.time() - t_job
        D.dump_json(out / 'summary.json', report)

    # ------------------------------------------------------------- 1. cohort truth
    ev_par = F.parameters(int(cfg['eval_seed']), int(cfg['eval_cases']))
    report['eval_parameter_sha256'] = D.sha256_array(ev_par)
    tr_rows = {tuple(r) for r in np.round(F.parameters(int(cfg['train_seed']), int(cfg['operator_train_cases'])), 9)}
    overlap = int(sum(tuple(r) in tr_rows for r in np.round(ev_par, 9)))
    report['eval_overlap_with_training_rows'] = overlap
    if overlap:
        raise RuntimeError('evaluation cohort overlaps the training draws')
    log('generating evaluation truth')
    truth, _, _ = D.generate(ev_par, n, float(cfg['dt_truth']), horizon)
    D.require_finite(truth, 'truth')
    ncase = len(truth)
    visc = ev_par[:, -1]
    u0n = np.sqrt(np.sum(truth[:, 0].reshape(ncase, -1) ** 2, axis=1))

    # ------------------------------------------------------------- 2. the NM-ROM
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
    Rp = int(cfg['span_rank'])
    Gr = G @ V
    span_args = (jnp.asarray(np.ascontiguousarray(Gr[:, :Rp])),
                 jnp.asarray(np.ascontiguousarray((ops['A'] @ V)[:, :Rp])),
                 jnp.asarray(np.ascontiguousarray(np.einsum('mjk,ja,kb->mab', ops['T'], V[:, :Rp], V[:, :Rp],
                                                            optimize=True))),
                 jnp.asarray(ops['lam']),
                 jnp.asarray(np.ascontiguousarray(np.einsum('dmr,rs->dms', ops['Dd'], V[:, :Rp]))))
    del Gr
    dtv, iters = float(cfg['rom_dt']), int(cfg['rom_iters'])
    steps = D.nsteps_for(dtv, horizon)
    head_run = H.make_head_run(dtv, steps, steps // 5, n, R, 8, 0, iters=iters, damping=float(cfg['damping']),
                               ic_iters=int(cfg['ic_iters']), frame='free')
    span_run = SR.make_frozen_run(dtv, steps, steps // 5, n, Rp, iters=iters, damping=float(cfg['damping']),
                                  extrapolate=True, diagnose=False)
    geom = F.geometry(n)

    # device-resident inputs for both frameworks
    u0_jax = [jnp.asarray(truth[c, 0]) for c in range(ncase)]
    u0_t = [torch.as_tensor(truth[c, 0], device='cuda')[None] for c in range(ncase)]
    nu_t = [torch.as_tensor([visc[c]], dtype=torch.float64, device='cuda') for c in range(ncase)]
    projector = O.Projector(n)

    arms = {}   # name -> dict(kind, call(case) -> device output, sync)

    def jsync(x):
        jax.block_until_ready(x)

    def tsync(x):
        torch.cuda.synchronize()

    arms['nmrom_accurate_head_k8'] = dict(kind='nmrom', sync=jsync, call=lambda c: head_run(
        u0_jax[c], float(visc[c]), *opsj, p, C0, Hc, Hn, Zj))
    arms[f'nmrom_fast_span{Rp}'] = dict(kind='nmrom', sync=jsync, call=lambda c: span_run(
        u0_jax[c], float(visc[c]), *span_args))
    for st in cfg['cnab2_steps']:
        st = int(st)
        solver = F.make_solver(horizon / st, st, st // 5)
        arms[f'cnab2_s{st}'] = dict(kind='cnab2', steps=st, sync=jsync,
                                    call=(lambda c, s_=solver: s_(u0_jax[c], float(visc[c]), geom)))
    op_meta = {}
    for op in cfg['operators']:
        path = Path(op['checkpoint'])
        digest = sha_file(path)
        if digest != op['sha256']:
            raise RuntimeError(f"checkpoint {op['arm']} hash {digest} != training record {op['sha256']}")
        model, norm, ckd = O.load_checkpoint(path)
        if int(ckd['mesh']) != n or ckd['arm'] != op['arm']:
            raise RuntimeError(f"checkpoint {op['arm']} is for mesh {ckd['mesh']} arm {ckd['arm']}")
        O.check_dtypes(model)
        name = f"op_{op['arm']}"
        op_meta[name] = dict(arm=op['arm'], family=op['family'], selected=bool(op['selected']), sha256=digest,
                             parameter_dtype=str(getattr(model, 'parameter_dtype', torch.float64)),
                             real_parameter_count=O.parameter_count(model), training=op['training'])

        def op_call(c, m_=model, nm_=norm):
            with torch.no_grad():
                return O.predict(m_, u0_t[c], nu_t[c], nm_, projector)
        arms[name] = dict(kind='operator', sync=tsync, call=op_call)
    report['operators'] = op_meta

    def to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().reshape(truth.shape[1:])
        return np.asarray(x).reshape(truth.shape[1:])

    # -------------------------------------------------------- 3. accuracy pass
    rng = np.random.default_rng(int(cfg['sample_seed']))
    sample_idx = np.sort(rng.choice(n ** 3, size=min(int(cfg['sample_points']), n ** 3), replace=False))
    np.save(out / 'fields' / 'sample_index.npy', sample_idx)

    def sample(field):  # (6,3,n,n,n) -> (6,3,S)
        return field.reshape(6, 3, -1)[:, :, sample_idx]

    np.save(out / 'fields' / 'truth_samples.npy', np.stack([sample(truth[c]) for c in range(ncase)]))
    results = {}
    acc_samples = {}
    saved_truth_cases = set()
    written = [0]
    cap = float(cfg.get('field_cap_gb', 4.0)) * 2 ** 30
    for name, arm in arms.items():
        t0 = time.time()
        errors = np.empty((ncase, 6))
        fields = {}
        samples = []
        finite = True
        for c in range(ncase):
            o = to_np(arm['call'](c))
            if not np.all(np.isfinite(o)):
                finite = False
                errors[c] = np.nan
                samples.append(np.full((6, 3, len(sample_idx)), np.nan))
                continue
            errors[c] = D.rel_rows(o, truth[c], truth[c, 0])
            chk = D.sumsq_rows(o, truth[c], truth[c, 0])
            if float(np.max(np.abs(errors[c] - chk))) > 1e-12 * max(1.0, float(np.max(np.abs(chk)))):
                raise RuntimeError('error reductions disagree')
            samples.append(sample(o))
            if c in tuple(cfg.get('full_cases', (0, 1))):
                fields[c] = o
        stats = D.stats_from_cases(errors, 0.05) if finite else None
        if finite:
            worst = int(np.argmax(errors[:, 1:].max(1)))
            if worst not in fields:
                fields[worst] = to_np(arm['call'](worst))
            saved_truth_cases |= set(fields)
            disk = shutil.disk_usage(out).free / 2 ** 30
            if disk < float(cfg['min_free_gb']):
                raise RuntimeError(f'only {disk:.0f} GB free; refusing to write fields')
            nbytes = 8 * (len(fields) * fields[0 if 0 in fields else next(iter(fields))].size
                          + len(samples) * samples[0].size)
            if written[0] + nbytes > cap:
                raise RuntimeError(f'saved-field cap {cap / 2**30:.1f} GB would be exceeded')
            written[0] += nbytes
            np.savez(out / 'fields' / f'{name}.npz', cases=np.asarray(sorted(fields)),
                     fields=np.stack([fields[c] for c in sorted(fields)]), samples=np.stack(samples))
        acc_samples[name] = np.stack(samples)
        results[name] = dict(kind=arm['kind'], finite=finite, stats=stats,
                             errors=errors.tolist(), seconds=time.time() - t0,
                             unstable=bool((not finite) or stats['evolved_worst'] > UNSTABLE))
        if 'steps' in arm:
            results[name]['steps'] = arm['steps']
        log(f"{name}: evolved worst {stats['evolved_worst'] if stats else float('nan'):.6f} "
            f"({time.time() - t0:.1f}s)")
    tcases = sorted(saved_truth_cases)
    np.savez(out / 'fields' / 'truth.npz', cases=np.asarray(tcases), fields=truth[tcases], u0_norm=u0n,
             viscosity=visc, parameters=ev_par)
    report['results'] = results
    dump()

    # reproduction gate against the frozen models' own development job
    repro = {}
    for name, refname in cfg['reference_errors'].items():
        if name not in results:
            continue
        got = np.asarray(results[name]['errors'])
        want = np.asarray(cfg['reference_values'][refname])
        rel = float(np.max(np.abs(got - want) / np.maximum(np.abs(want), 1e-9)))  # floor: t=0 errors are ~1e-16
        repro[name] = dict(reference=refname, max_relative_gap=rel, passed=bool(rel <= 1e-6))
    report['reproduction_gate'] = dict(arms=repro, passed=all(v['passed'] for v in repro.values()))
    log(f"reproduction gate: {report['reproduction_gate']['passed']} "
        f"{ {k: round(v['max_relative_gap'], 12) for k, v in repro.items()} }")
    dump()

    # ------------------------------------------------------------- 4. timing (A-B-A)
    names = [nm for nm, r in results.items() if r['finite']]
    reps = int(cfg['timing_rounds'])
    samples_log = []   # (phase, round, arm, case, predecessor, ms)
    last_out = {}

    def timed(name, c, phase, rnd, prev):
        arm = arms[name]
        torch.cuda.synchronize()
        start = time.perf_counter()
        o = arm['call'](c)
        arm['sync'](o)
        ms = (time.perf_counter() - start) * 1e3
        samples_log.append((phase, rnd, name, c, prev, ms))
        if phase == 'A2' and rnd == reps - 1:
            last_out[(name, c)] = o
        return o

    log(f'warm-up: {len(names)} arms')
    for name in names:
        for _ in range(int(cfg['burn_calls'])):
            for c in range(ncase):
                arms[name]['sync'](arms[name]['call'](c))
    order_rng = np.random.default_rng(int(cfg['timing_seed']))

    def burn(seconds=2.0):
        a = jnp.ones((1024, 1024)) * 1e-3
        f = jax.jit(lambda a: a @ a * 1e-3 + 1e-3)
        t = time.perf_counter()
        while time.perf_counter() - t < seconds:
            a = f(a)
        jax.block_until_ready(a)
    burn()
    prev = None
    for rnd in range(reps):                                   # A1
        for name in order_rng.permutation(names):
            for c in range(ncase):
                timed(name, c, 'A1', rnd, prev)
                prev = name
    burn()
    b_order = list(order_rng.permutation(names))              # B, arm-major
    for name in b_order:
        for rnd in range(reps):
            for c in range(ncase):
                timed(name, c, 'B', rnd, prev)
                prev = name
    burn()
    for rnd in range(reps):                                   # A2
        for name in order_rng.permutation(names):
            for c in range(ncase):
                o = timed(name, c, 'A2', rnd, prev)
                prev = name
                if rnd == reps - 1:
                    # outside the timed region: the timed output's errors vs the accuracy pass
                    e = D.rel_rows(to_np(o), truth[c], truth[c, 0])
                    ref_e = np.asarray(results[name]['errors'][c])
                    tol = 1e-4 if (arms[name]['kind'] == 'operator' and 'float32' in
                                   op_meta.get(name, {}).get('parameter_dtype', '')) else 1e-9
                    gapv = float(np.max(np.abs(e - ref_e) / np.maximum(np.abs(ref_e), 1e-12)))
                    # direct field parity on the fixed sample points (A3)
                    ts = sample(to_np(o))
                    ref_s = acc_samples[name][c]
                    fgap = float(np.max(np.abs(ts - ref_s)) / max(float(np.max(np.abs(ref_s))), 1e-300))
                    results[name]['timed_field_gap'] = max(results[name].get('timed_field_gap', 0.0), fgap)
                    gapv = max(gapv, fgap)
                    results[name].setdefault('timed_output_gap', 0.0)
                    results[name]['timed_output_gap'] = max(results[name]['timed_output_gap'], gapv)
                    results[name]['timed_output_tolerance'] = tol
    last_out.clear()
    raw = {}
    for phase, rnd, name, c, pv, ms in samples_log:
        raw.setdefault(name, dict(A1=[], B=[], A2=[], cases=[], phases=[], predecessor=[], ms=[]))
        raw[name][phase].append(ms)
        raw[name]['cases'].append(c)
        raw[name]['phases'].append(phase)
        raw[name]['predecessor'].append(pv)
        raw[name]['ms'].append(ms)
    timing = {}
    for name, r in raw.items():
        allms = r['A1'] + r['B'] + r['A2']
        drift, drift_ok = gate_ratio(np.median(r['A2']), np.median(r['A1']))
        order, order_ok = gate_ratio(np.median(r['B']), np.median(r['A1'] + r['A2']))
        timing[name] = dict(median_ms=float(np.median(allms)), min_ms=float(np.min(allms)),
                            max_ms=float(np.max(allms)), samples=len(allms),
                            median_A1=float(np.median(r['A1'])), median_B=float(np.median(r['B'])),
                            median_A2=float(np.median(r['A2'])), drift_ratio=drift, drift_passed=drift_ok,
                            order_ratio=order, order_passed=order_ok)
    # positive control: x1.15 on A2 must fail the drift gate
    ctrl = {name: gate_ratio(np.median(1.15 * np.asarray(r['A2'])), np.median(r['A1']))[1] for name, r in raw.items()}
    report['timing'] = dict(protocol='A1 interleaved / B arm-major / A2 interleaved', rounds=reps,
                            burn_calls=int(cfg['burn_calls']), arms=timing,
                            positive_control_all_failed=bool(not any(ctrl.values())), raw=raw)
    dump()

    # ------------------------------------------------------------- 5. FOM rule, speedups, gates
    acc = results['nmrom_accurate_head_k8']['stats']['evolved_worst']
    cands = [nm for nm, r in results.items() if r['kind'] == 'cnab2' and not r['unstable']
             and r['stats']['evolved_worst'] <= acc]
    if cands:
        fom = min(cands, key=lambda nm: timing[nm]['median_ms'])
        fom_ms = timing[fom]['median_ms']
        for nm in results:
            if nm in timing:
                results[nm]['speedup_vs_fom'] = fom_ms / timing[nm]['median_ms']
    else:
        fom = None
    # context: vs the fastest stable CNAB2 at least as accurate as each arm
    for nm, r in results.items():
        if not r['finite'] or nm not in timing:
            continue
        own = [c for c, rr in results.items() if rr['kind'] == 'cnab2' and not rr['unstable']
               and rr['stats']['evolved_worst'] <= r['stats']['evolved_worst']]
        if own:
            best = min(own, key=lambda c: timing[c]['median_ms'])
            r['matched_cnab2'] = best
            r['speedup_vs_matched_cnab2'] = timing[best]['median_ms'] / timing[nm]['median_ms']
    gated = ['nmrom_accurate_head_k8', f'nmrom_fast_span{Rp}'] + ([fom] if fom else []) + \
        [nm for nm, m in op_meta.items() if m['selected'] and nm in timing]
    timed_ok = all(results[nm].get('timed_output_gap', 0.0) <= results[nm].get('timed_output_tolerance', 1e-9)
                   for nm in names)
    report['fom_rule'] = dict(rule='fastest stable CNAB2 with evolved worst <= NM-ROM accurate', chosen=fom,
                              nmrom_accurate_evolved_worst=acc, candidates=cands)
    report['gates'] = dict(
        reproduction=report['reproduction_gate']['passed'],
        drift=all(timing[nm]['drift_passed'] for nm in gated),
        order=all(timing[nm]['order_passed'] for nm in gated),
        positive_control_failed_as_required=report['timing']['positive_control_all_failed'],
        timed_outputs_match=bool(timed_ok), gated_arms=gated,
        bank_rebuild=bool(gap <= 1e-8))
    report['gates']['coverage'] = all(
        len(raw[nm]['ms']) == 3 * reps * ncase and min(raw[nm]['ms']) > 0 for nm in names)
    report['gates']['finite_reported_arms'] = all(results[nm]['finite'] for nm in gated)
    report['gates']['fom_found'] = fom is not None
    report['gates']['all_passed'] = all(v for k, v in report['gates'].items() if k != 'gated_arms')
    report['status'] = 'final' if report['gates']['all_passed'] else 'PROVISIONAL (diagnostic only)'
    report['saved_field_bytes'] = written[0]
    report['results'] = results
    report['complete'] = True
    dump()
    log(f"FOM {fom}; gates {report['gates']}")
    for nm in gated:
        log(f"  {nm}: {results[nm]['stats']['evolved_worst']:.5f} {timing[nm]['median_ms']:.3f} ms "
            f"x{results[nm].get('speedup_vs_fom', float('nan')):.3f}")


if __name__ == '__main__':
    main()
