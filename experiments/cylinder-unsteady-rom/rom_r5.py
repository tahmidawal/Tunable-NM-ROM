"""G5 R5 paired timing (DESIGN-G5 G5-A5 item 6, rev.3) in ONE allocation, CPU, 16 threads.

Workload: 0.1 s at dt0 from the first saved state of the held-out recorded window. Output
contract, identical for both methods: reaction forces and probe pressures every step and the
state at every 1/400 save (FE: coefficient vector; ROM: reduced coefficients). The ROM's
reconstruction of velocity on Q at every save is a separate, separately timed output.
  FE  : transient.Stepper in timing_mode (no surface-force integral / kinetic energy per step;
        snapshots buffered in memory, persistence after the timed loop and not charged).
        cold = construction + state load + stepping; warm = the stepping loop (includes the
        4-substep startup and the periodic LU refreshes, which belong to stepping).
  ROM : cold = operator/tensor build + projection of the start state + rollout; warm = rollout.
Before EVERY timed invocation an untimed warm-up of the same method runs immediately
(FE: 20 steps; ROM: 20 steps). 7 paired repetitions in alternating order. Every repetition
stores times, work counts (steps, LU refreshes, all Newton iterations incl. startup),
result checksums and an accuracy witness against the recorded reference at the same saves
(FE: velocity L2 relative to the record snapshots; ROM: phase-sensitive e_u).
Inputs are identity-bound: record-run snapshot SHAs, case/viscosity, projection start time,
trained-bank projection SHA.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'vendor' / 'ref'))
import rom_core as rc  # noqa: E402

REPS, WINDOW, WARM_STEPS = 7, 0.1, 20
TIME_KEYS = ('t', 'uu', 'pp', 'pv', 'pref')


def digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    for k in ('--record-run', '--export', '--case', '--manifest', '--arms', '--out'):
        ap.add_argument(k, required=True)
    ap.add_argument('--trained-bank'); ap.add_argument('--trained-proj')
    a = ap.parse_args()
    import ngsolve as ng
    import transient as tr
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    run = Path(a.record_run)
    man = json.loads(Path(a.manifest).read_text())
    (case,) = [c for c in man['cases'] if c['name'] == a.case]
    if Path(case['run_dir']).resolve() != run.resolve():
        raise ValueError('record run is not the manifest case')
    cfg0 = json.loads((run / 'config.json').read_text())
    nu = 0.1 / case['Re']
    if abs(cfg0['nu'] - nu) > 1e-15:
        raise ValueError('case viscosity differs from the record run')
    st = json.loads((run / 'stages.json').read_text())
    (rec,) = [s for s in st['stages'] if s['name'] == 'record']
    ref_t, ref_x = [], []
    for sn in st['snapshots']:
        if sha(run / sn['file']) != sn['sha256']:
            raise ValueError(f'snapshot hash mismatch {sn["file"]}')
        z = np.load(run / sn['file'])
        for t, x in zip(z['t'], z['x']):
            if rec['t_start'] - 1e-9 <= t <= rec['t_start'] + WINDOW + 1e-9:
                ref_t.append(float(t)); ref_x.append(x)
    ref_t = np.array(ref_t); ref_x = np.array(ref_x)
    t0, x0 = ref_t[0], ref_x[0]
    if abs(t0 - rec['t_start']) > 1e-9:
        raise ValueError('record start state not found')
    state = out / 'start_state.npz'
    np.savez(state, x=x0, t=t0, nu=cfg0['nu'], velocity_order=cfg0['velocity_order'],
             mesh_sha256=cfg0['mesh_sha256'], layout_sha256=st['layout_sha256'])
    dt = cfg0['stages'][1]['dt']
    n_steps = int(round(WINDOW / dt)); save_every = int(round(1 / 400 / dt))

    def fe_cfg(duration):
        return dict(cfg0, initial={'kind': 'state', 'path': str(state), 'sha256': sha(state)}, kick=None,
                    timing_mode=True, snapshot_chunk=10 ** 6, wall_limit_seconds=10 ** 6,
                    stages=[{'name': 'timing', 'dt': dt, 'duration': duration, 'save_every_steps': save_every,
                             'diagnostics_every_saves': 0}])
    ng.SetNumThreads(int(cfg0['threads']))
    ex = rc.Export(a.export)
    proj_all = dict(np.load(Path(a.export) / f'proj_{a.case}.npz'))
    if abs(proj_all['t'][0] - t0) > 1e-9:
        raise ValueError('projection data does not start at the timing start state')
    if a.trained_bank:
        z = np.load(a.trained_bank)
        ex.add_bank('trained', z['V'], z['G'], z['outlet_V'], {lab: (z[f'{lab}_V'], z[f'{lab}_G']) for lab in ('inlet', 'outlet')})
        tp = dict(np.load(a.trained_proj))
        if str(tp['bank_eval_sha256']) != sha(a.trained_bank) or not np.array_equal(tp['t'], proj_all['t']):
            raise ValueError('trained-bank projections are not bound to this bank evaluation')
        proj_all['uv_trained'] = tp['uv_trained']
    n_saves = n_steps // save_every + 1
    proj = {k: v[:n_saves] for k, v in proj_all.items() if k in TIME_KEYS or k.startswith('uv')}
    if not np.allclose(proj['t'], ref_t[:n_saves], atol=1e-9):
        raise ValueError('projection saves differ from the record saves')
    arms = json.loads(a.arms)
    counter = [0]

    def fe_once(steps, timed):
        counter[0] += 1
        d = out / f'fe_{counter[0]:03d}'; d.mkdir()
        t_cold = time.perf_counter()
        with ng.TaskManager():
            s = tr.Stepper(fe_cfg(steps * dt), d, ng)
            s.run()
        cold = time.perf_counter() - t_cold
        if not timed:
            return None
        sj = json.loads((d / 'stages.json').read_text())
        h = np.load(d / 'history.npz')['rows']
        xs = np.concatenate([np.load(d / f['file'])['x'] for f in sj['snapshots']])
        nv = s.V.ndof
        err = [np.sqrt(s.apply('M', xs[i] - ref_x[i]) @ (xs[i] - ref_x[i]) / (s.apply('M', ref_x[i]) @ ref_x[i]))
               for i in range(1, len(xs))]
        return {'cold': cold, 'warm': sj['stages'][0]['wall_seconds'], 'steps': sj['stages'][0]['steps'],
                'lu_refreshes': sj['factorizations'], 'checksum': digest(h[:, [0, 1, 2, 5, 6]], xs[:, :nv]),
                'accuracy_velocity_rel_L2_vs_record_max': float(max(err))}

    def rom_once(spec, steps, timed):
        t_cold = time.perf_counter()
        arm = rc.Arm(ex, spec['kind'], spec['r'], spec.get('key'))
        a0 = arm.ref_coeffs({k: v[:1] for k, v in proj.items()})[0][0]
        t_warm = time.perf_counter()
        A, C, F, P, its = arm.rollout(nu, dt, a0, steps)
        warm = time.perf_counter() - t_warm
        cold = time.perf_counter() - t_cold
        if not timed:
            return None
        t_rec = time.perf_counter()
        fields = np.asarray([arm.lV + arm.Vb @ A[k] for k in range(0, steps + 1, save_every)])
        recon = time.perf_counter() - t_rec
        idx = np.arange(0, steps + 1, save_every)
        err = arm.errors(proj, A, C, idx)['e_u']
        return {'cold': cold, 'warm': warm, 'reconstruction': recon, 'steps': steps,
                'newton_iterations_total': int(its.sum()), 'velocity_dimension': arm.d, 'pressure_dimension': arm.dp,
                'checksum': digest(A[idx], C[idx], F, P), 'field_checksum': digest(fields),
                'accuracy_e_u_max': float(err.max())}

    res = {'workload': {'window_s': WINDOW, 'dt': dt, 'steps': n_steps, 'save_every_steps': save_every,
                        'start_time': t0, 'case': a.case, 'warmup_steps': WARM_STEPS,
                        'output_contract': 'reaction forces + probes every step; state at every 1/400 save'},
           'threads': cfg0['threads'], 'fe': [], 'rom': {s['name']: [] for s in arms}}
    for rep in range(REPS):
        order = ['fe', 'rom'] if rep % 2 == 0 else ['rom', 'fe']
        for m in order:
            if m == 'fe':
                fe_once(WARM_STEPS, False)
                res['fe'].append(fe_once(n_steps, True))
            else:
                for spec in arms:
                    rom_once(spec, WARM_STEPS, False)
                    res['rom'][spec['name']].append(rom_once(spec, n_steps, True))
        (out / 'R5-TIMING.json').write_text(json.dumps(res, indent=2) + '\n')
    med = lambda xs, k: float(np.median([x[k] for x in xs]))
    res['summary'] = {'fe_warm_median_s': med(res['fe'], 'warm'), 'fe_cold_median_s': med(res['fe'], 'cold'),
                      'rom': {n: {'warm_median_s': med(v, 'warm'), 'cold_median_s': med(v, 'cold'),
                                  'reconstruction_median_s': med(v, 'reconstruction'),
                                  'warm_speedup_vs_fe_warm': med(res['fe'], 'warm') / med(v, 'warm'),
                                  'accuracy_e_u_max': v[0]['accuracy_e_u_max']} for n, v in res['rom'].items()},
                      'checksums_identical_across_reps': {'fe': len({x['checksum'] for x in res['fe']}) == 1,
                                                          **{n: len({x['checksum'] for x in v}) == 1 for n, v in res['rom'].items()}},
                      'scope': 'speed versus the production reference configuration only (no accuracy-matched FOM ladder)'}
    (out / 'R5-TIMING.json').write_text(json.dumps(res, indent=2) + '\n')
    print(json.dumps(res['summary'], indent=1))


if __name__ == '__main__':
    main()
