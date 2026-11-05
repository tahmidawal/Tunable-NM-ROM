"""G5 R5 paired timing (DESIGN-G5 G5-A5 item 6) in ONE allocation, CPU, 16 threads.

Workload: 0.1 s at dt0 from the first saved state of the held-out recorded window, outputs
identical for both methods: forces/probes every step, the state at every 1/400 save; the ROM
additionally reconstructs velocity on Q at every save (timed separately).
  FE  : transient.Stepper (P4/P3, frozen Oseen, production settings) from that FE state.
        cold = Stepper construction + state load + stepping; warm = stepping only (includes
        the 4-substep startup and the periodic LU refreshes, which belong to stepping).
  ROM : rom_core.Arm per arm. cold = operator/tensor build + rollout; warm = rollout only.
One discarded warm-up repetition per method immediately before the timed block; 7 paired
repetitions in alternating order (FE first on even repetitions). Saved: all repetition
arrays, result checksums (SHA-256 of output arrays) and work counts.
Usage: rom_r5.py --record-run <dir> --export <dir> --case <name> --manifest <json> --arms <json>
       [--trained-bank npz --trained-proj npz] --out <dir>
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

REPS, WINDOW = 7, 0.1


def digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


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
    cfg0 = json.loads((run / 'config.json').read_text())
    st = json.loads((run / 'stages.json').read_text())
    (rec,) = [s for s in st['stages'] if s['name'] == 'record']
    first = None
    for sn in st['snapshots']:
        z = np.load(run / sn['file'])
        i = np.where(np.abs(z['t'] - rec['t_start']) < 1e-9)[0]
        if len(i):
            first = (float(z['t'][i[0]]), z['x'][i[0]]); break
    if first is None:
        raise ValueError('record start state not found')
    t0, x0 = first
    state = out / 'start_state.npz'
    np.savez(state, x=x0, t=t0, nu=cfg0['nu'], velocity_order=cfg0['velocity_order'],
             mesh_sha256=cfg0['mesh_sha256'], layout_sha256=st['layout_sha256'])
    dt = cfg0['stages'][1]['dt']
    fe_cfg = dict(cfg0, initial={'kind': 'state', 'path': str(state), 'sha256': tr.sha(state)}, kick=None,
                  wall_limit_seconds=10 ** 6,
                  stages=[{'name': 'timing', 'dt': dt, 'duration': WINDOW, 'save_every_steps': int(round(1 / 400 / dt)),
                           'diagnostics_every_saves': 0}])
    ng.SetNumThreads(int(cfg0['threads']))
    man = json.loads(Path(a.manifest).read_text())
    (case,) = [c for c in man['cases'] if c['name'] == a.case]
    nu = 0.1 / case['Re']
    ex = rc.Export(a.export)
    proj = dict(np.load(Path(a.export) / f'proj_{a.case}.npz'))
    if abs(proj['t'][0] - t0) > 1e-9:
        raise ValueError('projection data does not start at the timing start state')
    if a.trained_bank:
        z = np.load(a.trained_bank); ex.add_bank('trained', z['V'], z['G'], z['outlet_V'])
        proj['uv_trained'] = np.load(a.trained_proj)['uv_trained']
    arms = json.loads(a.arms)
    n_steps = int(round(WINDOW / dt)); save_every = int(round(1 / 400 / dt))

    def fe_once(tag):
        d = out / f'fe_{tag}'
        t_cold = time.perf_counter()
        with ng.TaskManager():
            s = tr.Stepper(fe_cfg, d.mkdir(parents=True) or d, ng)
            s.run()
        cold = time.perf_counter() - t_cold
        sj = json.loads((d / 'stages.json').read_text())
        h = np.load(d / 'history.npz')['rows']
        return {'cold': cold, 'warm': sj['stages'][0]['wall_seconds'], 'steps': sj['stages'][0]['steps'],
                'factorizations': sj['factorizations'], 'checksum': digest(h[:, :8])}

    def rom_once(spec):
        t_cold = time.perf_counter()
        arm = rc.Arm(ex, spec['kind'], spec['r'], spec.get('key'))
        a0 = arm.ref_coeffs({k: v[:1] for k, v in proj.items()})[0][0]
        t_warm = time.perf_counter()
        A, C, F, P, its = arm.rollout(nu, dt, a0, n_steps)
        warm = time.perf_counter() - t_warm
        t_rec = time.perf_counter()
        fields = [arm.lV + arm.Vb @ A[k] for k in range(0, n_steps + 1, save_every)]
        recon = time.perf_counter() - t_rec
        return {'cold': time.perf_counter() - t_cold - recon, 'warm': warm, 'reconstruction': recon,
                'steps': n_steps, 'newton_iterations': int(its.sum()), 'dimension': arm.d,
                'checksum': digest(A, C, F, P), 'field_checksum': digest(np.asarray(fields))}

    res = {'workload': {'window_s': WINDOW, 'dt': dt, 'steps': n_steps, 'save_every_steps': save_every,
                        'start_time': t0, 'case': a.case}, 'threads': cfg0['threads'], 'fe': [], 'rom': {}}
    res['fe_warmup'] = fe_once('warmup')
    for spec in arms:
        res.setdefault('rom_warmup', {})[spec['name']] = rom_once(spec)
        res['rom'][spec['name']] = []
    for rep in range(REPS):
        order = ['fe', 'rom'] if rep % 2 == 0 else ['rom', 'fe']
        for m in order:
            if m == 'fe':
                res['fe'].append(fe_once(f'rep{rep}'))
            else:
                for spec in arms:
                    res['rom'][spec['name']].append(rom_once(spec))
        (out / 'R5-TIMING.json').write_text(json.dumps(res, indent=2) + '\n')
    med = lambda xs, k: float(np.median([x[k] for x in xs]))
    res['summary'] = {'fe_warm_median_s': med(res['fe'], 'warm'), 'fe_cold_median_s': med(res['fe'], 'cold'),
                      'rom': {n: {'warm_median_s': med(v, 'warm'), 'cold_median_s': med(v, 'cold'),
                                  'reconstruction_median_s': med(v, 'reconstruction'),
                                  'warm_speedup_vs_fe_warm': med(res['fe'], 'warm') / med(v, 'warm')}
                              for n, v in res['rom'].items()},
                      'checksums_identical_across_reps': {'fe': len({x['checksum'] for x in res['fe']}) == 1,
                                                          **{n: len({x['checksum'] for x in v}) == 1 for n, v in res['rom'].items()}}}
    (out / 'R5-TIMING.json').write_text(json.dumps(res, indent=2) + '\n')
    print(json.dumps(res['summary'], indent=1))


if __name__ == '__main__':
    main()
