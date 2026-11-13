"""R4 animation frames (DESIGN-G5 5.4 products): reference vs ROM velocity magnitude on Q.

Evaluates, at one quadrature point per element (every `stride`-th point of the export rule),
the reference velocity of a DEVELOPMENT trajectory at every 4th save over the last two periods,
and each arm's reconstructed ROM velocity from its saved R4 coefficient series.
Usage: rom_frames.py --export <dir> --manifest <json> --case re95 --series-dir <r4 out>
       --arms <json> [--trained-bank npz] --out frames.npz
"""
import argparse
import json
from pathlib import Path

import numpy as np

import rom_core as rc
import rom_export as rx


def main():
    ap = argparse.ArgumentParser()
    for k in ('--export', '--manifest', '--case', '--series-dir', '--arms', '--out'):
        ap.add_argument(k, required=True)
    ap.add_argument('--trained-bank'); ap.add_argument('--stride', type=int, default=49)  # order-12 rule: 49 points per triangle
    a = ap.parse_args()
    import ngsolve as ng
    man = json.loads(Path(a.manifest).read_text())
    (case,) = [c for c in man['cases'] if c['name'] == a.case]
    if case['split'] != 'dev':
        raise PermissionError('frames only for the development trajectory before R5')
    ex = rc.Export(a.export)
    if a.trained_bank:
        z = np.load(a.trained_bank)
        ex.add_bank('trained', z['V'], z['G'], z['outlet_V'], {lab: (z[f'{lab}_V'], z[f'{lab}_G']) for lab in ('inlet', 'outlet')})
    sel = np.arange(0, len(ex.w), a.stride)
    pts = np.load(Path(a.export) / 'quadrature.npz')['points'][sel]
    out = {'points': pts}
    arms = json.loads(a.arms)
    first = np.load(Path(a.series_dir) / f"series_{a.case}_{arms[0]['name']}.npz")
    ts = first['t_save']
    period = 0.333
    frames = np.where(ts >= ts[-1] - 2 * period)[0][::4]
    out['t'] = ts[frames]
    for spec in arms:
        arm = rc.Arm(ex, spec['kind'], spec['r'], spec.get('key'))
        s = np.load(Path(a.series_dir) / f"series_{a.case}_{spec['name']}.npz")
        U = arm.lV[sel][None] + np.einsum('qcj,tj->tqc', arm.Vb[sel], s['A'][frames])
        out[f"rom_{spec['name']}"] = np.linalg.norm(U, axis=2)
    with ng.TaskManager():
        fe = rx.FE(ng, man['mesh_path'], man['mesh_sha256'], k=man['velocity_order'], qorder=man['quadrature_order'])
        fe.rule()
        ref = []
        want = set(np.round(out['t'], 9))
        for t, x in rx.recorded(case['run_dir'], case['stage']):
            if round(t, 9) in want:
                u, _ = fe.eval_v(x[:fe.nv]); ref.append(np.linalg.norm(u[sel], axis=1))
    out['reference'] = np.array(ref)
    np.savez_compressed(a.out, **out)
    print('frames', len(out['t']), 'points', len(pts))


if __name__ == '__main__':
    main()
