"""jcp-speed3d FOM / table gates (DESIGN A1.9, A1.10, A2.4):

  G6  lane fomcore == E0 refcore (bitwise lattice output) at L for F-UB, F-UC, F-CC on one case
  G6v fomcore F-UB (dt 0.005, tolerances 1e-8 / 1e-9) == vendor common.make_fom(65, 0.005, 1e-8, 1e-9) to 1e-12
  K8  F-CC differs from F-UB by > 1e-6 (relative to n0)
  G7  psjax == pscore.make_ps run to 1e-12; half-step control differs by > 1e-9
  G5  bigtables at n = 65 == tables.build_tables; K7 (n = 66 differs)

    python test_fom.py --out checks/test_fom-<where>.json [--L 64] [--skip-g5]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

import bigtables as BT
import c3common as C3
import fomcore as FC
import fomrun as FR
import psjax

E0 = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references')


def load_e0_refcore():
    sys.path.insert(0, str(E0))
    spec = importlib.util.spec_from_file_location('e0_refcore', E0 / 'refcore.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--L', type=int, default=64)
    ap.add_argument('--skip-g5', action='store_true')
    ap.add_argument('--skip-e0', action='store_true', help='cluster: the E0 lane source is not staged')
    a = ap.parse_args()
    rep = dict(backend=jax.default_backend(), checks={})
    print(f'jax_backend={jax.default_backend()}', flush=True)
    tab = C3.C.table(923801, 64)
    j = 0
    L = a.L
    u = jnp.asarray(C3.C.initial_interior(L + 1, tab, j))
    nu = float(tab['nu'][j])
    n0 = float(np.linalg.norm(np.asarray(u)))
    foms = FR.FOMs()
    out = {}
    for fam, dt in (('UB', .01), ('UC', .0125), ('CC', .0125)):
        t = time.perf_counter()
        F, info = foms.run(('fd', fam, L, dt, 0), u, nu)
        out[fam] = np.asarray(F)
        rep['checks'][f'run_{fam}'] = dict(seconds=time.perf_counter() - t, converged=foms.converged(('fd', fam, L, dt, 0), info))
        print(fam, rep['checks'][f'run_{fam}'], flush=True)
    if not a.skip_e0:
        RC = load_e0_refcore()
        for fam, dt in (('UB', .01), ('UC', .0125), ('CC', .0125)):
            adv, sch = FR.FAM[fam]
            o = RC.make_solver(3, L, adv=adv)(u, nu, dt, sch, 1e-8, 1e-9)
            ref = RC.to_lattice(np.asarray(o['out']), L, 3)
            rep['checks'][f'G6_{fam}'] = bool(np.array_equal(ref, out[fam]))
        print('G6', {k: v for k, v in rep['checks'].items() if k.startswith('G6')}, flush=True)
    if L == 64:
        Fv, _, _ = C3.C.make_fom(65, .005, 1e-8, 1e-9)(u, nu)
        Fl, _ = foms.run(('fd', 'UB', 64, .005, 0), u, nu)
        lat = np.asarray(FR.lattice_index(64))
        Fv = np.asarray(Fv).reshape(6, 63, 63, 63)[:, lat][:, :, lat][:, :, :, lat].reshape(6, -1)
        rel = float(np.abs(Fv - np.asarray(Fl)).max() / np.abs(Fv).max())
        rep['checks']['G6v'] = dict(rel=rel, pass_=rel <= 1e-12)
        print('G6v', rel, flush=True)
    k8 = float(max(np.linalg.norm(out['CC'][k] - out['UB'][k]) for k in range(1, 6)) / n0)
    rep['checks']['K8'] = dict(diff=k8, fires=k8 > 1e-6)
    g7 = psjax.gate(tab)
    rep['checks']['G7'] = g7
    print('K8', k8, 'G7', g7, flush=True)
    if not a.skip_g5:
        bank = C3.load_bank('M2')
        tb = C3.Tables(65, bank, [512], log=print)
        rep['checks']['G5_M2_512'] = BT.check65(bank['params'], bank['T'], 512, tb, log=print)
        print('G5', rep['checks']['G5_M2_512'], flush=True)
    json.dump(rep, open(a.out, 'w'), indent=1, default=float)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
