"""Local smoke for the EVALUATION path. Sub-minute, tiny mesh, local GB10.

Gate 1 is the one that matters: run the incumbent checkpoint through the arm
machinery this cell uses, on the archived L = 64 operators, and require it to
reproduce the consolidated saved Burgers case and to agree with the incumbent
`accuracy_paths.make_rom`. That makes the evaluation path the RETAINED solver
rather than a re-implementation of it, which is what licenses comparing new
checkpoints against abl01's numbers at all.

Gate 2 proves a checkpoint written by `train.emit` is a drop-in for that same
machinery: bank, empirical quadrature, cold initializer and complete query all
build and run from it.

The stronger cross-job gate -- `incumbent_eq` reproducing abl01's `a_neural_eq`
on all six cases to 1e-9 -- needs the 256-interval operators and the refined
reference, so it runs INSIDE the evaluation job, where `evaluate.py` records it
under `gates.incumbent_reproduces_abl01`. Gate 3 here only pins the target
numbers that gate will be held to.
"""
import json
import pickle
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[2]
for d in ('experiments/mr-burgers2d', 'experiments/head-ablation',
          'experiments/separable-decoder', 'experiments/b-head-train'):
    sys.path.insert(0, str(ROOT / d))

import engines as e            # noqa: E402
import accuracy_paths as ap    # noqa: E402
import arms as A               # noqa: E402
import evaluate as EV          # noqa: E402

CK = ROOT / 'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'
FIX = ROOT / 'consolidated/fixtures/burgers'
RAW = ROOT / ('consolidated/evidence/worktrees/2026-09-07-mr-burgers2d/experiments/'
              'mr-burgers2d/runs/accuracy09/archive/out/result.json')
ABL = ROOT / 'experiments/head-ablation/artifacts/abl01/result.json'


def main():
    assert jax.default_backend() == 'gpu', jax.default_backend()
    print('jax_backend=gpu', flush=True)
    out = {}
    t0 = time.perf_counter()
    ck = pickle.load(open(CK, 'rb'))
    params = jax.tree_util.tree_map(jnp.asarray, ck['params'])
    Zold = np.asarray(ck['Z_tr'])
    K, R = int(Zold.shape[1]), int(np.asarray(ck['params']['h_lin']).shape[1])
    raw = json.loads(RAW.read_text())
    cfg = raw['config']
    setup = next(r for r in raw['mesh_setup'] if (r['intervals'], r['model']) == (64, 'frozen'))

    # ---- gate 1: the retained solver ---------------------------------------
    L = 64
    arch = {k: jnp.asarray(v) for k, v in np.load(FIX / 'operators.npz').items()}
    bank = A.CoordBank(params, K, R)
    G = bank.on_grid(L)
    data = dict(A=arch['A'], lam=arch['lam'], G5=arch['G5'], Pq=arch['Pq'], G=G)
    head = A.neural_head(params)
    Hrot = A.sc.head(params, arch['candidate_Z']) @ arch['cold_R'].T
    cold = (arch['cold_xy'], arch['cold_w'], arch['cold_Q'], arch['cold_R'],
            Hrot, jnp.sum(Hrot * Hrot, 1), arch['candidate_Z'])
    trust = setup['trust_radius']
    assert abs(trust - EV.radius(Zold)) < 1e-12
    query = A.make_query(head, K, L, cfg['dt'], trust, 'eq', linear='gj', **cfg['strict'])
    phys = raw['physical_cases'][0]
    u0 = jnp.asarray(e.initial(L, phys))
    value = jax.device_get(query(u0, jnp.asarray(phys[4]), data, cold))
    fields = value[0]
    saved = np.load(FIX / 'expected.npz')
    rel = float(np.linalg.norm(fields - saved['fields']) / np.linalg.norm(saved['fields']))
    rel_z = float(np.linalg.norm(value[7] - saved['internal_latents'])
                  / np.linalg.norm(saved['internal_latents']))
    ref = jax.device_get(ap.make_rom(params, L, cfg['dt'], trust, **cfg['strict'])(
        u0, jnp.asarray(phys[4]),
        (G, arch['A'], arch['lam'], arch['G5'], arch['Pq'], None, None, arch['candidate_Z']),
        (arch['cold_xy'], arch['cold_w'], arch['cold_Q'], arch['cold_R'], Hrot,
         jnp.sum(Hrot * Hrot, 1))))
    inc = float(np.linalg.norm(fields - ref[0]) / np.linalg.norm(ref[0]))
    out['incumbent_vs_saved_case'] = dict(relative_l2=rel, latent_relative_l2=rel_z,
                                          tolerance=1e-8)
    out['incumbent_vs_accuracy_paths'] = dict(relative_l2=inc, tolerance=1e-12)
    assert rel <= 1e-8 and rel_z <= 1e-8 and inc <= 1e-12, (rel, rel_z, inc)
    print('GATE1 saved-case', rel, 'incumbent', inc, flush=True)

    # ---- gate 2: a checkpoint written by train.emit is a drop-in ------------
    Ls = 32
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / 'ckpt_smoke.pkl'
        newp = {kk: (np.asarray(vv) if not isinstance(vv, list)
                     else [(np.asarray(w), np.asarray(b)) for w, b in vv])
                for kk, vv in ck['params'].items()}
        with open(p, 'wb') as f:
            pickle.dump(dict(params=newp, Z_tr=Zold[::len(Zold) // 512],
                             cfg=dict(btrain_arm='smoke', k=K, r=R)), f)
        d2 = pickle.load(open(p, 'rb'))
        p2 = jax.tree_util.tree_map(jnp.asarray, d2['params'])
        Z2 = np.asarray(d2['Z_tr'])
        b2 = A.CoordBank(p2, K, R)
        h2 = A.neural_head(p2)
        dat, inf = A.build_operators(b2, Ls, 4 * K, 'eq', Zcoef=Z2, m=16 * K, eq_seed=20259,
                                     candidate_cap=512, fit_states=8, head=h2)
        cld, _ = A.build_cold(b2, h2, Z2, 24)
        qn = A.make_query(h2, K, Ls, cfg['dt'], EV.radius(Z2), 'eq', linear='gj',
                          ic_budget=200, step_budget=60, gtol=1e-6)
        ph = np.asarray(raw['physical_cases'][0])
        v = jax.device_get(qn(jnp.asarray(e.initial(Ls, ph)), jnp.asarray(ph[4]), dat, cld))
        f = np.asarray(v[0])
        assert np.isfinite(f).all() and f.shape == (6, Ls + 1, Ls + 1), f.shape
        out['emitted_checkpoint_runs'] = dict(
            shape=list(f.shape), eq_relative_fit=inf['eq_relative_fit'],
            max_step_stationarity=float(np.max(v[8])), ic_reason=int(v[6]),
            field_norm=float(np.linalg.norm(f)))
    print('GATE2 emitted checkpoint runs', flush=True)

    # ---- gate 3: the targets the in-job gate will be held to ---------------
    abl = json.loads(ABL.read_text())
    want = {r['case']: r['error']['fixed_initial_max'] for r in abl['invocations']
            if r['intervals'] == 256 and r['name'] == 'a_neural_eq' and r['rep'] == 0}
    rec = next(r for r in abl['reconstruction']
               if r['intervals'] == 256 and r['arm'] == 'a_neural_eq')
    out['abl01_targets'] = dict(
        a_neural_eq_fixed_initial_max={str(k): v for k, v in sorted(want.items())},
        worst_reference_percent=100 * max(want.values()),
        worst_bank_projection_percent=100 * rec['worst_bank_projection'],
        worst_best_found_percent=100 * rec['worst_best_found'],
        note='the evaluation job must reproduce the per-case column to 1e-9 relative')
    out['seconds'] = time.perf_counter() - t0
    print('GATE3 abl01 targets', out['abl01_targets']['worst_best_found_percent'], flush=True)

    if len(sys.argv) > 1:
        Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
        Path(sys.argv[1]).write_text(json.dumps(out, indent=2, default=float) + '\n')
    print('SMOKE EVAL OK', round(out['seconds'], 1), flush=True)


if __name__ == '__main__':
    main()
