"""imp-fronts smoke (DESIGN Amendment 11): full-size training-data generation of one staged arm, timed per population.

    python smoke_datagen.py <train2d dir> <arm> <out.json> [--budget-s S]

Generates the two training populations exactly as the pipeline draws them (seed 0 x 576 = Stage A's and Stage B's
canonical draw, seed 1000 x 4032 = Stage B's appended draw) through the STAGED generator, in chunks of 64 (Stage A /
Stage B GEN_CHUNK), with Stage A build_data's per-chunk work: rollout, the independent per-step residual check
(cov: the upwind Newton residual; fid: the finiteness sentinel) and the device->host transfer. Nothing is stored.
The timing therefore matches the definition of K1's Stage A `data.seconds` (generation + residual check, no
projection), which phase_times.py uses as the K1 reference rate.
  cov: Newton relative-residual statistics per population (max of rollout rel_res and of the independent check,
       median / p99 / max over trajectories), finiteness; C-B4 on both tables (arm_gate.c_b4).
  fid: finiteness; the arm gate's C-B3 on dev6 + PS self-convergence (arm_gate.check_dev6, timed separately).
--budget-s: stop starting new chunks after S seconds (records partial counts; a partial record is not a certificate).
"""
import json
import os
import sys
import time
from pathlib import Path

for v in ('N', 'K', 'R', 'SEED', 'SEED0', 'STEPS', 'N_TRAIN', 'N_VAL', 'N_TRAJ', 'N_TEST', 'HIDDEN', 'N_LAYERS',
          'NEWTON_ITERS', 'LIN_TOL', 'LIN_MAXITER', 'EXTRA_SEED', 'EXTRA_TRAJ'):
    os.environ.pop(v, None)
T2, ARM, OUT = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
BUDGET = float(sys.argv[sys.argv.index('--budget-s') + 1]) if '--budget-s' in sys.argv else float('inf')
sys.path.insert(0, str(T2 / 'stageA/deps'))
import numpy as np                                   # noqa: E402
import jax                                           # noqa: E402
import jax.numpy as jnp                              # noqa: E402
import burgers2d_film as bf                          # noqa: E402  (the staged arm generator)
import arm_gate                                      # noqa: E402

T_START = time.time()
dev = jax.devices()[0]
assert os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest'
rec = dict(arm=ARM, backend=jax.default_backend(), device=getattr(dev, 'device_kind', str(dev)), n_nodes=256, chunk=64,
           budget_s=BUDGET, populations={})
print(f'jax_backend={jax.default_backend()} device={rec["device"]} arm={ARM}', flush=True)
if ARM == 'fid':
    t = time.time()
    g = arm_gate.check_dev6()
    g.pop('u128')
    rec['C_B3'] = dict(g, seconds_total=time.time() - t)
    print(f"C-B3 worst {g['worst_err_vs_E0']:.3e} self-conv {g['worst_self_conv']:.3e} pass "
          f"{g['C_B3_pass'] and g['self_conv_pass']} [{time.time() - t:.0f}s]", flush=True)
else:
    rec['C_B4'] = arm_gate.c_b4(bf.sample_params)
    print(f"C-B4 passed {rec['C_B4']['passed']}", flush=True)

n = 256
rollout, res_fn = bf.make_rollout(n)
chk = jax.jit(jax.vmap(lambda u1, u0, nu_: jnp.linalg.norm(res_fn(u1, u0, nu_)) / jnp.linalg.norm(u0)))
for name, seed, m in (('seed0', 0, 576), ('seed1000', 1000, 4032)):
    cx, cy, w, a, nu, _ = bf.sample_params(seed=seed, m=m)
    per_traj_rollout, per_traj_check, done, finite = [], [], 0, True
    t0 = time.time()
    for s in range(0, m, 64):
        if time.time() - T_START > BUDGET:
            break
        e = min(s + 64, m)
        U0 = np.stack([bf.blob_ic(n, cx[i], cy[i], w[i], a[i]) for i in range(s, e)])
        nu_j = jnp.asarray(nu[s:e])
        snaps, res = rollout(jnp.asarray(U0), nu_j)
        rr = np.asarray(jnp.max(res, axis=0))
        cc = np.max(np.stack([np.asarray(chk(snaps[k + 1], snaps[k], nu_j)) for k in range(bf.NUM_STEPS)]), axis=0)
        snaps_np = np.asarray(snaps)                                          # device -> host, as build_data
        finite &= bool(np.isfinite(snaps_np).all())
        per_traj_rollout += rr.tolist(); per_traj_check += cc.tolist(); done = e
        del snaps, snaps_np
    sec = time.time() - t0
    q = lambda x: dict(max=float(np.max(x)), p99=float(np.percentile(x, 99)), median=float(np.median(x))) if x else None
    rec['populations'][name] = dict(seed=seed, m=m, done=done, complete=done == m, seconds=sec,
                                    s_per_traj=sec / max(done, 1), finite=finite,
                                    rollout_rel_res=q(per_traj_rollout), independent_residual=q(per_traj_check),
                                    residual_le_1e8=bool(done and np.isfinite(per_traj_check).all()
                                                         and max(per_traj_check) <= 1e-8
                                                         and max(per_traj_rollout) <= 1e-8))
    print(f"{name}: {done}/{m} trajectories in {sec:.0f}s ({sec / max(done, 1):.3f} s/traj) finite {finite} "
          f"residual max {rec['populations'][name]['independent_residual']}", flush=True)
rec['seconds_total'] = time.time() - T_START
rec['complete'] = all(p['complete'] for p in rec['populations'].values())
rec['passed'] = bool(rec['complete'] and all(p['finite'] and p['residual_le_1e8'] for p in rec['populations'].values())
                     and (rec['C_B3']['C_B3_pass'] and rec['C_B3']['self_conv_pass'] if ARM == 'fid'
                          else rec['C_B4']['passed']))
OUT.write_text(json.dumps(rec, indent=1, default=float) + '\n')
print('DATAGEN', 'PASSED' if rec['passed'] else 'NOT PASSED', f"complete={rec['complete']}", flush=True)
