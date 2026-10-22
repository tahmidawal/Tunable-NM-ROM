"""imp-fronts training smoke: gates + paired generator timing (DESIGN Amendments 11 and 15).

    python smoke_datagen.py <train2d dir> <incumbent burgers2d_film.py> <arm> <out.json> --budget-s S

In one process on one GPU:
  1. gates: fid = arm_gate.check_dev6() (C-B3 on dev6 + PS self-convergence); cov = arm_gate.c_b4() on both complete
     parameter tables;
  2. GPU burn-in: one UNTIMED 64-trajectory chunk through each generator (compilation and warm-up; the timed chunks
     reuse the same jitted rollouts and the same chunk shape, so nothing recompiles);
  3. paired timing (Amendment 15): the incumbent generator (sha256 c521b36a..., the K1 generator) and the arm generator
     on the SAME 576-trajectory seed-0 population and the same first 576 trajectories of the seed-1000 population, in
     the order inc/seed0, arm/seed0, inc/seed1000, arm/seed1000; each timed pass = rollout + the per-step residual check
     (cov/incumbent: upwind Newton residual; fid: finiteness sentinel) + device->host copy, in chunks of 64 (Stage A /
     Stage B GEN_CHUNK). Nothing is stored. Newton/sentinel statistics are recorded for every timed trajectory and the wall seconds of
     every chunk are persisted (chunk_seconds arrays, Amendment 17).
Writes <out.json> and exits 1 unless `passed` (gates passed, every timed pass complete inside the budget, all states
finite, every residual <= 1e-8). The timing record carries the device kind and SLURM_JOB_ID (provenance check in
phase_times.py).
"""
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

for v in ('N', 'K', 'R', 'SEED', 'SEED0', 'STEPS', 'N_TRAIN', 'N_VAL', 'N_TRAJ', 'N_TEST', 'HIDDEN', 'N_LAYERS',
          'NEWTON_ITERS', 'LIN_TOL', 'LIN_MAXITER', 'EXTRA_SEED', 'EXTRA_TRAJ'):
    os.environ.pop(v, None)
args = [a for a in sys.argv[1:] if a != '--budget-s']
T2, INC, ARM, OUT = Path(args[0]), Path(args[1]), args[2], Path(args[3])
BUDGET = float(sys.argv[sys.argv.index('--budget-s') + 1])
INC_SHA = 'c521b36a9bae5471f0f3c1644d03451109c84688cf3808f9c5ea6ffea18493e4'
CHUNK, N_NODES, M_TIMED = 64, 256, 576
sys.path.insert(0, str(T2 / 'stageA/deps'))
import numpy as np                                   # noqa: E402
import jax                                           # noqa: E402
import jax.numpy as jnp                              # noqa: E402
import burgers2d_film as bf                          # noqa: E402  (the staged ARM generator)
import arm_gate                                      # noqa: E402

T_START = time.time()
assert hashlib.sha256(INC.read_bytes()).hexdigest() == INC_SHA, 'incumbent generator sha256 mismatch'
assert hashlib.sha256((T2 / 'stageA/deps/burgers2d_film.py').read_bytes()).hexdigest() != INC_SHA, 'arm tree has the incumbent generator'
spec = importlib.util.spec_from_file_location('bf_incumbent', INC)
bfi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bfi)
assert os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest'
dev = jax.devices()[0]
rec = dict(schema='imp-fronts-datagen/2', arm=ARM, backend=jax.default_backend(),
           device=getattr(dev, 'device_kind', str(dev)), job_id=os.environ.get('SLURM_JOB_ID'),
           host=os.uname().nodename, n_nodes=N_NODES, chunk=CHUNK, budget_s=BUDGET,
           arm_generator_sha256=hashlib.sha256((T2 / 'stageA/deps/burgers2d_film.py').read_bytes()).hexdigest(),
           incumbent_generator_sha256=INC_SHA, errors=[])
print(f'jax_backend={rec["backend"]} device={rec["device"]} job={rec["job_id"]} arm={ARM}', flush=True)

# ---------------------------------------------------------------- 1. gates
t = time.time()
if ARM == 'fid':
    g = arm_gate.check_dev6()
    g.pop('u128')
    gate_ok = bool(g['C_B3_pass'] and g['self_conv_pass'])
else:
    g = arm_gate.c_b4(bf.sample_params)
    gate_ok = bool(g['passed'])
rec['gate'] = dict(g, passed=gate_ok, seconds=time.time() - t)
print(f'gate ({ARM}) passed={gate_ok} [{time.time() - t:.0f}s]', flush=True)


def make(mod):
    ro, res_fn = mod.make_rollout(N_NODES)
    chk = jax.jit(jax.vmap(lambda u1, u0, nu_: jnp.linalg.norm(res_fn(u1, u0, nu_)) / jnp.linalg.norm(u0)))
    return mod, ro, chk


def run(gen, pop, idx):
    """One pass over trajectories idx of population pop (seed, m) in chunks of 64; returns stats dict."""
    mod, ro, chk = gen
    cx, cy, w, a, nu, _ = mod.sample_params(seed=pop[0], m=pop[1])
    rr_all, cc_all, finite, done, chunk_s = [], [], True, 0, []
    t0 = time.time()
    for s in range(0, len(idx), CHUNK):
        if time.time() - T_START > BUDGET:
            break
        tc = time.time()
        ii = idx[s:s + CHUNK]
        U0 = np.stack([mod.blob_ic(N_NODES, cx[i], cy[i], w[i], a[i]) for i in ii])
        nu_j = jnp.asarray(nu[ii])
        snaps, res = ro(jnp.asarray(U0), nu_j)
        rr = np.asarray(jnp.max(res, axis=0))
        cc = np.max(np.stack([np.asarray(chk(snaps[k + 1], snaps[k], nu_j)) for k in range(mod.NUM_STEPS)]), axis=0)
        sn = np.asarray(snaps)
        finite &= bool(np.isfinite(sn).all())
        rr_all += rr.tolist(); cc_all += cc.tolist(); done += len(ii)
        del snaps, sn
        chunk_s.append(time.time() - tc)
    sec = time.time() - t0
    q = lambda x: dict(max=float(np.max(x)), p99=float(np.percentile(x, 99)), median=float(np.median(x))) if x else None
    ok = bool(done == len(idx) and finite and np.isfinite(cc_all).all() and np.isfinite(rr_all).all()
              and max(cc_all) <= 1e-8 and max(rr_all) <= 1e-8)
    return dict(seed=pop[0], population_m=pop[1], units=done, requested=len(idx), complete=done == len(idx),
                seconds=sec, chunk_seconds=chunk_s, finite=finite, rollout_rel_res=q(rr_all), independent_residual=q(cc_all), ok=ok)


# ---------------------------------------------------------------- 2. burn-in, 3. paired timing
gens = dict(incumbent=make(bfi), arm=make(bf))
idx = np.arange(M_TIMED)
t = time.time()
for k, gen in gens.items():
    run(gen, (0, 576), idx[:CHUNK])                                      # untimed burn-in chunk
rec['burn_in_seconds'] = time.time() - t
rec['timed'] = {}
for pname, pop in (('seed0', (0, 576)), ('seed1000', (1000, 4032))):
    for k in ('incumbent', 'arm'):
        r = run(gens[k], pop, idx)
        rec['timed'][f'{k}/{pname}'] = r
        print(f"{k}/{pname}: {r['units']}/{r['requested']} in {r['seconds']:.1f}s ok={r['ok']} "
              f"resid max {r['independent_residual'] and r['independent_residual']['max']}", flush=True)
rec['seconds_total'] = time.time() - T_START
rec['timing_complete'] = all(r['complete'] for r in rec['timed'].values())
rec['passed'] = bool(gate_ok and rec['timing_complete'] and all(r['ok'] for r in rec['timed'].values())
                     and rec['backend'] == 'gpu')
if rec['backend'] != 'gpu':
    rec['errors'].append('backend is not gpu: timing invalid')
OUT.write_text(json.dumps(rec, indent=1, default=float) + '\n')
print('DATAGEN', 'PASSED' if rec['passed'] else 'NOT PASSED', flush=True)
sys.exit(0 if rec['passed'] else 1)
