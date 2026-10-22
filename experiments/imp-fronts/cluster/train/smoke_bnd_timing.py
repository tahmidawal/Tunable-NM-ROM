"""imp-fronts SM-c: paired incumbent vs arm-(c) ('sine_even') timing for the training projection (DESIGN Amendments
19-21). GPU, one process.

    python smoke_bnd_timing.py <staged bnd train2d dir> <out.json> --budget-s S [--sub 1200 --full 400 --log 100]

Uses the STAGED bnd tree's smoothtrain.train() (its deps: the arm's sep_common, which runs the incumbent code path
whenever 'arch' is absent) at the production sizes (16,384 states x 254^2 interior points of the 256-node grid,
K = 16, R = 512, n_ff = 128, g_hidden = 1024, h_hidden = 256, p_sub = 4,096, the recipe's lr / wd / lam_orth / EMA).
Data are a fixed synthetic N(0, 0.1^2) matrix: step cost does not depend on data values, and generating the real
576 trajectories would cost ~ 125 s of the smoke for no timing information (data generation is taken from K1's record).
  1. burn-in: an untimed incumbent train() of 200 steps (50 full);
  2. timed train() calls, order incumbent, arm: S_sub point-subsampled steps then S_full full-grid steps, logged every
     L steps; repetitions = the per-interval seconds / L (persisted). Sub rate = median over the intervals after the
     first log point (the first interval holds step_sub's compilation); full rate = median over the full-phase
     intervals after the first (which holds step_full's compilation). Compilation = first sub log time (step 1) and
     first full interval minus L x the full rate.
  3. Stage B extraction as sep_coeff_extract does it: per bank (arm and incumbent, seed-1 init draw) the features on the
     254^2 interior, Gram and Cholesky built once (timed), then 12 timed PROJ_CHUNK = 256-state projection chunks with
     _project's formula, after an untimed build and a sustained 20 s projection burn-in; production-weighted estimate =
     build + 514 chunks (131,072 picked + 408 test states).
Writes <out.json>; exits 1 if anything is missing, non-finite, or the budget ran out.
"""
import json
import os
import sys
import time
from pathlib import Path

T2, OUT = Path(sys.argv[1]), Path(sys.argv[2])
arg = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
BUDGET = arg('--budget-s', 1e9)
S_SUB, S_FULL, L = arg('--sub', 1200), arg('--full', 400), arg('--log', 100)
for v in ('N', 'SEED', 'STEPS', 'N_TRAIN', 'N_VAL'):
    os.environ.pop(v, None)
sys.path.insert(0, str(T2 / 'stageA'))
T_START = time.time()
import numpy as np                     # noqa: E402
import jax                             # noqa: E402
import jax.numpy as jnp                # noqa: E402
import smoothtrain as sm               # noqa: E402  (staged bnd smoothtrain; deps/ = the arm's sep_common)
sc = sm.sc
assert os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest'
assert sm.RECIPE['arch'].get('arch') == 'sine_even', 'not the bnd tree'
dev = jax.devices()[0]
rec = dict(schema='imp-fronts-smc-timing/1', backend=jax.default_backend(), device=getattr(dev, 'device_kind', str(dev)),
           job_id=os.environ.get('SLURM_JOB_ID'), small=('--small' in sys.argv), sizes=dict(S=16384, n_pts=254 ** 2, sub=S_SUB, full=S_FULL, log=L),
           errors=[])
print(f"jax_backend={rec['backend']} device={rec['device']} job={rec['job_id']}", flush=True)
R = sm.RECIPE
SMALL = '--small' in sys.argv           # local plumbing test only (n = 128, 512 states); never for the certificate
n = 128 if SMALL else 256
coords = sm.grid_coords(n)[sm.interior_indices(n)]
U = (0.1 * np.random.default_rng(0).standard_normal((512 if SMALL else R['max_snaps'], coords.shape[0]))).astype(np.float64)
nb = sm.neighbours(n)
arch_inc = dict(R['arch'], ff_scale=4.0)
arch_inc.pop('arch')
arch_arm = dict(R['arch'], ff_scale=4.0)


def train(arch, steps, full_last, log_every):
    _, _, info = sm.train(jax.random.PRNGKey(1), coords, U, nb, 1. / (n - 1), R['K'], R['R'], steps, R['lr'],
                          R['lam_orth'], R['wd'], R['p_sub'], R['ema_decay'], full_last, log_every=log_every,
                          tag='timing', **arch)
    return info


def budget_ok(tag):
    if time.time() - T_START > BUDGET:
        rec['errors'].append(f'budget exhausted before {tag}')
        return False
    return True


t = time.time()
train(arch_inc, 200, 50, 100)
rec['burn_in_seconds'] = time.time() - t
rec['train'] = {}
for name, arch in (('incumbent', arch_inc), ('arm', arch_arm)):
    if not budget_ok(name):
        break
    t = time.time()
    info = train(arch, S_SUB + S_FULL, S_FULL, L)
    h = {e['step']: e['seconds'] for e in info['history']}
    sub_iv = [(h[s + L] - h[s]) / L for s in range(L, S_SUB, L)]
    full_iv = [(h[s + L] - h[s]) / L for s in range(S_SUB, S_SUB + S_FULL, L)]
    r_sub, r_full = float(np.median(sub_iv)), float(np.median(full_iv[1:]))
    rec['train'][name] = dict(sub_intervals_s_per_step=sub_iv, full_intervals_s_per_step=full_iv, sub_s_per_step=r_sub,
                              full_s_per_step=r_full, compile_sub_s=h[1], compile_full_s=max(0., full_iv[0] * L - r_full * L),
                              call_seconds=time.time() - t, final_rel_mse=info['final_rel_mse'])
    print(f"{name}: sub {r_sub:.5f} s/step, full {r_full:.4f} s/step, compile sub {h[1]:.1f}s full "
          f"{rec['train'][name]['compile_full_s']:.1f}s [{time.time() - t:.0f}s]", flush=True)

# Stage B extraction, mirroring sep_coeff_extract (audit codex-job-smC-1): the bank quantities (features on the interior,
# Gram, Cholesky) are built ONCE per bank (timed: build_s), then the projection (_project's formula) runs in PROJ_CHUNK =
# 256-state chunks (timed per chunk: the repetitions). Production extraction work per bank = build + N_PROJ_CHUNKS x chunk,
# N_PROJ_CHUNKS = ceil((131,072 picked + 8 x 51 test states) / 256) = 514; median_s = that production-weighted estimate.
PROJ_CHUNK, N_CHUNKS_TIMED, N_PROJ_CHUNKS = 256, 12, -(-(131072 + 8 * 51) // 256)
if budget_ok('extraction'):
    key = jax.random.PRNGKey(1)
    key, kz, kp = jax.random.split(key, 3)
    kw = dict(n_ff=128, ff_scale=4.0, g_hidden=1024, h_hidden=256, out_scale=1.0)
    banks = dict(incumbent=sc.init_separable(kp, 16, 512, **kw), arm=sc.init_separable(kp, 16, 512, arch='sine_even', **kw))
    Cj = jnp.asarray(coords)

    @jax.jit
    def project(Ub, Gb, Lb, Gr):
        B = Gb.T @ Ub.T
        C = jax.scipy.linalg.cho_solve((Lb, True), B).T
        Rres = B - Gb.T @ (Gb @ C.T)
        C = C + jax.scipy.linalg.cho_solve((Lb, True), Rres).T
        un2 = jnp.sum(Ub * Ub, axis=1)
        E = Ub - C @ Gb.T
        return C, un2, jnp.sum(E * E, axis=1)

    def build(p):
        G = sc.SeparableDecoder(p, 16, 512).feat_at(Cj)
        Gram = G.T @ G
        Lc = jnp.linalg.cholesky(Gram + 1e-12 * jnp.trace(Gram) / 512 * jnp.eye(512, dtype=G.dtype))
        Lc.block_until_ready()
        return G, Lc, Gram

    chunks = [jnp.asarray(U[(k * PROJ_CHUNK) % len(U):(k * PROJ_CHUNK) % len(U) + PROJ_CHUNK]) for k in range(N_CHUNKS_TIMED)]
    rec['extract'] = {}
    built = {}
    for name, p in banks.items():                      # untimed warm-up build (compilation of the feature path)
        built[name] = build(p)
    G0, L0, R0 = built['incumbent']                    # sustained burn-in immediately before the timed extraction
    tb = time.time()
    while time.time() - tb < 20.0:
        project(chunks[0], G0, L0, R0)[2].block_until_ready()
    rec['extract_burn_in_s'] = time.time() - tb
    for name, p in banks.items():
        if not budget_ok(f'extraction {name}'):
            break
        t = time.time(); G, Lc, Gram = build(p); build_s = time.time() - t
        reps, fin = [], True
        for Ub in chunks:
            t = time.time(); C, un2, f2 = project(Ub, G, Lc, Gram); f2.block_until_ready(); reps.append(time.time() - t)
            fin &= bool(jnp.all(jnp.isfinite(f2)) & jnp.all(jnp.isfinite(C)))
        chunk_s = float(np.median(reps))
        rec['extract'][name] = dict(build_s=build_s, chunk_reps_s=reps, chunk_median_s=chunk_s, proj_chunk=PROJ_CHUNK,
                                    n_production_chunks=N_PROJ_CHUNKS, median_s=build_s + N_PROJ_CHUNKS * chunk_s,
                                    finite=fin)
        print(f"extraction {name}: build {build_s:.2f}s, chunk median {chunk_s * 1e3:.2f} ms -> production-weighted "
              f"{rec['extract'][name]['median_s']:.1f}s", flush=True)

ok = ((rec['backend'] == 'gpu' or SMALL) and not (SMALL and not os.environ.get('SMC_ALLOW_SMALL')) and not rec['errors'] and set(rec.get('train', {})) == {'incumbent', 'arm'}
      and set(rec.get('extract', {})) == {'incumbent', 'arm'}
      and all(np.isfinite(v['sub_s_per_step']) and np.isfinite(v['full_s_per_step']) and v['sub_s_per_step'] > 0
              and v['full_s_per_step'] > 0 for v in rec['train'].values())
      and all(v['finite'] and v['median_s'] > 0 for v in rec['extract'].values()))
rec['passed'] = bool(ok)
rec['seconds_total'] = time.time() - T_START
OUT.write_text(json.dumps(rec, indent=1, default=float) + '\n')
print('SMC-TIMING', 'PASSED' if ok else 'NOT PASSED', rec['errors'], flush=True)
sys.exit(0 if ok else 1)
