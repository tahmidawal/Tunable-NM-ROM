"""Gate 1 of `smoke_xm.py` alone, without asserting: prints the relative difference of the
retained q = 0 path against the consolidated audited fixture at 64 intervals, plus the GPU
load at the time, so a contention-induced deviation can be told from a code one.
    python smoke_gate1.py <out.json>
"""
import json, pickle, subprocess, sys, time
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
ROOT = Path(__file__).resolve().parents[2]
for sub in ('experiments/b-qxm', 'experiments/q-ridge', 'experiments/b-ladder-top',
            'experiments/cheap-corrections', 'experiments/head-ablation', 'experiments/mr-burgers2d'):
    sys.path.insert(0, str(ROOT / sub))
import engines as e, arms as A, ladder as LD, ridge as RG  # noqa: E402
CK = ROOT / 'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'
FIX = ROOT / 'consolidated/fixtures/burgers'
RAW = ROOT / ('consolidated/evidence/worktrees/2026-09-07-mr-burgers2d/experiments/'
              'mr-burgers2d/runs/accuracy09/archive/out/result.json')
rel = lambda a, b: float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(np.asarray(b)))
load = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader'],
                      capture_output=True, text=True).stdout.strip()
ck = pickle.load(open(CK, 'rb'))
params = jax.tree_util.tree_map(jnp.asarray, ck['params'])
K, R = np.asarray(ck['Z_tr']).shape[1], int(np.asarray(ck['params']['h_lin']).shape[1])
raw = json.loads(RAW.read_text()); cfg = raw['config']; L, dt = 64, cfg['dt']
arch = {k: jnp.asarray(v) for k, v in np.load(FIX / 'operators.npz').items()}
bank = A.CoordBank(params, K, R); G = bank.on_grid(L)
data0 = dict(A=arch['A'], lam=arch['lam'], G5=arch['G5'], Pq=arch['Pq'], G=G)
setup = next(r for r in raw['mesh_setup'] if (r['intervals'], r['model']) == (64, 'frozen'))
phys = raw['physical_cases'][0]
u0 = jnp.asarray(e.initial(L, phys)); nu = jnp.asarray(phys[4])
saved = np.load(FIX / 'expected.npz')
C0 = jnp.zeros((R, 0)); Zaug0 = np.asarray(arch['candidate_Z'])
Hrot = jax.jit(jax.vmap(LD.corrected_head(params, C0, K)))(jnp.asarray(Zaug0)) @ arch['cold_R'].T
cold0 = (arch['cold_xy'], arch['cold_w'], arch['cold_Q'], arch['cold_R'], Hrot, jnp.sum(Hrot * Hrot, 1), jnp.asarray(Zaug0))
fn = RG.make_query(params, C0, K, 0, L, dt, setup['trust_radius'], 'eq', 0., linear='gj', **cfg['strict'])
out = []
for rep in range(3):
    v = jax.device_get(fn(u0, nu, data0, cold0))
    out.append(dict(rep=rep, relative_l2=rel(v[0], saved['fields']), latent_relative_l2=rel(v[7], saved['internal_latents']),
                    iterations=np.asarray(v[1]).tolist()[:10], ic_iterations=int(v[5])))
    print(out[-1], flush=True)
res = dict(gpu_load_at_start=load, reps=out, saved_iterations_first10=None)
Path(sys.argv[1]).write_text(json.dumps(res, indent=2) + '\n')
print('GATE1', load, [round(o['relative_l2'], 16) for o in out])
