"""Exploration (local GB10, NOT a result): rho of lattice rules at q=256, M=1088 on EVERY per-step state
the deployed lat64 EQ query visits (chol+clip+lamcarry+pred2, gtol 1e-3), split into k=0 (initial fit)
and k>=1 (evolved), for a few params_draw(0,128) trajectories incl. the two worst-w0 ones (18, 22).
Decides whether an 'exact first step' arm is worth cluster time. These trajectories are NOT used by any
certificate of this lane (the lane's populations come from a fresh seed).

    jaxrun python evolved_lattice_rho.py L traj...
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import engines as e
import arms as A
import topfix as TF
import ladder as LD
import hops as H
import hfast as HF

ROOT = Path(__file__).resolve().parents[3]
ck = pickle.load(open(ROOT / 'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl', 'rb'))
params = jax.tree_util.tree_map(jnp.asarray, LD.host(ck['params']))
Zold = np.asarray(ck['Z_tr'])
K = Zold.shape[1]
R = int(np.asarray(ck['params']['h_lin']).shape[1])
q, M = 256, 1088
L = int(sys.argv[1])
traj = [int(x) for x in sys.argv[2:]]
C = jnp.asarray(np.load(ROOT / 'experiments/b-panel/inputs/directions_qtd02.npz')['C'])[:, :q]
bank = A.CoordBank(params, K, R)
head = TF.corrected_head(params, C, K)
Zsub = Zold[::max(1, len(Zold) // 8192)]
cold = A.build_cold(bank, head, np.concatenate((Zsub, np.zeros((len(Zsub), q))), 1), 48)[0]
trust = .01 * float(np.max(np.linalg.norm(Zold - Zold.mean(0), axis=1)))
train = e.params_draw(0, 128)
t0 = time.perf_counter()
G = bank.on_grid(L)
kx, ky, lam = H.modes_lean(L, M)
sx, sy = (jnp.asarray(a) for a in H.sine_tables(L, kx, ky))
Am = H.project_bank((G,), sx, sy, L)
rules = {}
for s in (64, L // 2):
    ij, w = H.lattice_rule(L, s)
    rules[s] = H.rule_ops(bank, L, kx, ky, ij, w)[0]
data = dict(A=Am, lam=jnp.asarray(lam), G=(G,), G5=rules[64]['G5'], Pq=rules[64]['Pq'])
tab = HF.build_tables(params, C, K, data, cold)
qf = HF.make_query(params, C, K, q, L, .005, trust, 'eq', gtol=1e-3, solver='chol', clip=True, lam_carry=True,
                   predictor='quad')
hv = jax.jit(jax.vmap(head))
out = dict(L=L, trajectories=traj, per_traj={})
for i in traj:
    ph = train[i]
    v = qf(jnp.asarray(e.initial(L, ph)), float(ph[4]), data, cold, tab)
    st = np.asarray(hv(jnp.asarray(np.asarray(v[7]))))
    tg = H.dense_targets((G,), st, sx, sy, L, chunk=8)
    row = dict(iterations=int(np.sum(np.asarray(v[1]))))
    for s, ops in rules.items():
        r = H.rho(ops['Pq'], H.sampled_advection(ops['G5'], st, L), tg)
        row[f'lat{s}'] = dict(k0=float(r[0]), evolved_max=float(r[1:].max()), evolved_argmax_k=int(r[1:].argmax()) + 1,
                              evolved_median=float(np.median(r[1:])), k1=float(r[1]), k2=float(r[2]))
    out['per_traj'][i] = row
    print(i, json.dumps(row), flush=True)
out['seconds'] = time.perf_counter() - t0
(Path(__file__).parent / f"evolved_lattice_rho_L{L}.json").write_text(json.dumps(out, indent=1) + '\n')
